import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
from darts import TimeSeries
from darts.ad import QuantileDetector
from scipy.ndimage import label

from src.algorithms.base import AnomalyDetector
from src.algorithms.correlation import CorrelationDetector
from src.algorithms.fft import FFTDetector
from src.algorithms.isolation_forest import IsolationForestDetector
from src.algorithms.kmeans_darts import KMeansDartsDetector
from src.algorithms.knn import KNNDetector
from src.algorithms.regressor_knn import KNNRegressorDetector
from src.algorithms.xgb_darts import XGBDartsDetector


class EnsembleDetector(AnomalyDetector):
    def __init__(self, config_dir):
        super().__init__(config_path=os.path.join(config_dir, "ensemble.yaml"))

        self.models = [
            CorrelationDetector(config_path=os.path.join(config_dir, "correlation.yaml")),
            IsolationForestDetector(config_path=os.path.join(config_dir, "isolation_forest.yaml")),
            KNNDetector(config_path=os.path.join(config_dir, "knn.yaml")),
            KMeansDartsDetector(config_path=os.path.join(config_dir, "kmeans_darts.yaml")),
            KNNRegressorDetector(config_path=os.path.join(config_dir, "regressor_knn.yaml")),
            XGBDartsDetector(config_path=os.path.join(config_dir, "xgb_darts.yaml")),
            FFTDetector(config_path=os.path.join(config_dir, "fft.yaml")),
        ]

        self.detector = QuantileDetector(high_quantile=self.config['detector']['detection_quantile'])
        # self.operation = self.config['detector']['operation']
        # self.normalization = self.config['detector']['normalization']
        self.normalization = 'Standard'
        self.win_size_before_combining_soft = self.config['detector']['smoothing_window_before_combination_soft_version']
        self.win_size_before_combining_hard = self.config['detector']['smoothing_window_before_combination_hard_version']
        self.sigmoide_temp_soft = self.config['detector']['sigmoide_temperature_soft_version']
        self.sigmoide_temp_hard = self.config['detector']['sigmoide_temperature_hard_version']
        self.win_size_after_combining = self.config['detector']['smoothing_window_after_combination']
        self.min_length_pos_processing = self.config['detector']['min_length_pos_processing']
        self.detection_threshold = self.config['detector']['detection_threshold']

    def _fit(self, series, **kwargs):
        for model in self.models:
            print("\n" + model.__class__.__name__)
            model.fit(series)

        # Train detector
        individual_results = self._get_individual_results(series, train=True)
        _, anomaly_score = self.combine_anomaly_scores(individual_results)
        self.detector.fit(TimeSeries.from_series(anomaly_score))

    def _predict(self, series, **kwargs):
        individual_results = self._get_individual_results(series, train=False)
        time_index, anomaly_score = self.combine_anomaly_scores(individual_results)

        darts_series = TimeSeries.from_series(anomaly_score)
        detections = self.detector.detect(series=darts_series)
        detections = self._posprocessing_detections(
            detections=detections.pd_series(),
            min_length=self.min_length_pos_processing
        )
        # return time_index, anomaly_score, detections
        return time_index.values, anomaly_score.values, detections.values

    def _get_individual_results(self, series, train: bool):

        statistical_data = {}
        results_dict = {}
        for model in self.models:
            model.predict(series)
            results = model.results
            if len(results) == 0:
                continue

            results_dict[model.__class__.__name__] = results
            if train:
                statistical_data[model.__class__.__name__] = self._get_statistical_data(results)
        if train:
            self.statistical_data = statistical_data

        return results_dict

    def combine_anomaly_scores(self, results: dict):

        # Find the time indexes that are common to all the results
        methods = list(results.keys())
        common_time_index = results[methods[0]].index
        for method in methods[1:]:
            result = results[method]
            common_time_index = common_time_index.intersection(result.index)

        # Align and normalize the anomaly scores
        aligned_anomalied_dict = {}
        aligned_anomalied_without_norm_dict = {}
        aligned_anomalied_smoothed_dict = {}
        for i in range(len(methods)):
            method = methods[i]
            result = results[method]
            mask = result.index.isin(common_time_index)
            anomaly_score_mask = result['anomaly_score'][mask]
            aligned_anomalied_without_norm_dict[method] = anomaly_score_mask
            if method in ['KNNDetector','KNNRegressorDetector','XGBDartsDetector']:
                window_size = self.win_size_before_combining_hard
                temperature = self.sigmoide_temp_hard
            else:
                window_size = self.win_size_before_combining_soft
                temperature = self.sigmoide_temp_soft
            
            anomaly_score_mask = self._rolling_window(
                data=anomaly_score_mask,
                window_size=window_size
            )
            aligned_anomalied_smoothed_dict[method] = anomaly_score_mask
            standardized_data = self._normalization(
                data=anomaly_score_mask,
                method=method,
                normalization=self.normalization
            )
            aligned_anomalied_dict[method] = self._normalization(
                data=standardized_data,
                method=method,
                normalization='Sigmoid',
                temperature=temperature
            )

        # Applied the operation to combine the anomaly scores
        aligned_anomalies_df = pd.DataFrame(aligned_anomalied_dict)
        combined_anomalies_df = pd.DataFrame(index=common_time_index)

        groups = [
            ['FFTDetector'],
            ['CorrelationDetector'],
            ['KNNRegressorDetector'],
            ['IsolationForestDetector', 'XGBDartsDetector'],
            ['KNNDetector', 'KMeansDartsDetector']
        ]
        # groups = [
        #     ['FFTDetector'],
        #     ['CorrelationDetector'],
        #     ['KNNRegressorDetector', 'KNNDetector', 'XGBDartsDetector'],
        #     ['IsolationForestDetector', 'KMeansDartsDetector'],
        # ]

        corrected_groups = self._corrected_groups(groups, methods)
        combined_anomalies_df['custom_3'] = self._custom_combination(
            groups=corrected_groups,
            operation_in_group='product',
            data=aligned_anomalies_df
        )

        # Smooth the combined anomaly scores
        combined_anomalies_df = combined_anomalies_df.rolling(window=self.win_size_after_combining, min_periods=1).mean()
        selected_combination = combined_anomalies_df['custom_3']

        return common_time_index, selected_combination

    # def combine_detections(self, series: pd.Series, results: dict):
    #
    #     # Find the time indexes that are common to all the results
    #     methods = list(results.keys())
    #     common_time_index = results[methods[0]]['time_index']
    #     for method in methods[1:]:
    #         result = results[method]
    #         common_time_index = common_time_index.intersection(result['time_index'])
    #
    #     # Align the detections
    #     aligned_detections_dict = {}
    #     for i in range(len(methods)):
    #         result = results[methods[i]]
    #         mask = result['time_index'].isin(common_time_index)
    #         aligned_detections_dict[methods[i]] = result['detection'][mask]
    #
    #     # Applied the operation to combine the anomaly scores
    #     aligned_detections_df = pd.DataFrame(aligned_detections_dict)
    #
    #     combined_detections_df = pd.DataFrame(index=common_time_index)
    #
    #     combined_detections_df['sum'] = aligned_detections_df.sum(axis=1).values
    #
    #     combined_detections_df['max'] = aligned_detections_df.max(axis=1).values
    #
    #     groups = [
    #         ['FFTDetector'],
    #         ['CorrelationDetector'],
    #         ['KNNRegressorDetector', 'KNNDetector', 'XGBDartsDetector'],
    #         ['IsolationForestDetector', 'KMeansDartsDetector'],
    #     ]
    #     corrected_groups = self._corrected_groups(groups, methods)
    #
    #     combined_detections_df['custom_1'] = self._voting_combination(
    #         groups=corrected_groups,
    #         data=aligned_detections_df
    #     )
    #
    #     groups = [
    #         ['FFTDetector'],
    #         ['CorrelationDetector'],
    #         ['KNNRegressorDetector'],
    #         ['IsolationForestDetector', 'XGBDartsDetector'],
    #         ['KNNDetector', 'KMeansDartsDetector']
    #     ]
    #
    #     corrected_groups = self._corrected_groups(groups, methods)
    #
    #     combined_detections_df['custom_2'] = self._voting_combination(
    #         groups=corrected_groups,
    #         data=aligned_detections_df
    #     )
    #
    #     # Smooth the combined anomaly scores
    #     combined_detections_df = combined_detections_df.rolling(window=self.win_size_after_combining, min_periods=1).mean()
    #
    #     self._plot_df(series=series, df=combined_detections_df, title='Combine detections', y_axe_title='Detections', show=True)
    #
    #     # Run the detector
    #
    #     anomaly_score = combined_detections_df['custom_1']
    #     detections = anomaly_score > self.detection_threshold
    #     detections = self._posprocessing_detections(detections=detections, min_length=self.min_length_pos_processing)
    #
    #     return {
    #         'anomaly_score': anomaly_score,
    #         'detection': detections,
    #         'time_index': common_time_index
    #     }

    def _normalization(self, data: np.array, method: str, normalization: str, temperature: int=3):

        if normalization == 'MinMax':
            min_as = self.statistical_data[method]['q_min']
            max_as = self.statistical_data[method]['q_max']
            data_normalize = (data - min_as)/(max_as - min_as)
        elif normalization == 'Standard':
            mean = self.statistical_data[method]['mean']
            std = self.statistical_data[method]['std']
            data_normalize = (data - mean)/std
        elif normalization == 'Sigmoid':
            data_normalize = 1 / (1 + np.exp(-data/temperature))
        else:
            data_normalize = data
            print('Normalization method not implemented.')

        return data_normalize

    def _get_statistical_data(self, results: dict):

        data = results['anomaly_score']
        if self.normalization == 'Standard':
            mean = data.mean()
            std = data.std()
            statistical_data = {'mean': mean, 'std': std}
        elif self.normalization == 'MinMax':
            q_min = np.quantile(data, q=0.01)
            q_max = np.quantile(data, q=0.99)
            statistical_data = {'q_min': q_min, 'q_max': q_max}
        else:
            statistical_data = {}

        return statistical_data

    def _custom_combination(self, groups: list, operation_in_group: str, data: pd.DataFrame):
        combined_anomalies = np.zeros(len(data))
        for group in groups:
            if len(group)>1:
                if operation_in_group == 'product':
                    combined_anomalies += data[group].product(axis=1)
                elif operation_in_group == 'max':
                    combined_anomalies += data[group].max(axis=1)
                elif operation_in_group == 'mean':
                    combined_anomalies += data[group].mean(axis=1)
            else:
                combined_anomalies += data[group].values[:,0]

        return combined_anomalies.values

    def _voting_combination(self, groups: list, data: pd.DataFrame):
        tmp_df = pd.DataFrame(index=data.index)
        for i,group in enumerate(groups):
            if len(group)>1:
                tmp_df[i] = data[group].sum(axis=1)>len(group)/2
            else:
                tmp_df[i] = data[group].values[:,0]

        combined_anomalies = tmp_df.max(axis=1)

        return combined_anomalies.values

    def _posprocessing_detections(self, detections: pd.Series, min_length: int = 6*2):

        # Convert the boolean Series to integers (True -> 1, False -> 0)
        binary_signal = detections.astype(int).values

        # Label connected components of 1s
        labeled_array, num_features = label(binary_signal)

        # Create a result array initialized to zeros
        result = np.zeros_like(binary_signal)

        # Loop through each feature to check its size
        for feature in range(1, num_features + 1):
            # Find the indices of the current feature
            feature_indices = np.where(labeled_array == feature)[0]
            if len(feature_indices) >= min_length:
                # Keep the ones if the feature size meets the minimum length
                result[feature_indices] = 1

        # Convert the result back to a Pandas Series with the same index
        return pd.Series(result.astype(bool), index=detections.index)

    def _corrected_groups(self, groups: list, methods: list):

        corrected_groups = []
        for group in groups:
            partial_group = set(group) & set(methods)  # Find the intersection
            if partial_group:  # If there's any overlap
                corrected_groups.append(list(partial_group))

        return corrected_groups

    def _rolling_window(self, data, window_size):
        new_data = np.array([np.mean(data[i:i + window_size]) for i in range(len(data) - window_size + 1)])
        padding = (window_size-1)
        new_data = np.pad(new_data, (0, padding), mode='edge')

        if len(data) == len(new_data):
            return new_data
        else:
            return np.concatenate([[np.nan], new_data])



    # def plot(self, series, result, title, show=True, save_path=None, **kwargs):
    #
    #     series.index = pd.to_datetime(series.index)
    #     transformed_series = pd.DataFrame({
    #                 'anomaly_score': result['anomaly_score'],
    #                 'signal': series['signal']
    #             }, index=result['time_index'])
    #
    #     if 'detection' in result.keys():
    #         transformed_series['detection'] = result['detection']
    #
    #     fig = go.Figure()
    #
    #     # Add Anomaly Score trace if available
    #     if 'anomaly_score' in result.keys():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=transformed_series.index,
    #                 y=result['anomaly_score'],
    #                 mode='lines',
    #                 name='Anomaly Score',
    #                 line=dict(color='#ff7f0e'),
    #                 yaxis='y2',
    #                 opacity=0.7
    #             )
    #         )
    #
    #     # Add Signal trace
    #     fig.add_trace(
    #         go.Scatter(
    #             x=series.index,
    #             y=series['signal'],
    #             mode='lines',
    #             name='Signal',
    #             line=dict(color='#1f77b4'),
    #             yaxis='y1'
    #         )
    #     )
    #
    #     # Add Anomalies trace if available
    #     if 'detection' in result.keys():
    #         anomalies = result['detection']
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=transformed_series.index[anomalies],
    #                 y=transformed_series['signal'][anomalies],
    #                 mode='markers',
    #                 name='Anomalies',
    #                 marker=dict(color='#d62728'),
    #                 yaxis='y1'
    #             )
    #         )
    #
    #     # Update layout with titles and labels
    #     fig.update_layout(
    #         title=title,
    #         xaxis_title='Date',
    #         yaxis=dict(title='Signal', side='right'),
    #         yaxis2=dict(title='Anomaly Score', overlaying='y', side='left'),
    #         showlegend=True,
    #         template='seaborn',
    #         height=600  # Adjust the height as necessary
    #     )
    #
    #     if show:
    #         fig.show()
    #
    #
    #     if save_path is not None:
    #         fig.write_image(save_path)
    #
    #     return fig