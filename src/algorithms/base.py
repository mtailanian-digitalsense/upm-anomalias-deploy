import yaml
import torch
import warnings
import pandas as pd
import plotly.graph_objects as go
from abc import ABC, abstractmethod

from darts import TimeSeries
from darts.ad.utils import eval_metric_from_scores, eval_metric_from_binary_prediction

from src.utils.preprocessing import Preprocessor
from src.utils.registry import step_registry


class AnomalyDetector(ABC):

    def __init__(self, config_path):

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        # elif torch.backends.mps.is_available():
        #     device = "mps"

        self.complete_series = pd.DataFrame()
        self.results = pd.DataFrame()

        self.config = self._load_config(config_path)
        self.signals_to_use = self.config['signals_to_use']
        self.pre_processor = self._initialize_preprocessor()

    @staticmethod
    def _load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _initialize_preprocessor(self):
        # Add security in case there is no preprocessing
        if 'preprocessor' not in self.config.keys() or not self.config['preprocessor'] or  not self.config['preprocessor']['steps']:
            return Preprocessor()
        steps = []
        for step_config in self.config['preprocessor']['steps']:
            step_type = step_config.pop('type')
            step_class = step_registry.get(step_type)
            if step_class:
                steps.append(step_class(**step_config))
        return Preprocessor(steps=steps)

    def pre_process(self, series, train=False):
        if not all(signal in series.columns for signal in self.signals_to_use):
            missing_signals = set(self.signals_to_use) - set(series.columns)
            self.signals_to_use = [signal for signal in self.signals_to_use if signal in series.columns]
            warnings.warn(
                f"Warning: Missing signals {missing_signals} in the series\n"
                f"Using only the available signals: {self.signals_to_use}",
                UserWarning
            )

        return self.pre_processor(series, self.signals_to_use, train)

    def fit(self, series, **kwargs):
        self.complete_series = series
        return self._fit(series, **kwargs)

    @abstractmethod
    def _fit(self, series, **kwargs):
        raise NotImplementedError

    def predict(self, series, **kwargs):
        if (series is None) or (len(series) == 0):  # or (not 'control' in series.columns):
            return

        # Append new data to self.complete_series
        new_series = series[series.index > self.complete_series.index[-1]]
        self.complete_series = pd.concat([self.complete_series, new_series])

        time_index, anomaly_score, detection = self._predict(series, **kwargs)

        if time_index is not None:
            self.results = pd.DataFrame({
                'time_index': time_index,
                'anomaly_score': anomaly_score,
                'detection': detection
            }).set_index('time_index')

    @abstractmethod
    def _predict(self, series, **kwargs):
        """
        This method should return the time index, anomaly score, and the detection, for the given series
        :param series: the series to predict
        :return: time_index, anomaly_score, detection
        """
        raise NotImplementedError

    def evaluate(self, series, results):
        gt = TimeSeries.from_times_and_values(
            series.index, series[["ground_truth"]], columns=["is_anomaly"]
        )
        score = TimeSeries.from_times_and_values(
            results['time_index'], results['anomaly_score'], columns=["anomaly_score"]
        )
        detection = TimeSeries.from_times_and_values(
            results['time_index'], results['detection'], columns=["detection"]
        )

        metric_data = {"AUC_ROC": [], "AUC_PR": []}
        for metric_name in metric_data:
            metric_data[metric_name].append(
                eval_metric_from_scores(
                    anomalies=gt,
                    pred_scores=score,
                    metric=metric_name,
                )
            )
        result = pd.DataFrame(data=metric_data, index=['anomaly_score'])

        metric_detector = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for metric_name in metric_detector:
                metric_detector[metric_name].append(
                    eval_metric_from_binary_prediction(
                        anomalies=gt,
                        pred_anomalies=detection,
                        window=1,
                        metric=metric_name,
                    )
                )
        result_detector = pd.DataFrame(data=metric_detector, index=['detection'])
        return result, result_detector

    def plot(self, title, show=True, save_path=None, **kwargs):

        series = self.complete_series
        result = self.results

        fig = go.Figure()

        # This is to avoid linking points with no data
        # new_index = pd.date_range(start=series.index[0], end=series.index[-1], freq='10min')
        # series = series.reindex(new_index)

        # Add Signal trace
        fig.add_trace(
            go.Scatter(
                x=series.index, 
                y=series['signal'], 
                mode='lines', 
                name='Signal', 
                line=dict(color='#1f77b4'),
                yaxis='y1'
            )
        )

        # Add Anomaly Score trace if available
        if 'anomaly_score' in result.keys():
            fig.add_trace(
                go.Scatter(
                    x=result.index,
                    y=result['anomaly_score'],
                    mode='lines',
                    name='Anomaly Score',
                    line=dict(color='#ff7f0e'),
                    yaxis='y2',
                    opacity=0.7
                )
            )

        # Add Anomalies trace if available
        filtered_series = series.loc[result.index, 'signal']
        if 'detection' in result.keys():
            anomalies = result['detection']
            fig.add_trace(
                go.Scatter(
                    x=result.index[anomalies],
                    y=filtered_series.values[anomalies],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='#d62728'),
                    yaxis='y1'
                )
            )

        # Update layout with titles and labels
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis=dict(title='Signal', side='right'),
            yaxis2=dict(title='Anomaly Score', overlaying='y', side='left'),
            showlegend=True,
            template='seaborn',
            height=600,  # Adjust the height as necessary
            uirevision=True
        )

        if show:
            fig.show()
            
        if save_path is not None:
            fig.write_image(save_path)
            
        return fig
