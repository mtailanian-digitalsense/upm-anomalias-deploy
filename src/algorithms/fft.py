import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.ad import QuantileDetector

from src.utils.fft_lib import detect_anomalies
from src.algorithms.base import AnomalyDetector


class FFTDetector(AnomalyDetector):

    def __init__(self, config_path, **kwargs):
        
        super().__init__(config_path)

        self.detection = None
        self.anomaly_scores = None
        self.ifft_parameters = self.config['detector']['ifft_parameters']
        self.local_neighbor_window = self.config['detector']['local_neighbor_window']
        self.local_outlier_threshold = self.config['detector']['local_outlier_threshold']
        self.max_region_size = self.config['detector']['max_region_size']
        self.max_sign_change_distance = self.config['detector']['max_sign_change_distance']
        self.filter_type = self.config['detector']['filter_type']

        self.detector = QuantileDetector(high_quantile=self.config['detector']['detection_quantile'])

    def _fit(self, series, **kwargs):
        pass

    def _predict(self, series, **kwargs):
        darts_series = self.pre_process(series, train=False)
        dart_series_df = darts_series.pd_dataframe()

        _, anomaly_score = detect_anomalies(
                np.squeeze(dart_series_df.values),
                max_region_size=self.max_region_size,
                local_neighbor_window=self.local_neighbor_window,
                ifft_parameters=self.ifft_parameters,
                local_outlier_threshold=self.local_outlier_threshold,
                max_sign_change_distance=self.max_sign_change_distance,
                filter_type=self.filter_type
            )
                
        anomaly_score_for_det = np.abs(anomaly_score)

        anomaly_score_test = pd.DataFrame(anomaly_score_for_det, index=series.index, columns=['signal'])

        anomaly_score_test['date'] = pd.to_datetime(anomaly_score_test.index)
        darts_result = TimeSeries.from_dataframe(
            anomaly_score_test,
            time_col='date',
            value_cols=self.signals_to_use,
            fill_missing_dates=True,
            freq='10min',
            fillna_value=series['signal'].mean()
        )

        detection = self.detector.fit_detect(darts_result)

        return series.index, anomaly_score_for_det, detection.values()[:, 0] == 1
