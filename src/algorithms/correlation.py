import numpy as np

from src.algorithms.base import AnomalyDetector
from src.utils.correlation_computation import compute_rolling_correlation_pd
from src.constants import SAMPLES_PER_HOUR


class CorrelationDetector(AnomalyDetector):

    def __init__(self, config_path, **kwargs):
        super().__init__(config_path)
        self.window = self.config['detector']['window_in_hours'] * SAMPLES_PER_HOUR
        self.threshold = self.config['detector']['correlation_threshold']

    def _fit(self, series, **kwargs):
        pass

    def _predict(self, series, **kwargs):
        darts_series_df = self.pre_process(series, train=False).pd_dataframe()

        corr = compute_rolling_correlation_pd(darts_series_df, self.window)
        abs_corr = np.abs(corr['correlation'].values)

        anomaly_score = 1 - abs_corr
        detection = abs_corr < self.threshold

        return corr.index, anomaly_score, detection
