import numpy as np
import warnings

from src.algorithms.base import AnomalyDetector
from src.utils.correlation_computation import compute_rolling_correlation_pd
from src.constants import SAMPLES_PER_HOUR


class CorrelationDetector(AnomalyDetector):

    def __init__(self, config_path, **kwargs):
        super().__init__(config_path)
        self.window = self.config['detector']['window_in_hours'] * SAMPLES_PER_HOUR
        self.threshold = self.config['detector']['correlation_threshold']

    def pre_process(self, series, train=False):
        if 'control' not in series.columns or 'signal' not in series.columns:
            warnings.warn(
                f"Warning: Missing signals for using {self.__class__.__name__}. Skipping computation...\n",
                UserWarning
            )
            return None
        return super().pre_process(series, train)

    def _fit(self, series, **kwargs):
        pass

    def _predict(self, series, **kwargs):
        darts_series_df = self.pre_process(series, train=False)

        if darts_series_df is None:
            return None, None, None

        darts_series_df = darts_series_df.pd_dataframe()
        corr = compute_rolling_correlation_pd(darts_series_df, self.window)
        abs_corr = np.abs(corr['correlation'].values)

        anomaly_score = 1 - abs_corr
        detection = abs_corr < self.threshold

        return corr.index, anomaly_score, detection
