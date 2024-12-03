import numpy as np
import pandas as pd

from src.constants import PREDICTION_UPDATE_INTERVAL_IN_MINUTES
from src.algorithms.base import AnomalyDetector
from src.utils.correlation_computation import compute_correlation_np, compute_rolling_correlation_pd


class CorrelationDetector(AnomalyDetector):

    def __init__(self, config_path, **kwargs):
        super().__init__(config_path)
        self.needed_history_in_hours = self.config['detector']['window_in_hours']
        self.window_in_samples = int(self.needed_history_in_hours * 60 / PREDICTION_UPDATE_INTERVAL_IN_MINUTES)
        self.threshold = self.config['detector']['correlation_threshold']

        self.complete_series = pd.DataFrame()
        self.results = pd.DataFrame()

    def fit(self, series, **kwargs):
        self.complete_series = series

    def predict(self, series, **kwargs):

        if (series is None) or (len(series) == 0) or (not 'control' in series.columns):
            return None

        corr = compute_correlation_np(series['signal'].values, series['control'].values)
        anomaly_score = 1 - np.abs(corr)
        detection = np.abs(corr) < self.threshold

        # Append historical data
        self.complete_series.loc[series.index[-1]] = series.iloc[-1]
        # self.complete_series = pd.concat([self.complete_series, series[-1:]])
        if self.results.empty:
            self.results = pd.DataFrame({'time_index': [series.index[-1]], 'anomaly_score': [anomaly_score], 'detection': [detection]}).set_index('time_index')
        else:
            self.results.loc[series.index[-1]] = {'anomaly_score': anomaly_score, 'detection': detection}

        return {
            'time_index': series.index[-1],
            'anomaly_score': anomaly_score,
            'detection': detection,
        }

    def predict_all(self, series, **kwargs):

        if (series is None) or (len(series) == 0) or (not 'control' in series.columns):
            return None

        darts_series_df = self.pre_process(series, train=False).pd_dataframe()
        corr = compute_rolling_correlation_pd(darts_series_df, self.window_in_samples)
        abs_corr = np.abs(corr['correlation'].values)
        anomaly_score = 1 - abs_corr
        detection = abs_corr < self.threshold

        # Append historical data
        self.complete_series = pd.concat([self.complete_series, series])
        self.results = pd.DataFrame({'time_index': series.index, 'anomaly_score': anomaly_score, 'detection': detection}).set_index('time_index')

    def plot(self, title, show=True, save_path=None, **kwargs):
        return super().plot(self.complete_series, self.results, title, show, save_path)
