import numpy as np
import warnings
from sklearn.neighbors import KNeighborsRegressor

from src.algorithms.base import AnomalyDetector


class KNNRegressorDetector(AnomalyDetector):

    def __init__(self, config_path, **kwargs):
        super().__init__(config_path)
        self.model = KNeighborsRegressor(n_neighbors=self.config['detector']['n_neighbors'])
        self.n_sigmas_threshold = self.config['detector']['n_sigmas_threshold']

    def pre_process(self, series, train=False):
        if 'control' not in series.columns or 'signal' not in series.columns:
            warnings.warn(
                f"Warning: Missing signals for using {self.__class__.__name__}. Skipping computation...\n",
                UserWarning
            )
            return None
        return super().pre_process(series, train)

    def _fit(self, series, **kwargs):
        darts_series_df = self.pre_process(series, train=True)
        if darts_series_df is not None:
            darts_series_df = darts_series_df.pd_dataframe()
            self.model.fit(darts_series_df[['control']], darts_series_df['signal'])

    def _predict(self, series, **kwargs):
        darts_series_df = self.pre_process(series, train=False)

        if darts_series_df is None:
            return None, None, None

        darts_series_df = darts_series_df.pd_dataframe()
        target = darts_series_df['signal']
        predicted_signal = self.model.predict(darts_series_df[['control']])
        residuals = target - predicted_signal


        anomaly_score = np.abs(residuals)
        detection = anomaly_score > self.n_sigmas_threshold * np.std(residuals)

        return darts_series_df.index, anomaly_score, detection

