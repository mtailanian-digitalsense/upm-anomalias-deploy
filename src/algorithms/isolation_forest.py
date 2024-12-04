from sklearn.ensemble import IsolationForest

from src.algorithms.base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = IsolationForest(**self.config['detector'])

    def _fit(self, series, **kwargs):
        darts_series_df = self.pre_process(series, train=True).pd_dataframe()
        self.model.fit(darts_series_df)

    def _predict(self, series, **kwargs):
        darts_series_df = self.pre_process(series, train=False).pd_dataframe()
        anomaly_score = -self.model.decision_function(darts_series_df)
        detection = self.model.predict(darts_series_df) == -1
        return darts_series_df.index, anomaly_score, detection
