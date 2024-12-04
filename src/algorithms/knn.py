import pandas as pd
from darts import TimeSeries
from darts.ad import QuantileDetector
from sklearn.neighbors import NearestNeighbors

from src.algorithms.base import AnomalyDetector


class KNNDetector(AnomalyDetector):

    def __init__(self, config_path, **kwargs):
        super().__init__(config_path)

        self.model = NearestNeighbors(
            n_neighbors = self.config['detector']['n_neighbors'],
            leaf_size = self.config['detector']['leaf_size'],
            algorithm = self.config['detector']['algorithm'],
            metric = self.config['detector']['metric'],
            p = self.config['detector']['p']
        )

        self.detector = QuantileDetector(high_quantile=self.config['detector']['detection_quantile'])

    def _fit(self, series, **kwargs):

        darts_series_df = self.pre_process(series, train=True).pd_dataframe()

        self.model.fit(darts_series_df.values)

        anomaly_score, _ = self.model.kneighbors(darts_series_df.values)
        anomaly_score = anomaly_score.mean(axis=1)

        anomaly_score_df = pd.DataFrame(anomaly_score, index=darts_series_df.index, columns=['anomaly_score'])
        anomaly_score_df['date'] = pd.to_datetime(anomaly_score_df.index)
        anomaly_score_darts = TimeSeries.from_dataframe(
            anomaly_score_df,
            time_col='date',
        )
        self.detector.fit(anomaly_score_darts)

    def _predict(self, series, **kwargs):

        darts_series_df = self.pre_process(series, train=False).pd_dataframe()
        anomaly_score, _ = self.model.kneighbors(darts_series_df.values)
        anomaly_score = anomaly_score.mean(axis=1)

        anomaly_score_df = pd.DataFrame(anomaly_score, index=darts_series_df.index, columns=['anomaly_score'])
        anomaly_score_df['date'] = pd.to_datetime(anomaly_score_df.index)
        anomaly_score_darts = TimeSeries.from_dataframe(
            anomaly_score_df,
            time_col='date',
        )
        detection = self.detector.detect(anomaly_score_darts)

        return darts_series_df.index, anomaly_score, detection.values()[:, 0] == 1
