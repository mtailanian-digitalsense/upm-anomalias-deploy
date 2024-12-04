from darts.ad import KMeansScorer, QuantileDetector

from src.algorithms.base import AnomalyDetector
from src.constants import SAMPLES_PER_HOUR


class KMeansDartsDetector(AnomalyDetector):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = KMeansScorer(
            k=self.config['detector']['n_neighbors'],
            window=self.config['detector']['window_in_hours'] * SAMPLES_PER_HOUR,
        )
        self.detector = QuantileDetector(
            high_quantile=self.config['detector']['detection_quantile']
        )

    def _fit(self, series, target=None, **kwargs):
        darts_series = self.pre_process(series, train=True)
        self.model.fit(darts_series)
        self.detector.fit(self.model.score(darts_series))

    def _predict(self, series, target=None, **kwargs):
        darts_series = self.pre_process(series, train=False)
        anomaly_score = self.model.score(darts_series)
        detection = self.detector.detect(anomaly_score)
        return anomaly_score.time_index, anomaly_score.values()[:, 0], detection.values()[:, 0] == 1
