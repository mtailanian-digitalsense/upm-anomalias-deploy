from darts.models import XGBModel
from darts.ad import QuantileDetector
from darts.ad import (
    ForecastingAnomalyModel,
    KMeansScorer,
    NormScorer,
)

from src.algorithms.base import AnomalyDetector

VERBOSE = False


class XGBDartsDetector(AnomalyDetector):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.model = XGBModel(
            lags=self.config['detector']['window'],
            output_chunk_length=self.config['detector']['output_chunk_length'],
            device=self.device,
        )

        self.anomaly_model = ForecastingAnomalyModel(
            model=self.model,
            scorer=[
                NormScorer(ord=1),
                KMeansScorer(k=10),
            ],
        )

        self.detector = QuantileDetector(high_quantile=self.config['detector']['detection_quantile'])

    def _fit(self, series, target=None, **kwargs):
        darts_series = self.pre_process(series, train=True)

        self.model.fit(darts_series)
        self.anomaly_model.fit(darts_series, start=0.1, allow_model_training=False, verbose=VERBOSE)

        anomaly_score = self.anomaly_model.score(darts_series, start=0.75, verbose=VERBOSE)
        self.detector.fit(anomaly_score[-1])  # Keeping the last scorer only

    def _predict(self, series, target=None, **kwargs):
        darts_series = self.pre_process(series, train=False)
        anomaly_scores, model_forecasting = self.anomaly_model.score(darts_series, start=0.1, return_model_prediction=True, verbose=VERBOSE)
        anomaly_score = anomaly_scores[-1]
        detection = self.detector.detect(series=anomaly_score)

        return anomaly_score.time_index, anomaly_score.values()[:, 0], detection.values()[:, 0] == 1
