import pandas as pd

from darts.ad import KMeansScorer, QuantileDetector

from src.algorithms.base import AnomalyDetector


class KMeansDartsDetector(AnomalyDetector):
    def __init__(self, config_path):
        super().__init__(config_path)
        sampling_step_in_minutes = self.config['detector']['sampling_step_in_minutes']
        kmeans_window_in_hours = self.config['detector']['window_in_hours']
        self.kmeans_window_in_samples = int(kmeans_window_in_hours * 60 / sampling_step_in_minutes)

        self.needed_history_in_hours = kmeans_window_in_hours
        for step in self.pre_processor.steps:
            if step.__class__.__name__ == 'SmoothSignalStep':
                large_smooth_win_in_hours = step.large_smooth_win * sampling_step_in_minutes / 60
                self.needed_history_in_hours = max(self.needed_history_in_hours, large_smooth_win_in_hours)
                break

        self.model = KMeansScorer(
            k=self.config['detector']['n_neighbors'],
            window=self.kmeans_window_in_samples,
        )
        self.detector = QuantileDetector(
            high_quantile=self.config['detector']['detection_quantile']
        )
        self.complete_series = pd.DataFrame()
        self.results = pd.DataFrame()

    def fit(self, series, target=None, **kwargs):
        self.complete_series = series
        darts_series = self.pre_process(series, train=True)
        self.model.fit(darts_series)
        self.detector.fit(self.model.score(darts_series))

    def predict(self, series, target=None, **kwargs):

        if (series is None) or (len(series) == 0):
            return None

        darts_series = self.pre_process(series, train=False)
        anomaly_score = self.model.score(darts_series)
        detection = self.detector.detect(anomaly_score)

        # Append historical data
        self.complete_series.loc[series.index[-1]] = series.iloc[-1]
        if self.results.empty:
            self.results = pd.DataFrame({'time_index': [series.index[-1]], 'anomaly_score': [anomaly_score.values()[0, 0]], 'detection': [detection.values()[0, 0]]}).set_index('time_index')
        else:
            self.results.loc[series.index[-1]] = {'anomaly_score': anomaly_score.values()[0, 0], 'detection': detection.values()[0, 0]}

        return {
            'time_index': series.index[-1],
            'anomaly_score': anomaly_score.values()[0, 0],
            'detection': detection.values()[0, 0] == 1,
        }

    def predict_all(self, series, **kwargs):

        if (series is None) or (len(series) == 0) or (not 'control' in series.columns):
            return None

        darts_series = self.pre_process(series, train=False)
        anomaly_score = self.model.score(darts_series)
        detection = self.detector.detect(anomaly_score)

        # Append historical data
        self.complete_series = pd.concat([self.complete_series, series])
        self.results = pd.DataFrame({
            'time_index': anomaly_score.time_index,
            'anomaly_score': anomaly_score.values()[:, 0],
            'detection': detection.values()[:, 0] == 1
        }).set_index('time_index')

    def plot(self, title, show=True, save_path=None, **kwargs):
        return super().plot(self.complete_series, self.results, title, show, save_path)
