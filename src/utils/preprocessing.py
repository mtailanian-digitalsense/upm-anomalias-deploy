import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values

from src.utils.registry import step_registry


def register_step(cls):
    step_registry.register(cls)
    return cls


class Preprocessor:
    def __init__(self, steps=None):
        self.steps = steps if steps is not None else []

    def add_step(self, step):
        self.steps.append(step)

    def __call__(self, original_series, signals_to_use, train=False):

        series = original_series.copy()
        for col in original_series.columns[:]:
            series[f'original_{col}'] = original_series[col]

        start = series.index[0]
        end = series.index[-1]

        for step in self.steps:
            series = step.process(series, train=train)

        series = series[signals_to_use]
        series = self._fill_missing_values(series, start, end, signals_to_use)

        return series

    @staticmethod
    def _fill_missing_values(series, start, end, signals_to_use):
        if series.index[0] != start:
            new_point = pd.Series({start: np.nan})
            series = pd.concat([new_point, series])
        if series.index[-1] != end:
            new_point = pd.Series({end: np.nan})
            series = pd.concat([series, new_point])

        series['date'] = pd.to_datetime(series.index)
        darts_series = TimeSeries.from_dataframe(
            series,
            time_col='date',
            value_cols=signals_to_use,
            fill_missing_dates=True,
            freq='10min',
        )
        darts_series = fill_missing_values(darts_series, 'auto')
        return darts_series


class PreprocessingStep:
    def process(self, series, train=False):
        raise NotImplementedError("Each preprocessing step must implement the process method.")


@register_step
class FilterParadasStep(PreprocessingStep):
    def __init__(self, filter_paradas, filter_on_control, filter_on_test):
        self.filter_paradas = filter_paradas
        self.filter_on_control = filter_on_control
        self.filter_on_test = filter_on_test

    def process(self, series, train=False):
        if train:
            if self.filter_on_control and 'control' in series.columns:
                return series[series['control'] > self.filter_paradas]
            elif self.filter_on_control and 'control' not in series.columns:
                return series[series['signal'] > 5] # TODO: choose a threshold for the signal
            else:
                return series[series['signal'] > self.filter_paradas]

        elif self.filter_on_test:
            if self.filter_on_control and 'control' in series.columns:
                return series[series['control'] > self.filter_paradas]
            elif self.filter_on_control and 'control' not in series.columns:
                return series[series['signal'] > 5] # TODO: choose a threshold for the signal
            else:
                return series[series['signal'] > self.filter_paradas]
                
        return series

@register_step
class DecimateStep(PreprocessingStep):
    def __init__(self, decimate):
        self.decimate = decimate

    def process(self, series, train=False):
        freq = str(self.freq*self.decimate) + 'T'
        return series.resample(freq).mean()

@register_step
class SmoothSignalStep(PreprocessingStep):
    def __init__(self, small_smooth_win, large_smooth_win, smooth_only_signal=True):
        self.small_smooth_win = small_smooth_win
        self.large_smooth_win = large_smooth_win
        self.smooth_only_signal = smooth_only_signal

    def process(self, series, train=False):
        columns_to_smooth = ['signal']
        if not self.smooth_only_signal:
            columns_to_smooth.extend(['control', 'production'])

        for c in columns_to_smooth:
            series[c] = series[f'original_{c}'].rolling(window=self.small_smooth_win, min_periods=1).mean()
            if self.large_smooth_win is not None:
                series[c] = series[c] - series[f'original_{c}'].rolling(window=self.large_smooth_win, min_periods=1).mean()
        return series

@register_step
class NormalizeStep(PreprocessingStep):
    def __init__(self, q_porcentage_small=0.01, q_porcentage_large=0.99):
        self.q_percentage_small = q_porcentage_small
        self.q_percentage_large = q_porcentage_large
        self.q_small, self.q_large = None, None

    def process(self, series, train=False):

        if train:
            self.q_small = series.quantile(q=self.q_percentage_small)
            self.q_large = series.quantile(q=self.q_percentage_large)

        normalized_signal = (series - self.q_small) / (self.q_large - self.q_small)

        return normalized_signal
