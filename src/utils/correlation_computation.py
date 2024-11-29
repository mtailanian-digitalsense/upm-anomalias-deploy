import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_rolling_correlation_pd(df, window=500):
    """
    Compute the correlation between the signal and the control columns of a dataframe
    :param df: has to have the columns 'signal' and 'control' columns
    :param window: the window size for the rolling correlation
    :return: the same dataframe with an additional column 'correlation' that contains the correlation between the signal
        and the control
    """
    corr = df['signal'].rolling(window=window, min_periods=1).corr(df['control'])
    corr = corr.fillna(1)
    corr = corr.replace([np.inf, -np.inf], 1)
    df['correlation'] = corr
    return df


def compute_rolling_correlation_np(df, window=500, verbose=True):
    """
    Compute the correlation between the signal and the control columns of a dataframe, using numpy, and simulating a
        sequential process
    :param df: has to have the columns 'signal' and 'control' columns
    :param window:  the window size for the rolling correlation
    :param verbose: if True, print the progress of the computation
    :return: the same dataframe with an additional column 'correlation' that contains the correlation between the signal
    """
    corr = np.empty(len(df))
    progress_bar = range(len(df))
    if verbose:
        progress_bar = tqdm(list(progress_bar), desc='Computing correlation')
    for t in progress_bar:
        start = max(0, t - window + 1)
        signal = df['signal'].iloc[start: t + 1]
        control = df['control'].iloc[start: t + 1]
        corr[t] = compute_correlation_np(signal, control)
    df['correlation'] = corr
    return df


def compute_correlation_np(window_signal, window_control):
    """
    This function computes the correlation between two windows of signal and control values
    :param window_signal: the signal values
    :param window_control: the control values
    :return: the correlation between the signal and the control
    """
    mean_signal = np.mean(window_signal)
    mean_control = np.mean(window_control)

    deviation_signal = window_signal - mean_signal
    deviation_control = window_control - mean_control

    covariance = np.mean(deviation_signal * deviation_control)
    std_signal = np.std(window_signal)
    std_control = np.std(window_control)

    if std_signal > 0 and std_control > 0:
        corr = covariance / (std_signal * std_control)
    else:
        corr = 1
    return corr


if __name__ == '__main__':
    import plotly.graph_objects as go

    series = pd.read_csv("../data/flow-meters/all_330A0013FIC.PIDA.PV.csv", index_col=0)

    series = compute_rolling_correlation_pd(series)
    series = series.rename(columns={'correlation': 'correlation_pd'})

    series = compute_rolling_correlation_np(series)
    series = series.rename(columns={'correlation': 'correlation_np'})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series['correlation_pd'], mode='lines', name='Corr Pandas'))
    fig.add_trace(go.Scatter(x=series.index, y=series['correlation_np'], mode='lines', name='Corr Numpy'))
    fig.show()
