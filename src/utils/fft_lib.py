from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class LocalOutlier:
    index: int
    z_score: float

    @property
    def sign(self) -> int:
        return np.sign(self.z_score)


@dataclass
class RegionOutlier:
    start_idx: int
    end_idx: int
    score: float


@contextmanager
def nested_break():
    class NestedBreakException(Exception):
        pass

    try:
        yield NestedBreakException
    except NestedBreakException:
        pass


def reduce_parameters(frequencies: np.ndarray, k: int, filter_type: str) -> np.ndarray:
    """
    Reduce the Fourier coefficients based on the specified filter type.

    :param frequencies: Fourier-transformed data
    :param k: number of coefficients to retain
    :param filter_type: type of filter ('LPF', 'BPF', 'HPF')
    :return: reduced Fourier coefficients
    """
    n = len(frequencies)
    reduced = np.zeros_like(frequencies)

    if filter_type == 'LPF':
        # Keep the first k coefficients (low frequencies)
        reduced[:k] = frequencies[:k]
    elif filter_type == 'HPF':
        # Keep the last k coefficients (high frequencies)
        reduced[-k:] = frequencies[-k:]
    elif filter_type == 'BPF':
        # Keep the k coefficients in the middle band
        start_idx = (n // 2) - (k // 2)
        end_idx = start_idx + k
        reduced[start_idx:end_idx] = frequencies[start_idx:end_idx]
    else:
        raise ValueError("filter_type must be one of 'LPF', 'BPF', or 'HPF'.")

    return reduced


def calculate_local_outlier(
    data: np.ndarray, k: int, c: int, threshold: float, filter_type: str) -> Tuple[List[LocalOutlier], np.ndarray]:
    """
    :param data: input data (1-dimensional)
    :param k: number of parameters to be used in IFFT
    :param c: lookbehind and lookahead size for neighbors
    :param threshold: outlier threshold
    :param filter_type: type of filter ('LPF', 'BPF', 'HPF')
    :return: list of local outliers and anomaly scores (z_scores) for each point in data
    """
    n = len(data)
    k = max(min(k, n), 1)

    # Fourier transform of data
    y = reduce_parameters(np.fft.fft(data), k, filter_type)
    f2 = np.real(np.fft.ifft(y))

    # Difference of actual data value and the fft-fitted curve
    so = np.abs(f2 - data)
    # Average difference
    mso = np.mean(so)

    scores = []
    score_idxs = []
    for i in range(n):
        if so[i] > mso:
            # Average value of 'c' neighbors on both sides
            nav = np.average(data[max(i - c, 0):min(i + c, n - 1)])
            scores.append(data[i] - nav)
            score_idxs.append(i)
    scores = np.array(scores)

    # Find average and standard deviation of local difference
    ms = np.mean(scores)
    sds = np.std(scores)

    # Create z_scores array with the same size as data
    z_scores = np.zeros(n)

    results = []
    for i, idx in enumerate(score_idxs):
        z_score = (scores[i] - ms) / sds
        if abs(z_score) > threshold:
            results.append(LocalOutlier(idx, z_score))
            z_scores[idx] = z_score  # Assign z_score to the corresponding index in data

    return results, z_scores


def calculate_region_outlier(l_outliers: List[LocalOutlier], max_region_length: int, max_local_diff: int) -> List[
    RegionOutlier]:
    """
    :param l_outliers: list of local outliers with their z_score
    :param max_region_length: maximum outlier region length
    :param max_local_diff: maximum difference between two closed oppositely signed outliers
    :return: list of region outliers
    """

    def distance(a: int, b: int) -> int:
        if a > b:
            h = a
            a = b
            b = h
        return l_outliers[b].index - l_outliers[a].index

    regions = []
    i = 0
    n_l = len(l_outliers) - 1
    while i < n_l:
        s_sign = l_outliers[i].sign
        s_sign2 = l_outliers[i + 1].sign
        if s_sign != s_sign2 and distance(i, i + 1) <= max_local_diff:
            i += 1
            start_idx = i
            for i in range(i + 1, n_l):
                e_sign = l_outliers[i].sign
                e_sign2 = l_outliers[i + 1].sign
                if s_sign2 == e_sign and distance(start_idx, i) <= max_region_length \
                        and e_sign != e_sign2 and distance(i, i + 1) <= max_local_diff:
                    end_idx = i
                    regions.append(RegionOutlier(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        score=np.mean([abs(l.z_score) for l in l_outliers[start_idx: end_idx + 1]])
                    ))
                    i += 1
                    break
            i = start_idx
        else:
            i += 1

    return regions


def detect_anomalies(data: np.ndarray,
                     ifft_parameters: int = 5,
                     local_neighbor_window: int = 21,
                     local_outlier_threshold: float = .6,
                     max_region_size: int = 50,
                     max_sign_change_distance: int = 10,
                     filter_type: str = 'LPF',
                     **args) -> np.ndarray:
    """
    :param data: input time series
    :param ifft_parameters: number of parameters to be used in IFFT
    :param local_neighbor_window: centered window of neighbors to consider for z_score calculation
    :param local_outlier_threshold: outlier threshold in multiples of sigma
    :param max_region_size: maximum outlier region length
    :param max_sign_change_distance: maximum difference between two closed oppositely signed outliers
    :return: anomaly scores (same shape as input)
    """
    neighbor_c = local_neighbor_window // 2
    # print(ifft_parameters, neighbor_c, local_outlier_threshold, max_region_size, max_sign_change_distance)
    local_outliers, z_scores = calculate_local_outlier(data, ifft_parameters, neighbor_c, local_outlier_threshold, filter_type)
    # print(f"Found {len(local_outliers)} local outliers")

    regions = calculate_region_outlier(local_outliers, max_region_size, max_sign_change_distance)
    # print("Regions: ", regions)

    # broadcast region scores to data points
    anomaly_scores = np.zeros_like(data)
    for reg in regions:
        start_local = local_outliers[reg.start_idx]
        end_local = local_outliers[reg.end_idx]
        anomaly_scores[start_local.index:end_local.index + 1] = [reg.score] * (end_local.index - start_local.index + 1)

    return anomaly_scores, z_scores
