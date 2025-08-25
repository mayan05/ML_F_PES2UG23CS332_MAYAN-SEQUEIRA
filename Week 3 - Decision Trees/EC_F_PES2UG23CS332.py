import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the dataset using the target variable (last column).
    """
    target_col = data[:, -1]  # last column = class labels
    values, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return float(entropy)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate weighted average entropy (average information) of an attribute.
    """
    total_samples = len(data)
    attribute_values = np.unique(data[:, attribute])
    avg_info = 0.0

    for val in attribute_values:
        subset = data[data[:, attribute] == val]
        weight = len(subset) / total_samples
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += weight * subset_entropy

    return float(avg_info)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Information Gain = Entropy(S) - Avg_Info(attribute)
    Rounded to 4 decimal places.
    """
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    gain = dataset_entropy - avg_info
    return round(gain, 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Return (dict of attribute:gain, best_attribute_index)
    """
    num_attributes = data.shape[1] - 1  # exclude target column
    gains = {}

    for attr in range(num_attributes):
        gains[attr] = get_information_gain(data, attr)

    # select the attribute with max gain
    best_attr = max(gains, key=gains.get)
    return gains, best_attr