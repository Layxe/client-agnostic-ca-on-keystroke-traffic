import pandas as pd
import numpy as np

def normalize(series: pd.Series) -> pd.Series:
    """Normalize a pandas series.

    Args:
        series (pd.Series): Series to normalize.

    Returns:
        pd.Series: Normalized series.
    """

    return (series - series.mean()) / series.std()

def get_receive_time_diff_without_skips(df: pd.DataFrame, threshold_seconds: int = 60 * 10) -> pd.Series:
    """Get the time difference between consecutive receive times without considering time skips.

    Args:
        df (pd.DataFrame): Dataframe of the flow data.

    Returns:
        np.ndarray: Array of time differences between consecutive receive times.
    """

    diffs = df["receive_time"].diff()
    diffs = diffs[diffs <= threshold_seconds]

    return diffs