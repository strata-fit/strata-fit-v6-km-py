import numpy as np
import pandas as pd

from .exceptions import InputError
from .log import info
from .types import NoiseType

def add_noise_to_event_times(
    df: pd.DataFrame,
    time_column_name: str,
    noise_type: NoiseType,
    snr: float | None,
    random_seed: int | None
) -> pd.DataFrame:
    noise_type = NoiseType.coerce(noise_type)

    if noise_type == NoiseType.NONE:
        return df

    if random_seed is not None:
        np.random.seed(random_seed)
        info(f"Random seed set to {random_seed}.")

    if noise_type == NoiseType.GAUSSIAN:
        return apply_gaussian_noise(df, time_column_name, snr)
    elif noise_type == NoiseType.POISSON:
        return apply_poisson_noise(df, time_column_name)
    else:
        raise InputError(f"Unknown noise type: {noise_type}")

def apply_gaussian_noise(df: pd.DataFrame, time_column_name: str, snr: float | None) -> pd.DataFrame:
    if snr is None or snr <= 0:
        raise InputError("For Gaussian noise, 'snr' must be provided and > 0.")
    variance = np.var(df[time_column_name])
    std_dev = np.sqrt(variance / snr)
    noise = np.random.normal(0, std_dev, size=len(df))
    info(f"Applying Gaussian noise with std dev {std_dev:.4f}.")
    df[time_column_name] += np.round(noise)
    df[time_column_name] = df[time_column_name].clip(lower=0.0)
    return df

def apply_poisson_noise(df: pd.DataFrame, time_column_name: str) -> pd.DataFrame:
    info("Applying Poisson noise.")
    df[time_column_name] = df[time_column_name].apply(
        lambda x: np.random.poisson(lam=x) if x > 0 else 0
    )
    return df
