from functools import wraps
from typing import Callable

import pandas as pd


def count_dropped(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:  # type: ignore
        size_before = df.shape[0]
        df = func(df, *args, **kwargs)
        num_dropped = size_before - df.shape[0]
        match num_dropped:
            case 0:
                print("No observation was dropped")
            case 1:
                print("1 observation was dropped")
            case _:
                print(f"{num_dropped} observations were dropped")
        return df

    return wrapper
