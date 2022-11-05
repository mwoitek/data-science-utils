from math import isclose

import numpy as np
import pandas as pd
import pytest
from missing_values.count import missing_values_column  # type: ignore


@pytest.fixture
def simple_df() -> pd.DataFrame:
    data_dict = {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, np.nan, np.nan, 8]),
        "z": np.array([np.nan, 9, 10, 11]),
    }
    return pd.DataFrame(data=data_dict)


def test_column_no_missing(simple_df: pd.DataFrame) -> None:
    count, percentage = missing_values_column(simple_df, "x")
    assert count == 0
    assert isclose(percentage, 0.0)


def test_columns_with_missing(simple_df: pd.DataFrame) -> None:
    count, percentage = missing_values_column(simple_df, "y")
    assert count == 2
    assert isclose(percentage, 50.0)

    count, percentage = missing_values_column(simple_df, "z")
    assert count == 1
    assert isclose(percentage, 25.0)
