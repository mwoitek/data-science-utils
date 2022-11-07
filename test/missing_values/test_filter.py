import numpy as np
import pandas as pd
import pytest
from missing_values.filter import get_columns_percentage_missing_values  # type: ignore


@pytest.fixture
def simple_df() -> pd.DataFrame:
    data_dict = {
        "w": np.array([1, 2, 3, 4]),
        "x": np.array(4 * [np.nan]),
        "y": np.array([5, np.nan, np.nan, 8]),
        "z": np.array([np.nan, 9, 10, 11]),
    }
    return pd.DataFrame(data=data_dict)


def test_percentage_0(simple_df: pd.DataFrame) -> None:
    assert get_columns_percentage_missing_values(simple_df, 0.0) == ["w", "x", "y", "z"]


def test_percentage_100(simple_df: pd.DataFrame) -> None:
    assert get_columns_percentage_missing_values(simple_df, 100.0) == ["x"]


def test_intermediate_percentages(simple_df: pd.DataFrame) -> None:
    assert get_columns_percentage_missing_values(simple_df, 25.0) == ["x", "y", "z"]
    assert get_columns_percentage_missing_values(simple_df, 50.0) == ["x", "y"]
    assert get_columns_percentage_missing_values(simple_df, 75.0) == ["x"]
