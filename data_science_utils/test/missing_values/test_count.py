from math import isclose

import numpy as np
import pandas as pd
import pytest
from missing_values.count import missing_values_column  # type: ignore
from missing_values.count import missing_values_df
from missing_values.count import no_missing_value


@pytest.fixture
def simple_df() -> pd.DataFrame:
    data_dict = {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, np.nan, np.nan, 8]),
        "z": np.array([np.nan, 9, 10, 11]),
    }
    return pd.DataFrame(data=data_dict)


@pytest.fixture
def df_no_missing() -> pd.DataFrame:
    data_dict = {
        "x": [1, 2, 3, 4],
        "y": [5, 6, 7, 8],
        "z": ["a", "b", "c", "d"],
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


def test_df_correct_counts(simple_df: pd.DataFrame) -> None:
    df_counts_percentages = missing_values_df(simple_df)
    counts = df_counts_percentages["Count"].to_numpy()
    expected_counts = np.array([0, 2, 1])
    assert np.all(counts == expected_counts)


def test_df_correct_percentages(simple_df: pd.DataFrame) -> None:
    df_counts_percentages = missing_values_df(simple_df)
    percentages = df_counts_percentages["Percentage"].to_numpy()
    expected_percentages = np.array([0.0, 50.0, 25.0])
    assert np.allclose(percentages, expected_percentages)


def test_no_missing_false(simple_df: pd.DataFrame) -> None:
    assert not no_missing_value(simple_df)


def test_no_missing_true(df_no_missing: pd.DataFrame) -> None:
    assert no_missing_value(df_no_missing)
