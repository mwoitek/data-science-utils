import numpy as np
import pandas as pd
import pytest
from decorators.decorators import count_dropped  # type: ignore


@pytest.fixture
def simple_df() -> pd.DataFrame:
    data_dict = {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, np.nan, np.nan, 8]),
        "z": np.array([np.nan, 9, 10, 11]),
    }
    return pd.DataFrame(data=data_dict)


def test_no_row_dropped(simple_df: pd.DataFrame, capsys) -> None:  # type: ignore
    @count_dropped
    def drop_rows(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df.x.notna(), :]
        return df

    simple_df = drop_rows(simple_df)
    captured = capsys.readouterr()
    assert captured.out == "No observation was dropped\n"


def test_1_row_dropped(simple_df: pd.DataFrame, capsys) -> None:  # type: ignore
    @count_dropped
    def drop_rows(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df.z.notna(), :]
        return df

    simple_df = drop_rows(simple_df)
    captured = capsys.readouterr()
    assert captured.out == "1 observation was dropped\n"


def test_2_rows_dropped(simple_df: pd.DataFrame, capsys) -> None:  # type: ignore
    @count_dropped
    def drop_rows(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df.y.notna(), :]
        return df

    simple_df = drop_rows(simple_df)
    captured = capsys.readouterr()
    assert captured.out == "2 observations were dropped\n"


def test_count_dropped_decorated_func(simple_df: pd.DataFrame) -> None:
    @count_dropped
    def drop_rows(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df.y.notna(), :]
        return df

    assert drop_rows.__name__ == "drop_rows"
    simple_df = drop_rows(simple_df)
    expected_df = pd.DataFrame(
        data={
            "x": np.array([1, 4]),
            "y": np.array([5.0, 8.0]),
            "z": np.array([np.nan, 11]),
        },
        index=[0, 3],
    )
    assert simple_df.equals(expected_df)
