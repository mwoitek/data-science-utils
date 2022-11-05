import pandas as pd


def missing_values_column(
    df: pd.DataFrame,
    column: str,
    places: int = 2,
) -> tuple[int, float]:
    count = df[column].isna().sum()
    percentage = round(100 * count / len(df[column]), ndigits=places)
    return count, percentage


def missing_values_df(
    df: pd.DataFrame,
    places: int = 2,
    only_non_zero: bool = False,
) -> pd.DataFrame:
    counts = df.isna().sum()
    if only_non_zero:
        counts = counts[counts > 0]
    return pd.DataFrame(
        data={
            "Count": counts,
            "Percentage": (100 * counts / df.shape[0]).round(decimals=places),
        }
    )
