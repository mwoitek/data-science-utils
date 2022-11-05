import pandas as pd


def missing_values_column(
    df: pd.DataFrame,
    column: str,
    places: int = 2,
) -> tuple[int, float]:
    count = df[column].isna().sum()
    percentage = round(100 * count / len(df[column]), ndigits=places)
    return count, percentage
