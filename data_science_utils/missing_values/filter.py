import pandas as pd

from .count import missing_values_df


def get_columns_percentage_missing_values(
    df: pd.DataFrame,
    percentage: float,
) -> list[str]:
    """Get a list of strings containing the names of the columns for which the
    percentage of missing values is equal to or greater than a given value."""
    df_counts_percentages = missing_values_df(df)
    df_counts_percentages = df_counts_percentages[
        df_counts_percentages["Percentage"] >= percentage
    ]
    return df_counts_percentages.index.values.tolist()
