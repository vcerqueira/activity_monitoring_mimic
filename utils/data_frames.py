import numpy as np
import pandas as pd


def split_dataframe(df: pd.DataFrame, factor: np.ndarray):
    grouped_df = df.groupby(factor)
    grouped_df = list(grouped_df)

    out = [x[1] for x in grouped_df]

    return out
