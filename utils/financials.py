import numpy as np
import pandas as pd

def sortino_ratio(signals_df):
    # Calculate downside return values

    # Create a DataFrame that contains the Portfolio Daily Returns column
    sortino_ratio_df = signals_df[['portfolio_returns']].copy()

    # Create a column to hold downside return values
    sortino_ratio_df.loc[:,'Downside Returns'] = 0

    # Find Portfolio Daily Returns values less than 0,
    # square those values, and add them to the Downside Returns column
    sortino_ratio_df.loc[sortino_ratio_df['portfolio_returns'] < 0,
                         'Downside Returns'] = sortino_ratio_df['portfolio_returns']**2

    # Calculate the annualized return value
    annualized_return = (
        sortino_ratio_df['portfolio_returns'].mean() * 252
    )

    # Calculate the annualized downside standard deviation value
    downside_standard_deviation = (
        np.sqrt(sortino_ratio_df['Downside Returns'].mean()) * np.sqrt(252)
    )

    # The Sortino ratio is reached by dividing the annualized return value
    # by the downside standard deviation value
    if downside_standard_deviation == 0:
        sortino_ratio = float("nan")
    else:
        sortino_ratio = annualized_return/downside_standard_deviation
    
    return sortino_ratio