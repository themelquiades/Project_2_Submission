import pandas as pd
import numpy as np

# Import our utils libraries
from utils.financials import (
    sortino_ratio
)

# Runs backtest using our defined strategy
def run_backtest(backtest_df, commission = float(0.001), initial_capital = float(10000), trade_size = float(5000)):
    prev = 0
    busd_balance = initial_capital
    coin_balance = float(0)
    actions = []
    commissions = []
    fees = []
    transactions = []
    account_balance = []
    coin_holdings = []
    for i in range(0, len(backtest_df)):
        previous_trade = backtest_df.iloc[i]['trades']
        if previous_trade == 1 and prev != 1:
            #BUY and its neither SELL nor HOLD
            if busd_balance >= 0:
                action = "BUY"
                fee = trade_size*commission
                busd_balance=busd_balance-trade_size-fee
                trade=trade_size/backtest_df.iloc[i]['close']
                coin_balance=coin_balance+trade
                actions.append(action)
                transactions.append(trade_size)
                fees.append(fee)
                account_balance.append(busd_balance)
                coin_holdings.append(coin_balance)
            else:
                action = "EXIT"
                actions.append(action)
                transactions.append(0)
                fees.append(0)
                account_balance.append(busd_balance)
                coin_holdings.append(coin_balance)
        elif previous_trade == 1 and prev == 1:
            #BUY and previous is BUY
            action = "HOLD"
            actions.append(action)
            fees.append(0)
            transactions.append(0)
            account_balance.append(busd_balance)
            coin_holdings.append(coin_balance)
        elif previous_trade == -1 and prev != -1 and i>0:
            #SELL and its neither BUY nor HOLD
            action = "SELL"
            if coin_balance != 0:
                fee = trade_size*commission
            else: 
                fee =0
            trade = coin_balance * backtest_df.iloc[i]['close']
            busd_balance=busd_balance+trade-fee
            coin_balance = 0
            actions.append(action)
            transactions.append(trade)
            fees.append(fee)
            account_balance.append(busd_balance)
            coin_holdings.append(coin_balance)
        elif previous_trade == -1 and prev == -1 or i ==0:
            #SELL and previous is SELL
            action = "WAIT"
            actions.append(action)
            fees.append(0)
            transactions.append(0)
            account_balance.append(busd_balance)
            coin_holdings.append(coin_balance)
        prev = previous_trade

    backtest_df['action'] = actions
    backtest_df['transactions'] = transactions
    backtest_df['fees'] = fees
    backtest_df['busd_balance'] = account_balance
    backtest_df['coin_holdings'] = coin_holdings
    backtest_df['portfolio'] = (backtest_df['coin_holdings'] * backtest_df['close']) + backtest_df['busd_balance']
    backtest_df['portfolio_returns'] = backtest_df['portfolio'].pct_change()
    
    # Calculate the portfolio cumulative returns
    backtest_df['portfolio_cumulative_returns'] = (1 + backtest_df['portfolio_returns']).cumprod() - 1
    
    return backtest_df

def calculate_risk_metrics(signals_df):
    metrics = [
    'Annualized Return',
    'Cumulative Returns',
    'Annual Volatility',
    'Sharpe Ratio',
    'Sortino Ratio']
    columns = ['Backtest']
    
    # Initialize the DataFrame with index set to the evaluation metrics and the column
    portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)
    
    # Calculate annualized return
    portfolio_evaluation_df.loc['Annualized Return'] = (
        signals_df['portfolio_returns'].mean() * 252
    )
    # Calculate cumulative return
    portfolio_evaluation_df.loc['Cumulative Returns'] = signals_df['portfolio_cumulative_returns'][-1]
    
    # Calculate annual volatility
    portfolio_evaluation_df.loc['Annual Volatility'] = (signals_df['portfolio_returns'].std() * np.sqrt(252) )
    
    # Calculate Sharpe ratio
    portfolio_evaluation_df.loc['Sharpe Ratio'] = (
        signals_df['portfolio_returns'].mean() * 252) / (
        signals_df['portfolio_returns'].std() * np.sqrt(252)
    )
    
    portfolio_evaluation_df.loc['Sortino Ratio'] = sortino_ratio(signals_df)
    
    return portfolio_evaluation_df.T
    
    
# Backtest the model against our trading logic
def backtest_model(X_test, y_predicted_test, df_coinpair):
    # Create a new empty predictions DataFrame
    predictions_df = pd.DataFrame(index=X_test.index)
    
    # Return the best predicted, yet add
    predictions_df["trades"] = y_predicted_test
    
    # Calculate the daily returns using the closing prices and assign it as a new column called 'actual_returns'
    df_coinpair["actual_returns"] = df_coinpair["close"].pct_change()
    
    # Add in actual returns and calculate trading returns
    predictions_df["actual_returns"] = df_coinpair["actual_returns"]
    predictions_df["trading_returns"] = df_coinpair["actual_returns"] * predictions_df["trades"]
    backtest_df = run_backtest(pd.concat([df_coinpair["close"], predictions_df], axis="columns", join="inner"))
    
    # Calculate the risk metrics for each model
    risk_metrics = calculate_risk_metrics(backtest_df)
    
    # Sum all the actual_returns and portfolio returns to prepare the performance summary
    backtest_df[["actual_returns", "portfolio_returns", "trading_returns"]].sum()
    difference = (backtest_df["portfolio_returns"].sum() - backtest_df["actual_returns"].sum())
    
    # Defines the summary dictionary
    summary = {"dif": difference, "portfolio_returns": backtest_df["portfolio_returns"].sum(), "actual_returns": backtest_df["actual_returns"].sum(), "trading_returns": backtest_df["trading_returns"].sum(), "risk_metrics": risk_metrics.to_dict("records")}

    return {"df": backtest_df, "summary": summary }
