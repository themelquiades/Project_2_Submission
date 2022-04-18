# Function to set the signal based our own Business Logic
def set_signal(df):
    # Initialize the new `Signal` column
    df["signal"] = 0.0

    # Generate signal to buy stock long
    df.loc[(df["actual_returns"] >0), "signal"] = 1

    # Generate signal to sell stock short
    df.loc[(df["actual_returns"] <=0), "signal"] = -1

    # Copy the new "signal" column to a new Series called `y`.
    y = df["signal"]

    return y
