# Import Libraries
import pickle
import pandas as pd

# Import DateOffSet
from pandas.tseries.offsets import DateOffset

# Import the classifier Models
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

# Import Classification Report
from sklearn.metrics import classification_report

# Import our utils libraries
from utils.backtesting import (
    backtest_model
)

def select_models(X, X_test, X_train, y_test, y_train, df_coinpair, export_model_to_db=False, models = [
    "RandomForestClassifier", 
    "SVM", 
    "LinearSVC",
    "DecisionTreeClassifier", 
    "AdaBoostClassifier", 
    "AdaBoostClassifier_100",
    "MLPClassifier", 
    "GaussianNB", 
    "GaussianProcessClassifier", 
    "KNeighborsClassifier", 
    "KNeighborsClassifier_3",
    "KNeighborsClassifierBagging",
    "SGDClassifier"
    ]):
    f1_scores = []
    for model_name in models:
        print(f"Fitting {model_name}..")
        if model_name == "RandomForestClassifier":
            model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        elif model_name == "SVM":
            model = SVC(gamma=2, C=1)
        elif model_name == "LinearSVC":
            model = LinearSVC(dual=False)
        elif model_name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(max_depth=5)
        elif model_name == "AdaBoostClassifier":
            model = AdaBoostClassifier()
        elif model_name == "AdaBoostClassifier_5000":
            model = AdaBoostClassifier(n_estimators=5000)
        elif model_name == "MLPClassifier":
            model = MLPClassifier(alpha=1, max_iter=1000)
        elif model_name == "GaussianNB":
            model = GaussianNB()
        elif model_name == "GaussianProcessClassifier":
            model = GaussianProcessClassifier(1.0 * RBF(1.0))
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier()
        elif model_name == "KNeighborsClassifier_3":
            model = KNeighborsClassifier(3)
        elif model_name == "KNeighborsClassifierBagging":
            model = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
        elif model_name == "SGDClassifier":
            model = SGDClassifier(max_iter=1000, tol=1e-3)
            
        model.fit(X_train, y_train)

        # Use the trained model to predict the trading signals for the testing data.
        y_predicted_test = model.predict(X_test)

        # Get our Important Features for the Model
        importances = get_feature_importance(model, X)
        
        # Evaluate the model's ability to predict the trading signal for the testing data using a classification report
        report = classification_report(y_test, y_predicted_test, output_dict=True, zero_division=1)
        
        # Backtest our model for model total return
        backtest_data = backtest_model(X_test, y_predicted_test, df_coinpair)
        backtest_data_df = backtest_data["df"]
        backtest_summary = backtest_data["summary"]
        
        # Set the index as tiemstamp
        backtest_data_df = backtest_data_df.reset_index()
        backtest_data_df.columns = backtest_data_df.columns.str.replace("index", 'timestamp')
        
        # Sets the variable with the f1_scores and prepares to Merge with the Feature Importance Dict
        f1_scores_temp = {"model": model_name, "trader_vs_actual": backtest_summary["dif"],"actual_returns_sum": backtest_summary["actual_returns"],"portfolio_returns_sum": backtest_summary["portfolio_returns"], "accuracy": report["accuracy"], "f1(-1.0)": report["-1.0"]["f1-score"], "f1(1.0)": report["1.0"]["f1-score"], "risk_metrics": backtest_summary["risk_metrics"], "backtest_data": backtest_data_df.to_dict("records")}
        
        # Export model as Binary for
        if export_model_to_db == True:
             f1_scores_temp["pickle_model"] = pickle.dumps(model)

        # Merges both Dicts to create a single List per Model
        f1_scores_temp.update(importances)
        
        # Appends the data to f1_scores variable
        f1_scores.append(f1_scores_temp)
        
    # Creates a f1_scores_df DataFrame
    f1_scores_df = pd.DataFrame(f1_scores)
    f1_scores_df = f1_scores_df.set_index('model')
    f1_scores_df = f1_scores_df.sort_values(by=['accuracy'], ascending=False)

    # Returns the f1_scores_df Dataframe
    return {"f1_scores_df": f1_scores_df, "X_test": X_test }

# Function to return Feature Importances of a Model
def get_feature_importance(model, X):
    try:
        # Zip the feature importances with the associated feature name
        important_features = model.feature_importances_
    except:
        # Since the Model doesn't have Feature Importance, create an empty Dataframe with default values of 0
        important_features = [0] * len(X.columns)

    # Returns a dict of the features
    return dict(zip(X.columns,important_features))

# Set training and testing data
def train_test_data(X, y, training_begin,training_end, oversample=True):
    # Import StandardScaler and OverSampler
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import RandomOverSampler
    # Generate the X_train and y_train DataFrames using loc to select the rows from `training_begin` up to `training_end`
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # Generate the X_test and y_test DataFrames using loc to select from `training_end` to the last row in the DataFrame.
    X_test = X.loc[training_end:]
    y_test = y.loc[training_end:]

    # Use StandardScaler to scale the X_train and X_test data.
    scaler = StandardScaler()

    #X_scaler = scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if oversample:
        # Use RandomOverSampler to resample the datase using random_state=1
        resampler = RandomOverSampler(random_state=1)
        X_train, y_train = resampler.fit_resample(X_train, y_train)
        
    # Returns a dictionary with the Testing and Training Data
    return {"X_test":X_test, "y_test":y_test,  "X_train": X_train, "y_train":y_train}

# Function to select the training period
def get_training_dates(X, timeframe, date_range = {"training_start":12, "testing_range": 2, "period": "weeks"}):
    # Use monthly
    if date_range["period"] == "months":
        # Use the following code to select the start of the training period: `training_begin = X.index.min()`
        training_begin = X.index.max() - DateOffset(months=date_range["training_start"])

        # Use the following code to select the ending period for the training data: `training_end = X.index.min() + DateOffset(months=6)`
        training_end = X.index.max() - DateOffset(months=date_range["testing_range"])
        
    # Use weekly
    elif date_range["period"] == "weeks":
        # Use the following code to select the start of the training period: `training_begin = X.index.min()`
        training_begin = X.index.max() - DateOffset(weeks=date_range["training_start"])

        # Use the following code to select the ending period for the training data: `training_end = X.index.min() + DateOffset(months=6)`
        training_end = X.index.max() - DateOffset(weeks=date_range["testing_range"])
    # Use hours
    elif date_range["period"] == "hours":
        # Use the following code to select the start of the training period: `training_begin = X.index.min()`
        training_begin = X.index.max() - DateOffset(hours=date_range["training_start"])

        # Use the following code to select the ending period for the training data: `training_end = X.index.min() + DateOffset(months=6)`
        training_end = X.index.max() - DateOffset(hours=date_range["testing_range"])
    # Use minutes
    elif date_range["period"] == "minutes":
        # Use the following code to select the start of the training period: `training_begin = X.index.min()`
        training_begin = X.index.max() - DateOffset(minutes=date_range["training_start"])

        # Use the following code to select the ending period for the training data: `training_end = X.index.min() + DateOffset(months=6)`
        training_end = X.index.max() - DateOffset(minutes=date_range["testing_range"])

    # returns dict with training end and training begin
    return {"training_begin": training_begin, "training_end":training_end}