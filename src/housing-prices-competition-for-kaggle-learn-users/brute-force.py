#!/usr/bin/env python

from dotenv import load_dotenv
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import kagglehub
import pandas as pd


def init_env():
    load_dotenv()

    path = kagglehub.competition_download("home-data-for-ml-course")

    print("Path to competition files:", path)

    return path


def load_data(path):
    train_path = path + "/train.csv"
    test_path = path + "/test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return (train_data, test_data)


def calc_mae(rf_model, X, y_true):
    y_pred = rf_model.predict(X)
    return mean_absolute_error(y_true, y_pred)


def find_optimal_features(train_data, features):
    optimal_score = None
    optimal_features = None

    rf_model = RandomForestRegressor(random_state=1)

    for length in range(1, len(features) + 1):
        for chosen_features in combinations(features, length):
            chosen_features = list(chosen_features)
            X = train_data[chosen_features]
            y = train_data.SalePrice

            scores = cross_val_score(rf_model, X, y, scoring="neg_mean_absolute_error")
            mean_score = scores.mean()

            if optimal_score is None or mean_score > optimal_score:
                optimal_score = mean_score
                optimal_features = chosen_features

    return optimal_features


def encode_pred(test_data, preds):
    output = pd.DataFrame({"Id": test_data.Id, "SalePrice": preds})
    output.to_csv("submission.csv", index=False)


def main():
    path = init_env()

    train_data, test_data = load_data(path)

    features = [
        "MSSubClass",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YearRemodAdd",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "MoSold",
        "YrSold",
    ]

    optimal_features = find_optimal_features(train_data, features)

    optimal_model = RandomForestRegressor(random_state=1)
    optimal_model.fit(train_data[optimal_features], train_data.SalePrice)

    preds = optimal_model.predict(test_data[optimal_features])

    encode_pred(test_data, preds)


if __name__ == "__main__":
    main()
