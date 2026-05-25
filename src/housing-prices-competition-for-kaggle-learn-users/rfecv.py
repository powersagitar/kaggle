#!/usr/bin/env python

from dotenv import load_dotenv
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
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

    X = train_data[features]
    y = train_data.SalePrice

    selector = RFECV(
        RandomForestRegressor(random_state=1), scoring="neg_mean_absolute_error"
    )

    selector.fit(X, y)

    support = selector.get_support()

    optimal_features = X.columns[support].tolist()

    optimal_model = RandomForestRegressor(random_state=1)
    optimal_model.fit(train_data[optimal_features], y)

    preds = optimal_model.predict(test_data[optimal_features])

    encode_pred(test_data, preds)


if __name__ == "__main__":
    main()
