import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomExecption
from src.logger import logging

from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decison Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosing Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifire": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    # "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    # "max_features": ["auto", "sqrt", "log2"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Decison Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    # "max_features": ["auto", "sqrt", "log2"],
                    # "splitter": ["best", "random"],
                },
                "Gradient Boosting": {
                    "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                    #  "criterion": ["squared_error", "friedman_mse"],
                    "learning_rate": [0.01, 0.1, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    # "max_features": ["auto", "sqrt", "log2"],
                    "subsample": [0.6, 0.7,0.8, 0.9, 1.0]
                },
                "Linear Regression": {},
                "K-Neighbors Classifier": {
                    "n_neighbors": [5, 7, 9, 11],
                    # "weights": ["uniform", "distance"],
                    # "algorithm": ["ball_tree", "kd_tree", "brute"],
                },
                "XGBClassifier": {
                    "learning_rate": [0.01, 0.1, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "CatBoosing Classifier": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.001],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Classifire": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    # "loss": ["linear", "square", "exponential"],
                    "learning_rate": [0.01, 0.1, 0.5, 1.0]
                },
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomExecption("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square 
        

        except Exception as e:
            raise CustomExecption(e,sys)
            