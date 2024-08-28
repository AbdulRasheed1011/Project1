import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting Train and Test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Cat Boost Regressor': CatBoostRegressor(verbose=False),
                'Gradient Boost Regressor': GradientBoostingRegressor(),
                'Ada Boost Regressor': AdaBoostRegressor(),
                'Random Forest Regressor': RandomForestRegressor()
            }

            params = {
                'Linear Regression': {},
                'Lasso': {},
                'Ridge': {},
                'Decision Tree Regressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Cat Boost Regressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'Gradient Boost Regressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Ada Boost Regressor': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Random Forest Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'K-Neighbors Regressor' : {}
            }
            

            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, 
                                                 models=models, param = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score) 
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')

            logging.info('Best model found on Train and Test dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)