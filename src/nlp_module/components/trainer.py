import os
import sys
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from shared_utils.utils import save_object
from nlp_module.exception import CustomException


class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Model Accuracy: {accuracy}")

            save_object("artifacts/model.pkl", self.model)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)