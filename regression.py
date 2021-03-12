"""Regression AI model

It can be used for:
    * Linear regression
    * Non-linear regression
    * Polynomial regression


"""
import random


class Regression:

    TRAINING = "training"
    TESTING = "testing"
    BIAS_INDEX = 0

    # CONFIG
    ROUND_AFTER_COMA = 2

    def __init__(self):
        self.labels = ([], None)  # features, target

        self._training_features_data = [[]]
        self._training_target_data = []
        self.training_data = (self._training_features_data, self._training_target_data)

        self.coefficients = []
        self.alpha = 1  # learning rate
        self.regularization = 0  # regularization value
        self.epoch = 1  # number of iteration

        self._testing_features_data = [[]]
        self._testing_target_data = []
        self.testing_data = (self._testing_features_data, self._testing_target_data)

        self.archive = {}  # to store results/models/coefficients

        self.__temporary_coefficients = []

    def get_number_of_training_features(self):
        return len(self._training_features_data[0])

    def get_number_of_training_rows(self):
        return len(self._training_target_data)

    def get_number_of_testing_features(self):
        return len(self._testing_features_data[0])

    def get_number_of_testing_rows(self):
        return len(self._testing_target_data)

    def create_coefficients_array(self):
        """
        Create list of coefficients for each feature based on number of features
        :return: list of coefficients n+1 elements, where n - number of features
        """
        n = self.get_number_of_training_features()
        for idx in range(n+1):  # one more for bias coefficient
            self.coefficients.append(random.random())
        return self.coefficients

    def set_training_data(self, feature_data: list, target_data: list) -> bool:
        """
        Set training data.
        :param feature_data: 2D array of features' data
        :param target_data: 1D arrays of target's data
        :return: True if set succeed otherwise False
        """
        return self._set_features_data(feature_data) and self._set_target_data(target_data)

    def set_testing_data(self, feature_data: list, target_data: list) -> bool:
        """
        Set testing data.
        :param feature_data: 2D array of features' data
        :param target_data: 1D arrays of target's data
        :return: True if set succeed otherwise False
        """
        return self._set_features_data(feature_data, is_training=False) and \
               self._set_target_data(target_data, is_training=False)

    def _set_features_data(self, features_data: list, is_training=True) -> bool:
        """
        Helper method of set_training_data() and set_testing_data()
        """
        if not features_data:
            return False
        if is_training is True:
            self._training_features_data = features_data
        else:
            self._testing_features_data = features_data
        return True

    def _set_target_data(self, target_data: list, is_training=True) -> bool:
        """
        Helper method of set_training_data() and set_testing_data()
        """
        if not target_data:
            return False
        if is_training is True:
            self._training_target_data = target_data
        else:
            self._testing_target_data = target_data
        return True

    # def evaluation(self, metric="MAE"):
    #
    #     def mae(actual_target, predicted_target):
    #         pass
    #
    #     if metric == "MAE":
    #         return mae(self._training_target_data)
    #
    #     pass

    def fit(self, method="gd") -> list:
        """
        Train our model
        :param method: "gd" - gradient descent
        :return: optimized coefficients
        """

        if method == "fancy_algo":
            coefficients_optimization = self.fancy_algo
        else:
            coefficients_optimization = self.gradient_descent
            if len(self.__temporary_coefficients) != len(self.coefficients):
                self.__temporary_coefficients = self.coefficients.copy()

        for e in range(self.epoch):
            coefficients_optimization()
            # store data for evaluation

        return self.coefficients

    def gradient_descent(self) -> None:
        """
        Adjust each coefficient according to the following formula:
        omega_i = omega_i * (1 - alpha * (reg/m) - (alpha/m) * SUM((hypothesis(x) - y)*(hypothesis(omega)')
        """
        regularization_coefficient = 1 - (self.alpha * (self.regularization / self.get_number_of_training_rows()))
        for coefficient_idx in range(len(self.coefficients)):
            self.__temporary_coefficients[coefficient_idx] = self.coefficients[coefficient_idx] \
                                                             * regularization_coefficient - self.alpha \
                                                             * self.get_cost_function_derivative(coefficient_idx)
        self.coefficients = self.__temporary_coefficients.copy()  # update coefficients

    def fancy_algo(self):
        """Might be something else instead of gradient descent"""
        pass

    def get_cost_function(self, main_source="training"):
        """
        Cost function = SUM(hypothesis(x) - y) / (2*m) for i in range(m)
        where m - number of lines
        :param main_source:
        :return:
        """

        if main_source == Regression.TESTING:
            m_row = self.get_number_of_testing_rows()
            features_data, target_data = self.testing_data
        else:
            m_row = self.get_number_of_training_rows()
            features_data, target_data = self.training_data

        cost_function = 0
        for idx in range(m_row):
            cost_function += (self._get_hypothesis(features_data[idx]) - target_data[idx])
        return round(cost_function / (2 * m_row), Regression.ROUND_AFTER_COMA)

    def get_cost_function_derivative(self, current_coefficient_idx: int) -> float:
        """
        derivative of Cost function J = SUM((hypothesis(x_i) - y_i) * x_i_j) for i from 0 to m
        where j - coefficient index, m - number of rows in our training data
        :param current_coefficient_idx: coefficient index
        :return:
        """
        m_row = self.get_number_of_training_rows()
        features_data, target_data = self.training_data
        cost_function_derivative = 0
        for idx in range(m_row):
            if current_coefficient_idx == Regression.BIAS_INDEX:
                der_part = 1
            else:
                der_part = features_data[idx][current_coefficient_idx]
            cost_function_derivative += round(((self._get_hypothesis(features_data[idx]) - target_data[idx])
                                               * der_part),
                                              Regression.ROUND_AFTER_COMA)
        return round(cost_function_derivative / (2 * m_row), Regression.ROUND_AFTER_COMA)

    def _get_hypothesis(self, features_data_line):
        """
        Calculate current hypothesis sum with current coefficients
        :param features_data_line: line of features from main table
        :return: hypothesis sum
        """
        n_features = len(features_data_line)
        hypothesis = self.coefficients[0]
        for idx in range(1, n_features+1):
            coefficient = self.coefficients[idx]
            feature_value = features_data_line[idx]
            hypothesis += round(coefficient * feature_value, Regression.ROUND_AFTER_COMA)
        return hypothesis

    def predict(self, new_feature_data_line):
        """
        :param new_feature_data_line: data line of features input
        :return: predicted target value based on model coefficients
        """
        if not self.coefficients or len(new_feature_data_line) != len(self.coefficients)-1:
            return -1
        predicted_target = self.coefficients[0]  # interception coefficient
        for coefficient in range(1, len(self.coefficients)):
            predicted_target += coefficient * new_feature_data_line[coefficient-1]
        return predicted_target
