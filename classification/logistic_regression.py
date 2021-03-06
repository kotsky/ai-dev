"""Logistic Regression AI model

0 or 1 with LR.

Implemented with:
    * optimization algorithm: gradient descent
    * learning rate
    * regularization (penalty) L2 (Lasso)
    * for various number of iterations
    * logs writing for further debugging / plot
    * adjustable logistic coefficient for prediction threshold
    * evaluation with confusion matrix, precision and recall

Author: kotsky

"""

import random
import math

INT32_MAX = 2147483646 // 2  # in half


class LogisticRegression:
    """
    Regression model based on gradient descent, which comes with standard:
    * fit()
    * predict()
    * set_model_parameters()
    * set_training_data
    * set_testing_data
    * evaluate() - MAE technique
    """

    class _Logs:
        """
        Entity to save training/testing cost function each iteration.
        Cost functions are saved per each iteration (epoch).
        Model config is stored here as well for further drawing.
        ...
        """

        # logs_path = "logs/"

        # class LogNode:
        #     def __init__(self, cost_training_function, cost_test_function,
        #                  coefficients, alpha, regularization, d, i_d):
        #         self.cost_test_function = cost_test_function  # final after iteration
        #         self.cost_training_function = cost_training_function
        #         self.coefficients = coefficients
        #         self.alpha = alpha
        #         self.regularization = regularization
        #         self.d = d  # hypothesis power
        #         self.id = i_d
        #
        #     def __str__(self):
        #         return "ID {}, Jtr = {}, Jcv = {}, h_power = {}, alpha = {}, reg = {}". \
        #             format(self.id, self.cost_training_function, self.cost_test_function,
        #                    self.d, self.alpha, self.regularization)
        #
        #     def __repr__(self):
        #         return "Log {}".format(self.id)

        def __init__(self, alpha, regularization, iterations, training_cf=None, testing_cf=None):
            if training_cf is None:
                training_cf = []
            if testing_cf is None:
                testing_cf = []
            self.training_cf = training_cf  # training cost function
            self.testing_cf = testing_cf  # testing cost function
            self.alpha = alpha
            self.regularization = regularization
            self.iterations = iterations

        def __repr__(self):
            return "Logs of model settings alpha = {}, reg = {}".format(self.alpha, self.regularization)

        def __len__(self):
            return len(self.training_cf)

        def copy(self):
            return LogisticRegression._Logs(self.alpha, self.regularization, self.iterations,
                                            self.training_cf, self.testing_cf)

        def add_log(self, cost_training_function: float, cost_test_function: float,
                    coefficients: list, alpha: float, regularization: float) -> None:
            # node = Regression._Logs.LogNode(cost_training_function, cost_test_function,
            #                                 coefficients, alpha, regularization, self._d, len(self))
            # self.alpha = alpha
            # self.regularization = regularization
            self.training_cf.append(cost_training_function)
            self.testing_cf.append(cost_test_function)

        def get_logs(self):
            """
            Return brand new entity for further logs' procession
            """
            return self.copy()

    _TRAINING = "training"
    _TESTING = "testing"
    _BIAS_INDEX = 0

    # CONFIG
    ROUND_AFTER_COMA = 2
    RANDOM_WEIGHT_INITIALIZATION = 20  # range where coefficients will be defined initially

    def __init__(self):
        self.labels = ([], None)  # features, target

        self._training_features_data = [[]]
        self._training_target_data = []

        self.coefficients = []
        self.alpha = 1  # learning rate
        self.regularization = 0  # regularization value: Lasso Regularization
        self.epoch = 1  # number of iteration
        self.logistic_threshold = 0.5

        self._testing_features_data = [[]]
        self._testing_target_data = []

        # self.best_setup = {"minJ": float("inf"), "coefficients": []}  # to store best found model setup

        self._d = None  # power of hypothesis
        self._log_storage = None
        self._log_flag = False

        self.cost_training_function = None
        self.cost_testing_function = None

        self.__temporary_coefficients = []

    def get_number_of_training_features(self) -> int:
        return len(self._training_features_data[0])

    def get_number_of_training_rows(self) -> int:
        return len(self._training_target_data)

    def get_number_of_testing_features(self) -> int:
        return len(self._testing_features_data[0])

    def get_number_of_testing_rows(self) -> int:
        return len(self._testing_target_data)

    def log_mode(self, flag=True) -> None:
        if flag is True:
            print("Log mode is enable")
        else:
            print("Log mode is disabled")
        self._log_flag = flag

    def set_labels(self, labels: (list, str)):
        """Define training data labels"""
        self.labels = labels
        self._update_hypothesis_power()

    def take_model_snapshot(self):
        """
        Store model into logs into log storage for further visualization.
        :return: None
        """
        self._calculate_current_cost_functions()  # update cost functions
        self._log_storage.add_log(self.cost_training_function, self.cost_testing_function,
                                  self.coefficients.copy(), self.alpha, self.regularization)

    def get_logs(self):
        """
        :return: get brand new log entity.
        """
        return self._log_storage.get_logs() if self._log_storage is not None else [-1]

    def create_coefficients_array(self, r=False):
        """
        Create list of coefficients for each feature based on number of features
        :return: list of coefficients n+1 elements, where n - number of features
        """
        n = self.get_number_of_training_features()
        self.coefficients = []
        for idx in range(n + 1):  # one more for bias coefficient
            self.coefficients.append(random.randint(-self.RANDOM_WEIGHT_INITIALIZATION,
                                                    self.RANDOM_WEIGHT_INITIALIZATION))
            while self.coefficients[-1] == 0.0:  # avoid ZERO initialization
                self.coefficients[-1] = (random.randint(-self.RANDOM_WEIGHT_INITIALIZATION,
                                                        self.RANDOM_WEIGHT_INITIALIZATION))
            if r is True:
                self.coefficients[-1] /= self.RANDOM_WEIGHT_INITIALIZATION
        print("Initiated coefficients are " + str(self.coefficients))
        return self.coefficients

    def set_model_parameters(self, alpha=1, regularization=0, epoch=1) -> None:
        """
        Set model parameters.
        :param alpha: learning rate
        :param regularization: regularization coefficient
        :param epoch: number of learning iterations
        :return: None
        """
        self.alpha = alpha
        self.regularization = regularization
        self.epoch = epoch

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

    def evaluation(self, data_for_evaluation: (list, list), metric="confusion_matrix"):

        def confusion_matrix(data: (list, list),
                             lr_model: LogisticRegression) -> (list, float, float):
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            features, target = data

            for idx in range(len(features)):
                prediction = lr_model.predict(features[idx])
                if prediction == 1 and target[idx] == 1:
                    true_positive += 1
                elif prediction == 1 and target[idx] == 0:
                    false_positive += 1
                elif prediction == 0 and target[idx] == 1:
                    false_negative += 1
                else:
                    true_negative += 1

            m = len(features)
            matrix = [[round(true_positive / m, lr_model.ROUND_AFTER_COMA),
                       round(false_positive / m, lr_model.ROUND_AFTER_COMA)],
                      [round(false_negative / m, lr_model.ROUND_AFTER_COMA),
                       round(true_negative / m, lr_model.ROUND_AFTER_COMA)]]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            return matrix, precision, recall

        if not self.coefficients:
            print("There is no coefficients for prediction. Train model first")
            return [-1]

        if metric == "confusion_matrix":
            # conf_matrix, precision, recall = confusion_matrix(data_for_evaluation)
            return confusion_matrix(data_for_evaluation, self)

        return -1.0

    def fit(self, method="gd", scaled_coefficients=False) -> list:
        """
        Train our model
        :param scaled_coefficients: do we wanna to have coeff in range -1..1
        :param method: "gd" - gradient descent
        :return: optimized coefficients
        """

        # create new set of random coefficients
        self.create_coefficients_array(scaled_coefficients)

        if method == "fancy_algo":
            coefficients_optimization = self.fancy_algo
        else:
            coefficients_optimization = self.gradient_descent
            if len(self.__temporary_coefficients) != len(self.coefficients):
                self.__temporary_coefficients = self.coefficients.copy()

        _mod = 0

        # update Logs with model config
        self._log_storage = self._Logs(self.alpha, self.regularization, self.epoch)

        for e in range(self.epoch):
            coefficients_optimization()
            if _mod == self.epoch // 4:
                _mod = 0
                print("Iteration {} done".format(e + 1))
            _mod += 1
            # if we want to log it
            if self._log_flag is True:
                # store data for evaluation
                self.take_model_snapshot()
        print("Training is completed with {} iterations".format(self.epoch))

        return self.coefficients

    def predict(self, new_feature_data_line, raw_output=False):
        """
        :param raw_output: do we want to convert predicted value to 0 or 1?
        :param new_feature_data_line: data line of features input
        :return: predicted target value based on model coefficients, otherwise return -1 if error
        """
        if not self.coefficients or len(new_feature_data_line) != len(self.coefficients) - 1:
            print("No coefficients. Train model or too many features passed through")
            return -1
        prediction = self.sigmoid(self._get_hypothesis(new_feature_data_line))
        prediction = round(prediction, self.ROUND_AFTER_COMA)
        if raw_output:
            return prediction
        return 1 if prediction >= self.logistic_threshold else 0

    def sigmoid(self, hypothesis_value: float):
        return round(1 / (1 + pow(math.e, hypothesis_value)), self.ROUND_AFTER_COMA)

    def gradient_descent(self) -> None:
        """
        Adjust each coefficient according to the following formula:
        omega_i = omega_i * (1 - alpha * (reg/m) - (alpha/m) * SUM((hypothesis(x) - y)*(hypothesis(omega)')
        """
        regularization_coefficient = 1 - (self.alpha * (self.regularization / self.get_number_of_training_rows()))
        for coefficient_idx in range(len(self.coefficients)):
            self.__temporary_coefficients[coefficient_idx] = self.coefficients[coefficient_idx] \
                                                             * regularization_coefficient - self.alpha \
                                                             * self._get_cost_function_derivative(coefficient_idx)
            self.__temporary_coefficients[coefficient_idx] = round(self.__temporary_coefficients[coefficient_idx],
                                                                   self.ROUND_AFTER_COMA)
        self.coefficients = self.__temporary_coefficients.copy()  # update coefficients

    def fancy_algo(self):
        """Might be something else instead of gradient descent"""
        pass

    def get_cost_function(self, main_source="training"):
        """
        Cost function = SUM( (hypothesis(x) - y)^2 ) / (2*m)
        for i from 0 to m - number of lines
        :param main_source:
        :return:
        """

        if main_source == self._TESTING:
            m_row = self.get_number_of_testing_rows()
            features_data, target_data = self._testing_features_data, self._testing_target_data
        else:
            m_row = self.get_number_of_training_rows()
            features_data, target_data = self._training_features_data, self._training_target_data

        cost_function = 0
        counts = 0

        for idx in range(m_row):
            y = target_data[idx]
            hypothesis = self._get_hypothesis(features_data[idx])
            temp = y * math.log(hypothesis) + (1 - y) * math.log(1 - hypothesis)
            cost_function += round(temp, self.ROUND_AFTER_COMA)
            if cost_function >= INT32_MAX:
                counts += 1
                cost_function %= INT32_MAX
        cost_function = round((cost_function / m_row), self.ROUND_AFTER_COMA)
        for c in range(counts):
            cost_function += round((INT32_MAX / m_row), self.ROUND_AFTER_COMA)
        return -cost_function

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

    def _get_cost_function_derivative(self, current_coefficient_idx: int) -> float:
        """
        derivative of Cost function J = SUM((hypothesis(x_i) - y_i) * x_i_j) / m for i from 0 to m
        where j - coefficient index, m - number of rows in our training data
        :param current_coefficient_idx: coefficient index
        :return:
        """
        m_row = self.get_number_of_training_rows()
        features_data, target_data = self._training_features_data, self._training_target_data
        cost_function_derivative = 0
        for idx in range(m_row):
            if current_coefficient_idx == LogisticRegression._BIAS_INDEX:
                der_part = 1
            else:
                der_part = features_data[idx][current_coefficient_idx - 1]
            cost_function_derivative += round(((self._get_hypothesis(features_data[idx]) - target_data[idx])
                                               * der_part),
                                              self.ROUND_AFTER_COMA)
        return round(cost_function_derivative / m_row, self.ROUND_AFTER_COMA)

    def _get_hypothesis(self, features_data_line) -> float:
        """
        Calculate current hypothesis sum with current coefficients
        :param features_data_line: line of features from main table
        :return: hypothesis sum
        """
        n_features = len(features_data_line)
        hypothesis = self.coefficients[0]
        for idx in range(1, n_features + 1):
            coefficient = self.coefficients[idx]
            feature_value = features_data_line[idx - 1]
            hypothesis += round(coefficient * feature_value, self.ROUND_AFTER_COMA)
        return hypothesis

    def _calculate_current_cost_functions(self):
        """Calculate cost functions from current coefficients for testing and training data"""
        self.cost_testing_function = self.get_cost_function(main_source=self._TESTING)
        self.cost_training_function = self.get_cost_function()

    def _update_hypothesis_power(self):
        """
        Scan features and try to define the highest hysteresis power.
        h(x) = x1 + x2^3 + x3^0.5 => highest power is 3
        :return: highest power
        """
        max_power = -1
        for feature_name in self.labels[0]:
            feature_parts = feature_name.split('^')
            if len(feature_parts) <= 1:
                continue
            else:
                power = feature_parts[-1]
                max_power = max(float(power[1:-1]), max_power)
        self._d = max_power if int(max_power) != -1 else 1.0
