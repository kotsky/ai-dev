
import random


def linear_regression(features, target, epoch, alpha=1, reg=None):

    coefficients = create_coefficients_array(features[0])

    for e in range(epoch):
        for line_idx in range(len(features)):
            coefficients_optimization(coefficients, features, target, alpha, reg)

    return coefficients


def coefficients_optimization(coefficients, features, target, alpha, reg):

    for coefficient_idx in range(len(coefficients)):
        coefficients[coefficient_idx] = coefficients[coefficient_idx] - alpha * \
                                        cumulative_sum(coefficients, features, target, alpha, reg, coefficient_idx)


def cumulative_sum(coefficients, features, target, alpha, reg, coefficient_idx):

    derivative_part = 10


def create_coefficients_array(features):
    n = len(features)
    coefficients = features.copy()
    for idx in range(n):
        coefficients[n] = random.random()
    return coefficients

