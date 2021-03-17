import matplotlib.pyplot as plt
import regression as rg


def levenshtein_distance(str1: str, str2: str) -> int:
    """
    Return the minimum edits needs to do to
    change str1 to str2.
    :param str1: string
    :param str2: string
    :return: min edits
    """
    matrix = [0]
    for i in range(len(str1)):
        matrix.append(i + 1)
    matrix = [matrix]
    for j in range(len(str2)):
        matrix.append([j + 1] * (len(str1) + 1))

    for i in range(len(str1)):
        for j in range(len(str2)):
            if str2[j] == str1[i]:
                matrix[j + 1][i + 1] = matrix[j][i]
            else:
                matrix[j + 1][i + 1] = min(matrix[j][i + 1], matrix[j + 1][i], matrix[j][i]) + 1

    return matrix[-1][-1]


def check_spelling_helper(word: str, set_of_words) -> str:
    """
    Using levenshtein_distance() check which word from
    set_of_words is the most closest to the given word.
    :param word: given word as a string
    :param set_of_words: iterative entity of strings
    :return: the most closest word in string
    """
    the_most_right_name_edit = float("inf")
    proposed_name = None
    for reference in set_of_words:
        min_edit = levenshtein_distance(word, reference)
        if min_edit < the_most_right_name_edit:
            the_most_right_name_edit = min_edit
            proposed_name = reference
    return proposed_name


def column2list(matrix: list, column_idx: int) -> list:
    """
    Convert column from features 2D matrix to 1D list of
    that feature.
    :param matrix: 2D square array of number
    :param column_idx: column index in the matrix
    :return:
    """
    if len(matrix) <= 1:
        return matrix if matrix else []

    if column_idx >= len(matrix[0]):
        return [-1]

    new_column = []

    for line in matrix:
        new_column.append(line[column_idx])
    return new_column


def plot2d_target2predict(regression_model: rg.Regression, test_features: list,
                          test_target: list, feature_name: str, target_name: str,
                          feature_idx: int, loc_place="upper left"):
    """Draw script target vs prediction from a regression model defined by rg.Regression"""
    predicted = []
    for test in test_features:
        predicted.append(regression_model.predict(test))
    axis1 = column2list(test_features, feature_idx)
    plt.title("Config: alpha = {}, regul = {}, epoch = {}".format(regression_model.alpha,
                                                                  regression_model.regularization,
                                                                  regression_model.epoch))
    plt.scatter(axis1, test_target, label="Target")
    plt.scatter(axis1, predicted, label="Predicted")
    plt.legend(loc=loc_place)
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.show()


"""
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title('Axis [0, 0]')
    axs[0, 1].plot(x, y, 'tab:orange')
    axs[0, 1].set_title('Axis [0, 1]')
    axs[1, 0].plot(x, -y, 'tab:green')
    axs[1, 0].set_title('Axis [1, 0]')
    axs[1, 1].plot(x, -y, 'tab:red')
    axs[1, 1].set_title('Axis [1, 1]')
    
    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
"""