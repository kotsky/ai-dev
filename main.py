"""AI dev

linear regression

Data is taken from /data

"""

import time
import pandas as pd
import numpy as np


class DataTable:
    def __init__(self):
        self.head = []
        self.data = []


def open_table(file_path):

    def _remove_slash_n(string):
        """
        :param string: "something\n"
        :return: "something"
        """
        string = string[:-2]
        return string

    def _str2float(array_of_strings):
        """
        :param array_of_strings: ['2.3', '4']
        :return: [2.3, 4.0]
        """
        for idx in range(len(array_of_strings)):
            array_of_strings[idx] = float(array_of_strings[idx])
        return array_of_strings

    def _split_clean(line):
        """
        :param line: "apple,orange,something\n"
        :return: ["apple", "orange", "something"]
        """
        line = _remove_slash_n(line)
        line_split = line.split(',')
        return line_split

    file = open(file_path, "r")
    lines = file.readlines()
    file.close()
    table_of_data = DataTable()
    while len(lines) > 1:
        line = lines.pop()
        line_split = _split_clean(line)
        table_of_data.data.append(_str2float(line_split))
    table_of_data.head = _split_clean(lines[0])

    return table_of_data


def linear_regression():
    import linear_regression


if __name__ == '__main__':
    # linear_regression()
    start1 = time.time()
    data = open_table("test_data.csv")
    end1 = time.time()

    start2 = time.time()
    data2 = pd.read_csv("test_data.csv")
    train = data2[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    end2 = time.time()

    print(end1-start1)
    print(end2-start2)
