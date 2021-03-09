"""AI dev

linear regression

Data is taken from /data

"""

import time

# import pandas as pd
# import numpy as np
import helper_methods as helper


class DataTable:
    def __init__(self, file_path=None):
        self.head = []
        self.data = {}
        if file_path is not None:
            self.open_table(file_path)
        self.features = []
        self.target = None

    def __repr__(self):
        if self.head:
            for name in self.head:
                print(name, end=' ')
            print("")
            print_pretty = "Its shape is {}x{}".format(len(self.data), len(self.data[0]))
        else:
            print_pretty = "There is no data available. Please, upload data."
        return print_pretty

    def open_table(self, file_path):

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
                try:
                    array_of_strings[idx] = float(array_of_strings[idx])
                except:
                    array_of_strings[idx] = 0.0
            return array_of_strings

        def _split_clean(_line):
            """
            :param _line: "apple,orange,something\n"
            :return: ["apple", "orange", "something"]
            """
            _line = _remove_slash_n(_line)
            _line_split = _line.split(',')
            return _line_split

        file = open(file_path, "r")
        lines = file.readlines()
        file.close()

        self.head = _split_clean(lines[0])

        self.create_data_structure(len(lines))

        line_idx = 0

        while len(lines) > 1:
            line = lines.pop()
            line_split = _split_clean(line)

            line_split_float = _str2float(line_split)

            for idx in range(len(self.head)):
                name_col = self.head[idx]
                number = line_split_float[idx]
                self.data[name_col][line_idx] = number
            line_idx -= 1
        lines = None

    def create_data_structure(self, m):
        for name in self.head:
            self.data[name] = [0] * m

    def add_feature(self):
        pass

    def normalization(self):
        pass

    def plot(self):
        pass

    def select_features(self, feature_name):

        def _validate_feature_name(_name, _head):
            return _name in _head

        if _validate_feature_name(feature_name, self.head):
            self.features.append(feature_name)
            print("Feature {} was added".format(feature_name))
        else:
            the_most_right_name_edit = float("inf")
            proposed_name = None
            for name in self.head:
                min_edit = helper.levenshtein_distance(feature_name, name)
                if min_edit < the_most_right_name_edit:
                    the_most_right_name_edit = min_edit
                    proposed_name = name
            if proposed_name is not None:
                print("You made a typo mistake. Did you mean {}?".format(proposed_name))
                print("Type y/n")
                user_input = input()
                if user_input == 'y':
                    self.select_features(proposed_name)
            else:
                print("There is no data. Upload it before this operation")

    def select_target(self):
        if self.target


def linear_regression():
    import linear_regression


if __name__ == '__main__':
    # linear_regression()

    # Panda vs my implementation
    # start1 = time.time()
    # data = DataTable("test_data.csv")
    # end1 = time.time()
    #
    # start2 = time.time()
    # data2 = pd.read_csv("test_data.csv")
    # train = data2[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    # #
    # # train_x = np.asanyarray(train[['ENGINESIZE']])
    # # train_y = np.asanyarray(train[['CO2EMISSIONS']])
    # end2 = time.time()

    # print(end1-start1)
    # print(end2-start2)

    data = DataTable("test_data.csv")
    print(data)

    # preprocessing

    # set features

    # add features as x1*x1 or x1*x2
    # additional for loop for each line to calculate new features
    # and append them to existing data table

    # # # split data set on 80% / 20%
    # msk = np.random.rand(len(df)) < 0.8
    # print(msk)
    # train = cdf[msk]
    # test = cdf[~msk]
