"""AI dev

linear regression

Data is taken from /table

TODO list
- verify all DataTable methods
- do plot()
- integrate it to AI models
- do UI console interface

"""

import helper_methods as helper


class DataTable:
    """

    DataTable is generated from table-like file in a way,
    that it has the following attributes:
        head - shows all head names of each column
        table - contains columns in a dict way like
            table: { "Column_name1": DataColumn obj,
                    "Column_name2": DataColumn obj, ... }
        features - list which contains pointers on table,
            which user picked up as his features for learning
            (our x)
        target - column of control value, desired outcome of our AI model,
            (our y)

    methods to use:
        open_file(file_path)
        select_feature(feature_name: str) - let user decide which feature to use
        add_feature([feature_name1, feature_name2, ...]: list of str)
                    - user is free to add more features
        select_target(target_name) - let user decide which outcome parameter is desired
        scaling() - do scaling of our table itself
        plot() - show some figures of target vs features dependencies

    """

    class DataColumn:
        """

        Additional entity to have a clear solution.
        Represent a column of a main data table.

        """

        def __init__(self, _data, mean=0, _max=float("-inf"),
                     _min=float("inf"), _is_scaled=False, _is_centred=False):
            self.data = _data
            self.mean = mean
            self.max = _max
            self.min = _min
            self._is_scaled = _is_scaled
            self._is_centred = _is_centred

        def __len__(self):
            return len(self.data)

        def __copy__(self):
            return DataTable.DataColumn(self.data, self.mean, self.max,
                                        self.min, self._is_scaled, self._is_centred)

        def reset(self):
            """
            Reset attributes
            :return: None
            """
            self._is_scaled = False
            self._is_centred = False
            self.min = float("inf")
            self.max = float("-inf")
            self.mean = 0

        def attribute_calculation(self):
            """
            Calculate attributes once having data
            :return: None
            """
            self.reset()
            for number in self.data:
                self.min = min(self.min, number)
                self.max = max(self.max, number)
                self.mean += number
            self.mean = round((self.mean / len(self)), DataTable.ROUND_AFTER_COMA)

        def scaling(self):
            """
            Does data scaling based on max(abs(max), abs(min))
            to fit range -1 <= x <= 1.
            Also, does increasing to same range if
            data is too small.
            :return: None | scaled column of data
            """
            if self._is_scaled is True:
                return
            scaling_coefficient = max(abs(self.max), abs(self.min))
            for idx in range(len(self)):
                self.data[idx] = round((self.data[idx] / scaling_coefficient),
                                       DataTable.ROUND_AFTER_COMA)

    _data_is_scaled: bool

    YES = 'y'
    NO = 'n'

    # Settings
    ROUND_AFTER_COMA = 4  # 0.455555=> 0.4556

    def __init__(self, file_path=None):
        self.head = []
        self.table = {}
        if file_path is not None:
            self.open_table(file_path)
        self.features = {}
        self.target = None
        self._data_is_scaled = False

    def __repr__(self):
        if self.head:
            for name in self.head:
                print(name, end=' ')
            print("")
            print_pretty = "Its shape is {}x{}".format(len(self.table), len(self.table[0]))
        else:
            print_pretty = "There is no table available. Please, upload table."
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
                    # TODO option -> to read the word as a class
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

        self.__create_data_structure(len(lines) - 1)

        line_idx = len(lines) - 1

        while len(lines) > 1:
            line = lines.pop()
            line_split = _split_clean(line)

            line_split_float = _str2float(line_split)

            for idx in range(len(self.head)):
                name_col = self.head[idx]
                number = line_split_float[idx]
                column = self.table[name_col]
                column.data[line_idx - 1] = number
                column.min = min(column.min, number)
                column.max = max(column.max, number)
                column.mean += number

            line_idx -= 1

        for column_name in self.head:
            # do mean calculation
            column = self.table[column_name]
            column.mean = round(column.mean / len(column), DataTable.ROUND_AFTER_COMA)

        lines = None

    def add_new_feature(self, features: list, command=None):
        """
        Add new desired feature to use.
        :param features: list of features from main table, which user might
            want to combine to create a new feature like x3 = x1 * x2,
            where x3 - new feature, x1 and x2 - features from main table
        :param command: specific command to perform:
            if None -> new_feature = features[0] * features[1] * ...
            if 0.5 -> new_feature = sqrt(features[0]) for feature[0] >= 0
            if positive int -> new_feature = pow(features[0], command)
        :return: new column of added feature as native one
        """

        def _validate_feature_name(_name: str, _head: dict):
            return _name in _head

        if features is None:
            print("Type features' names in a list format")
            return

        new_feature_name = ''
        new_column_obj = None

        if command is None:
            _validation_check = False
            for feature_name in features:
                if not _validate_feature_name(feature_name, self.table):
                    proposed_feature_name = helper.check_spelling_helper(feature_name, self.head)
                    user_input = self.__user_confirmation(proposed_feature_name)
                    if user_input[0].lower() == DataTable.YES:

                        feature_name = proposed_feature_name
                    else:
                        print("Skip {} feature".format(feature_name))
                        continue

                if new_column_obj is None:
                    new_column_obj = self.table[feature_name].copy()
                    new_column_obj.reset()
                else:
                    _validation_check = True
                    new_data = self.table[feature_name]
                    for idx in range(len(new_column_obj)):
                        new_column_obj.data[idx] *= new_data[idx]

                new_feature_name += "*" + feature_name if len(new_feature_name) > 0 else feature_name

            if new_column_obj is not None and _validation_check is True:
                self._add_feature_helper(new_feature_name, new_column_obj)
            else:
                if _validation_check is False:
                    print("We cannot create same feature as we have in our main table")
                else:
                    print("Please, write write command input")
        else:
            if command <= 0:
                print("Set write command as a positive number")
                return
            feature_name = features[0]
            new_feature_name = feature_name + '^' + "({})".format(command)
            new_column_obj = self.table[feature_name].copy()
            for idx in range(len(new_column_obj)):
                new_column_obj[idx] = pow(new_column_obj[idx], command)
            self._add_feature_helper(new_feature_name, new_column_obj)

    def scaling(self):
        if self._data_is_scaled is True:
            return
        self._data_is_scaled = True
        for column_name in self.table:
            column = self.table[column_name]
            column.scaling()
            print("Column {} was scaled".format(column_name))

    def plot(self):
        pass

    def deactivate_feature(self, feature_name: str):
        """
        Remove feature from the training set.
        :param feature_name: feature name as a string
        :return: None
        """

        def _validate_feature_name(_name: str, _head: dict):
            return _name in _head

        if _validate_feature_name(feature_name, self.table):
            del self.features[feature_name]
            print("Feature {} was disabled from the training set".format(feature_name))
        else:
            proposed_name = helper.check_spelling_helper(feature_name, self.head)
            if proposed_name is not None:
                print("You made a typo mistake. Did you mean {}?".format(proposed_name))
                print("Type y/n")
                user_input = input()
                if user_input[0].lower() == DataTable.YES:
                    self.deactivate_feature(proposed_name)
            else:
                print("Nothing was done")

    def select_target(self, target_name: str):
        """
        Select target to be used from self.table for AI.
        :param target_name: target name per table as string
        :return: None
        """

        if self.target is None:
            self.activate_features(target_name, is_target=True)
        else:
            print("Do you want to replace existed {} target? Enter y/n".format(self.target))
            user_input = input()
            if user_input[0].lower() == DataTable.YES:
                self.activate_features(target_name, is_target=True)

    def activate_features(self, feature_name: str, is_target=False):
        """
        Select feature to be used from self.table for AI.
        :param feature_name: feature name per table as string
        :param is_target: are we setting feature or target?
        :return: None
        """

        def _validate_feature_name(_name: str, _head: dict):
            return _name in _head

        if _validate_feature_name(feature_name, self.table):
            if is_target is False:
                self.features[feature_name] = self.table[feature_name]
                print("Feature {} was added".format(feature_name))
            else:
                self.target = feature_name
                print("Target {} was settled".format(feature_name))
        else:
            proposed_name = helper.check_spelling_helper(feature_name, self.head)
            if proposed_name is not None:
                print("You made a typo mistake. Did you mean {}?".format(proposed_name))
                print("Type y/n")
                user_input = input()
                if user_input[0].lower() == DataTable.YES:
                    self.activate_features(proposed_name, is_target)
            else:
                print("There is no table. Upload it before this operation")

    def _add_feature_helper(self, new_feature_name: str, new_column_obj: DataColumn):
        """
        Add feature column to main data table.
        :param new_feature_name: its feature name as a string
        :param new_column_obj: DataColumn obj with data
        :return:
        """
        new_column_obj.attribute_calculation()
        self.table[new_feature_name] = new_column_obj
        print("New created feature {} was added".format(new_feature_name))
        print("This {} feature is added to the list of train set".format(new_feature_name))
        self._data_is_scaled = False

    def __create_data_structure(self, m):
        for name in self.head:
            self.table[name] = DataTable.DataColumn(m * [0])

    @staticmethod
    def __user_confirmation(proposed_word):
        if proposed_word is None:
            print("There is no table. Please, provide it first")
            return
        print("You made a typo mistake. Did you mean {}?".format(proposed_word))
        print("Type y/n")
        user_input = input()
        return user_input


def linear_regression():
    pass


if __name__ == '__main__':
    # linear_regression()

    # Panda vs my implementation
    # start1 = time.time()
    # table = DataTable("test_data.csv")
    # end1 = time.time()
    #
    # start2 = time.time()
    # data2 = pd.read_csv("test_data.csv")
    # train = data2[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    # #
    # # train_x = np.asanyarray(train[['ENGINESIZE']])
    # # train_y = np.asanyarray(train[['CO2EMISSIONS']])
    # end2 = time.time()
    #
    # print(end1-start1)
    # print(end2-start2)

    table = DataTable("test_data.csv")
    print(table)

    # preprocessing

    # set features

    # add features as x1*x1 or x1*x2
    # additional for loop for each line to calculate new features
    # and append them to existing table table

    # # # split table set on 80% / 20%
    # msk = np.random.rand(len(df)) < 0.8
    # print(msk)
    # train = cdf[msk]
    # test = cdf[~msk]
