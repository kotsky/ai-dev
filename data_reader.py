"""Data table build

Custom data structure building for AI training.

See DataTable description.

Author: kotsky

"""

import helper_methods as helper


class DataTable:
    """

    DataTable is a data structure, which is generated from only
    numerical table-like data in a way, that it has the following attributes:

        head - shows all head names of each column

        table - contains _DataColumn object in dict way:
        table: { "Column_name1": _DataColumn obj,
        "Column_name2": _DataColumn obj, ... }

            _DataColumn obj contains:
                - .data: list
                - .max: float
                - .min: float
                - .mean: float
                - .scale_value - after scaling

        features - list which contains pointers on table,
        which user picked up as his features for learning (our x)

        target - column of control value, desired outcome of our AI model, (our y)

    Methods to use:
        * open_file(file_path)
        * activate_feature(feature_name: str) - let user decide which feature to use in training set
        * deactivate_feature(feature_name: str) - let user decide which feature to disable in training set
        * add_new_feature([feature_name1, feature_name2, ...]: list of str, power: optional float) -
        user is free to add more features. Combine new feature from presented or
        make in power "power" some feature. This power activate new feature to be
        used in training set by default.
        Example:
            - add_new_feature(["feat1", "feat2"]) -> new feature feat1*feat2 will be created and
            - added to main table as new column.
            - add_new_feature(["feat1"], 2) -> new feature = feat1^2
        * select_target(target_name) - let user decide which outcome parameter is desired
        * max_scaling() - do scaling -1 <= x <= 1 of our table itself
        * plot(axis_name1: str, axis_name2: str) - show some figures of target vs features dependencies
        * copy() - to duplicate whole entire data structure DataTable
        * split_data(training_coeff: float) - return pointers, which shows which part of data is used for training/
          testing/cross-validation
        * shuffle() - to shuffle data
        * .ROUND_AFTER_COMA: int - to set user's desired round value. Default = 4
        * get_training_data() - returns training features set (defined by the user) and target set as arrays
        * get_cv_data() - returns CV features set (defined by the user) and target set as arrays
        * get_testing_data() - returns training features set (defined by the user) and target set as arrays
        * get_labels() - return training set labels and target name in same order as training set was generated

    """

    class _DataColumn:
        """

        Additional entity to have a clear solution.
        Represent a column of a main data table.

        """

        ROUND_AFTER_COMA = 2

        def __init__(self, _data, mean=0, _max=float("-inf"),
                     _min=float("inf"), _is_scaled=False,
                     _is_centred=False, _scale_value=None):
            self.data = _data
            self.mean = mean
            self.max = _max
            self.min = _min
            self._is_scaled = _is_scaled
            self._is_centred = _is_centred
            self.scaled_value = _scale_value

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            self.count = 0
            self.end_count = len(self)
            return self

        def __next__(self):
            if self.count > self.end_count:
                raise StopIteration
            else:
                self.count += 1
                return self.count - 1

        def copy(self):
            new_entity = DataTable._DataColumn(self.data.copy(), self.mean, self.max,
                                               self.min, self._is_scaled, self._is_centred,
                                               self.scaled_value)
            return new_entity

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
            self.mean = round((self.mean / len(self)), self.ROUND_AFTER_COMA)

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
            self.scaled_value = scaling_coefficient
            if self.scaled_value == 0:
                return
            for idx in range(len(self)):
                self.data[idx] = round((self.data[idx] / scaling_coefficient),
                                       self.ROUND_AFTER_COMA)
            # update min-max-mean
            self.attribute_calculation()
            self._is_scaled = True

    _data_is_scaled: bool

    _YES = 'y'
    _NO = 'n'
    _TRAINING = "training"
    _CV = "cv"
    _TESTING = "testing"

    # CONFIG
    ROUND_AFTER_COMA = 2  # 0.455555=> 0.4556

    def __init__(self, file_path=None):
        self.head = []
        self.table = {}
        self.file_path = file_path
        if file_path is not None:
            self.open_table(file_path)
        self.features = {}
        self.target = {}
        self._data_is_scaled = False
        self._split_pointers = {self._TRAINING: [[0, 0], False],
                                self._CV: [[0, 0], False],
                                self._TESTING: [[0, 0], False]}

    def __repr__(self):
        if self.head:
            for name in self.head:
                label = ' '
                if name in self.features:
                    label = "(f) "  # feature label
                elif self.target is not None and name in self.target:
                    label = "(t) "  # target label
                print(name, end=label)
            print("")
            print_pretty = "Its shape is {}x{}".format(len(self.head),
                                                       len(self))
        else:
            print_pretty = "There is no table available. Please, upload table."
        return print_pretty

    def __len__(self):
        return len(self.table[self.head[0]])

    def is_split(self) -> bool:
        """
        Did we split our data on training and test sets?
        :return True if data was split:
        """
        return self._split_pointers[self._TRAINING][1]

    def copy(self):
        """
        Copy entire data structure with brand new data location.
        :return: DataTable structure with copied data
        """

        def _deepcopy(tree: dict) -> dict:
            new_tree = {}
            for key, value in tree.items():
                new_value = value.copy()
                new_tree[key] = new_value
            return new_tree

        def _repointing(keys: list, new_table: dict) -> dict:
            main_tree = {}
            for key in keys:
                main_tree[key] = new_table[key]
            return main_tree

        new_structure = DataTable()
        new_structure.head = self.head.copy()
        new_structure.table = _deepcopy(self.table)
        new_structure.file_path = self.file_path
        new_structure.features = _repointing(list(self.features.keys()), new_structure.table)
        new_structure.target = _repointing(list(self.target.keys()), new_structure.table)
        new_structure._data_is_scaled = self._data_is_scaled
        new_structure._split_pointers = _deepcopy(self._split_pointers)
        return new_structure

    def open_table(self, file_path=None):
        """
        Upload data to main memory
        :param file_path: file path in the project
        :return: uploaded data and generated DataTable
        """

        def _remove_slash_n(string):
            """
            :param string: "something\n"
            :return: "something"
            """
            string = string[:-1]
            return string

        def _str2float(array_of_strings):
            """
            :param array_of_strings: ['2.3', '4']
            :return: [2.3, 4.0]
            """
            for _idx in range(len(array_of_strings)):
                try:
                    array_of_strings[_idx] = float(array_of_strings[_idx])
                except:
                    # TODO modification to read words for classification problems
                    array_of_strings[_idx] = 0.0
            return array_of_strings

        def _split_clean(_line):
            """
            :param _line: "apple,orange,something\n"
            :return: ["apple", "orange", "something"]
            """
            _line = _remove_slash_n(_line)
            _line_split = _line.split(',')
            return _line_split

        if not self.file_path:
            print("Specify file path")
            return

        file = open(file_path, "r")
        lines = file.readlines()
        file.close()
        self._data_is_scaled = False
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
        # do mean calculation
        for column_name in self.head:
            column = self.table[column_name]
            column.mean = round(column.mean / len(column), self.ROUND_AFTER_COMA)

    def _get_split_pointers(self):
        """
        Get set of split pointers.
        :return: cv_flag shows if we have cross-validation set,
        also, returns set of pointers
        """

        if self.is_split() is False:
            print("Data is not split. Prepare data first.")
            return

        cv_flag = self._split_pointers[self._CV][1]
        if cv_flag is True:
            set_of_pointers = [self._split_pointers[self._TRAINING][0],
                               self._split_pointers[self._CV][0],
                               self._split_pointers[self._TESTING][0]]
        else:
            set_of_pointers = [self._split_pointers[self._TRAINING][0],
                               self._split_pointers[self._TESTING][0]]
        return cv_flag, set_of_pointers

    def _get_training_pointers(self):
        return self._split_pointers[self._TRAINING][0] if self._split_pointers[self._TRAINING][1] else -1

    def _get_cv_pointers(self):
        return self._split_pointers[self._CV][0] if self._split_pointers[self._CV][1] else -1

    def _get_testing_pointers(self):
        return self._split_pointers[self._TESTING][0] if self._split_pointers[self._TESTING][1] else -1

    def _generate_data_by_pointers(self, mode=None):
        """
        Generate a brand new data sets for further usage
        :param mode: training / cv / testing as string
        :return: features set, target set as arrays of data
        """
        if mode == self._TESTING:
            set_of_pointers = self._get_testing_pointers()
        elif mode == self._CV:
            set_of_pointers = self._get_cv_pointers()
        else:
            set_of_pointers = self._get_training_pointers()

        if set_of_pointers == -1:
            print("There is no such data generated. Split first")
            return
        if not self.features or not self.target:
            print("Data is not ready. Add features and target")
            return
        start_p, end_p = set_of_pointers  # inclusively
        features_set = []
        target_set = []
        target_name = self._get_target_name()
        for line_idx in range(start_p, end_p + 1):
            features_line = []
            for feature_name in self.features:
                features_line.append(self.features[feature_name].data[line_idx])
            features_set.append(features_line)
            target_set.append(self.target[target_name].data[line_idx])
        return features_set, target_set

    def get_training_data(self) -> (list, list):
        """
        Generate training set from defined features and target
        :return: features set, target set as arrays
        """
        return self._generate_data_by_pointers()

    def get_cv_data(self) -> (list, list):
        """
        Generate cross-validation set from defined features and target
        :return: features set, target set as arrays
        """
        return self._generate_data_by_pointers(mode="cv")

    def get_testing_data(self) -> (list, list):
        """
        Generate testing set from defined features and target
        :return: features set, target set as arrays
        """
        return self._generate_data_by_pointers(mode="testing")

    def get_labels(self) -> (str, str):
        """

        :return: training set labels in same order as training set and target name
        """
        if not self.features or not self.target:
            print("Define training set first. Add features and target")
            return
        features_label = []
        for feature_name in self.features:
            features_label.append(feature_name)
        target_name = self._get_target_name()
        return features_label, target_name

    def get_column_data(self, column_name: str) -> list:
        """
        Get data from the given column
        :param column_name: name from data table
        :return: list of data
        """

        if column_name not in self.head:
            return []

        return self.table[column_name].data

    def plot(self, parameter1=None, parameter2=None, features2target=False, all2target=False) -> None:
        """
        Plot 2D pictures.
        :param parameter1: axis 1 column name
        :param parameter2: axis 2 column name
        :param features2target: plot all features to target
        :param all2target: plot all to target
        :return: figures 2D
        """

        if all2target is True:
            if not self.target:
                print("There is no defined target. Please, select one")
                return
            target_name = self._get_target_name()
            for column_name in self.head:
                if column_name in self.target:
                    continue
                self._plot2d_helper(column_name, target_name, "blue")

        elif features2target is True:
            if not self.target:
                print("There is no defined target. Please, select one")
                return
            target_name = self._get_target_name()
            for feature_name in self.features:
                if feature_name in self.target:
                    continue
                self._plot2d_helper(feature_name, target_name, "red")

        elif parameter1 is not None and parameter2 is not None:
            if parameter1 in self.table and parameter2 in self.table:
                self._plot2d_helper(parameter1, parameter2, "green")

    def _get_target_name(self) -> str:
        target_name = list(self.target.keys())[0]
        return target_name

    def split_data(self, training_size: float, cv_size=None, shuffle=False) -> None:
        """
        Split data according to user's preferences.
        :param training_size: 0.3 - 0.9 desired part of data to use for AI training
        :param cv_size: cross-validation data part to test different algorithms
        :param shuffle: do we want to shuffle first? True/False
        :return: assigned pointers is self._split_pointers which shows how the data
            is split on training/cv/testing sets. This is nice to do instead of
            copying data
        """

        if not 0.3 <= training_size <= 0.9 or (training_size + cv_size) >= 0.95:
            print("Wrong train-test-cv attitude")
            return None

        if shuffle is True:
            self.shuffle()

        m = len(self)

        tr_p_st = 0
        tr_p_end = int(m * training_size)
        self._split_pointers[self._TRAINING] = [[tr_p_st, tr_p_end], True]

        ts_p_st = tr_p_end + 1
        ts_p_end = m - 1

        if cv_size is not None:
            cv_part = int(cv_size * m)
            cv_p_st = tr_p_end + 1
            cv_p_end = cv_p_st + cv_part
            self._split_pointers[self._CV] = [[cv_p_st, cv_p_end], True]
            ts_p_st = cv_p_end + 1

        self._split_pointers[self._TESTING] = [[ts_p_st, ts_p_end], True]

        if cv_size is not None:
            print("Data was split as follows: {} training set, {} cross-validation set and {} test set".
                  format(training_size, cv_size, (1 - training_size - cv_size)))
        else:
            print("Data was split as follows: {} training set and {} test set".
                  format(training_size, 1 - training_size))

    def shuffle(self) -> None:
        """
        Random data shuffle.
        :return: shuffled data
        """

        def _swap(array, idx1, idx2):
            array[idx1], array[idx2] = array[idx2], array[idx1]

        import random
        for idx in range(len(self)):
            random_idx1 = random.randint(0, len(self) - 1)
            random_idx2 = random.randint(0, len(self) - 1)
            while random_idx1 == random_idx2:
                random_idx2 = random.randint(0, len(self) - 1)
            for column_name in self.head:
                column_obj = self.table[column_name]
                _swap(column_obj.data, random_idx1, random_idx2)
        print("Shuffle was done")

    def add_new_feature(self, features, power=None) -> None:
        """
        Add new desired feature to use.
        :param features: feature str or list of features from main table, which user might
            want to combine to create a new feature like x3 = x1 * x2,
            where x3 - new feature, x1 and x2 - features from main table
        :param power: specific command to perform:
            if None -> new_feature = features[0] * features[1] * ...
            if 0.5 -> new_feature = sqrt(features[0]) for feature[0] >= 0
            if positive int -> new_feature = pow(features[0], command)
        :return: new column of added feature as native one
        """

        def _validate_feature_name(_name: str, _head: dict):
            return _name in _head

        if type(features) == str:
            features = [features]

        if features is None:
            print("Type features' names in a list format")
            return

        new_feature_name = ''
        new_column_obj = None

        if power is None:
            _validation_check = False
            for feature_name in features:
                if not _validate_feature_name(feature_name, self.table):
                    proposed_feature_name = helper.check_spelling_helper(feature_name, self.head)
                    user_input = self.__user_confirmation(feature_name, proposed_feature_name)
                    if user_input[0].lower() == self._YES:

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
                        new_column_obj.data[idx] = round((new_column_obj.data[idx] + new_data.data[idx]),
                                                         self.ROUND_AFTER_COMA)

                new_feature_name += "*" + feature_name if len(new_feature_name) > 0 else feature_name

            if new_column_obj is not None and _validation_check is True:
                self._add_feature_helper(new_feature_name, new_column_obj)
            else:
                if _validation_check is False:
                    print("We cannot create same feature as we have in our main table")
                else:
                    print("Please, write write power input")
        else:
            if power <= 0:
                print("Set write power as a positive number")
                return
            feature_name = features[0]
            new_feature_name = feature_name + '^' + "({})".format(power)
            new_column_obj = self.table[feature_name].copy()
            for idx in range(len(new_column_obj)):
                new_column_obj.data[idx] = round(pow(new_column_obj.data[idx], power),
                                                 self.ROUND_AFTER_COMA)
            self._add_feature_helper(new_feature_name, new_column_obj)

    def max_scaling(self, column_name=None) -> None:
        """
        Min-Max scaling of assigned column or all table.
        :param column_name: string column name which we want to scale
        :return: None
        """
        if column_name is not None:
            column = self.table[column_name]
            column.scaling()
            print("Column {} was scaled".format(column_name))
        else:
            if self._data_is_scaled is True:
                return
            self._data_is_scaled = True
            for column_name in self.table:
                column = self.table[column_name]
                column.scaling()
                print("Column {} was scaled".format(column_name))

    def deactivate_feature(self, feature_name):
        """
        Remove feature from the training set.
        :param feature_name: feature name as a string or list of strings
        :return: None
        """

        def _validate_feature_name(_name: str, _head: dict):
            return _name in _head

        if type(feature_name) == list:
            for internal_feature_name in feature_name:
                self.deactivate_feature(internal_feature_name)
            return

        if _validate_feature_name(feature_name, self.table):
            if feature_name in self.features:
                del self.features[feature_name]
            print("Feature {} was disabled from the training set".format(feature_name))
        else:
            proposed_name = helper.check_spelling_helper(feature_name, self.head)
            if proposed_name is not None:
                print("You made a typo mistake. Did you mean {}?".format(proposed_name))
                print("Type y/n")
                user_input = input()
                if user_input[0].lower() == self._YES:
                    self.deactivate_feature(proposed_name)
            else:
                print("Nothing was done")

    def select_target(self, target_name: str):
        """
        Select target to be used from self.table for AI.
        :param target_name: target name per table as string
        :return: None
        """

        if not len(self.target):
            self.activate_features(target_name, is_target=True)
        else:
            if not target_name == self._get_target_name():
                print("Do you want to replace existed {} target? Enter y/n".format(self.target))
                user_input = input()
                if user_input[0].lower() == self._YES:
                    self.activate_features(target_name, is_target=True)

    def activate_features(self, feature_name, is_target=False) -> None:
        """
        Select feature to be used from self.table for AI.
        :param feature_name: feature name per table as string or list of features string
        :param is_target: are we setting feature or target?
        :return: None
        """

        def _validate_feature_name(_name: str, _head: dict):
            return _name in _head

        if type(feature_name) == list:
            for internal_feature_name in feature_name:
                self.activate_features(internal_feature_name)
            return

        if _validate_feature_name(feature_name, self.table):
            if is_target is False:
                self.features[feature_name] = self.table[feature_name]
                print("Feature {} was added".format(feature_name))
            else:
                self.target[feature_name] = self.table[feature_name]
                print("Target {} was added".format(feature_name))
        else:
            proposed_name = helper.check_spelling_helper(feature_name, self.head)
            if proposed_name is not None:
                print("You made a typo mistake in {}. Did you mean {}?".format(feature_name, proposed_name))
                print("Type y/n")
                user_input = input()
                if user_input[0].lower() == self._YES:
                    self.activate_features(proposed_name, is_target)
            else:
                print("There is no table. Upload it before this operation")

    def _plot2d_helper(self, axis1: str, axis2: str, colour: str) -> None:
        """
        Actual 2D plot.
        :param axis1: column name
        :param axis2: column name
        :param colour: "blue", "green", etc.
        :return: pictures
        """
        import matplotlib.pyplot as plt
        plt.scatter(self.table[axis1].data, self.table[axis2].data, color=colour)
        plt.xlabel(axis1)
        plt.ylabel(axis2)
        plt.show()

    def _add_feature_helper(self, new_feature_name: str, new_column_obj: _DataColumn) -> None:
        """
        Add feature column to main data table.
        :param new_feature_name: its feature name as a string
        :param new_column_obj: _DataColumn obj with data
        :return: new features in main data
        """
        new_column_obj.attribute_calculation()
        self.table[new_feature_name] = new_column_obj
        self.head.append(new_feature_name)
        self.features[new_feature_name] = new_column_obj
        print("New created feature {} was added".format(new_feature_name))
        print("This {} feature is added to the list of training set".format(new_feature_name))
        self._data_is_scaled = False

    def __create_data_structure(self, m: int) -> None:
        for name in self.head:
            self.table[name] = self._DataColumn(m * [0])
            self.table[name].ROUND_AFTER_COMA = self.ROUND_AFTER_COMA

    @staticmethod
    def __user_confirmation(word: str, proposed_word: str):
        """
        Script to ask user if proposed word is ok or not.
        :param word:
        :param proposed_word:
        :return:
        """
        if proposed_word is None:
            print("There is no table. Please, provide it first")
            return
        print("You made a typo mistake in {}. Did you mean {}?".format(word, proposed_word))
        print("Type y/n")
        user_input = input()
        return user_input


if __name__ == '__main__':
    table = DataTable("test_data.csv")
    print(table)
    head = table.head
    table.activate_features([head[1]])
    table.activate_features(head[3])
    table.select_target(head[-1])
    # table.add_new_feature([head[0], head[4]])
    table.add_new_feature([head[1]], 2)
    # table.deactivate_feature(head[1])

    table.split_data(0.6, 0.2)
    table.shuffle()
    print(table)

    # table.plot(features2target=True)
    table.plot(all2target=True)
    # table.plot(head[0], head[1])
    print()

    table2 = table.copy()
    table2.max_scaling()
    table.add_new_feature(head[0], power=0.5)

    scaled_training_data = table2.get_training_data()
    training_data = table.get_training_data()
    labels = table.get_labels()
    print(labels)
