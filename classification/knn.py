"""K-Nearest Neighbors

Classification with KNN, optimised for huge k with heap.

Methods:
    * set_data([features, target])
    * predict()
    * evaluate()

"""


class KNN:

    ROUND_AFTER_COMA = 4

    class MaxHeap:

        class Node:
            def __init__(self, distance: float, class_idx: int):
                self.distance = distance
                self.class_idx = class_idx

        def __init__(self, max_k=None):
            self.heap = []
            self.max_size = max_k

        def __len__(self):
            return len(self.heap)

        def swap(self, a, b):
            self.heap[a], self.heap[b] = self.heap[b], self.heap[a]

        def get_main_class(self) -> int:
            """Return the class idx which we have the most in heap"""
            table = {}
            for node in self.heap:
                class_idx = node.class_idx
                if class_idx not in table:
                    table[class_idx] = 0
                table[class_idx] += 1

            main_class_value = -1
            main_class_idx = -1
            for key in table:
                if main_class_value < table[key]:
                    main_class_value = table[key]
                    main_class_idx = key

            return int(main_class_idx)

        def peek(self) -> Node:
            return self.heap[0]

        def insert(self, distance, class_idx):

            if len(self) == self.max_size:
                max_node = self.peek()

                if max_node.distance > distance:
                    new_node = KNN.MaxHeap.Node(distance, class_idx)
                    self.heap[0] = new_node
                    self.sift_down(0)
            else:
                new_node = KNN.MaxHeap.Node(distance, class_idx)
                self.heap.append(new_node)
                self.sift_up(len(self.heap) - 1)

        def sift_down(self, start_index):
            if start_index < 0:
                start_index = len(self.heap) + start_index

            child_one_index = 2 * start_index + 1
            child_two_index = 2 * start_index + 2
            while child_one_index < len(self.heap):
                if child_two_index < len(self.heap):
                    if self.heap[child_one_index].distance >= self.heap[child_two_index].distance and \
                            self.heap[start_index].distance < self.heap[child_one_index].distance:
                        new_index = child_one_index
                    elif self.heap[child_one_index].distance < self.heap[child_two_index].distance and \
                            self.heap[start_index].distance < self.heap[child_two_index].distance:
                        new_index = child_two_index
                    else:
                        break
                else:
                    if self.heap[start_index].distance < self.heap[child_one_index].distance:
                        new_index = child_one_index
                    else:
                        break
                self.swap(start_index, new_index)
                start_index = new_index
                child_one_index = 2 * start_index + 1
                child_two_index = 2 * start_index + 2

        def sift_up(self, idx):
            if idx < 0:
                idx = len(self.heap) + idx

            idx_parent = (idx - 1) // 2
            while idx_parent >= 0:
                if self.heap[idx].distance > self.heap[idx_parent].distance:
                    self.swap(idx, idx_parent)
                    idx = idx_parent
                    idx_parent = (idx - 1) // 2
                else:
                    break

    def __init__(self):
        self.labels = ([], None)  # features, target
        self.data = [[], None]  # [feature_data, target_data]
        self.k_nn = None

    def get_number_of_rows(self):
        return len(self.data[1]) if self.data else -1

    def set_data(self, data):
        self.data = data

    def set_labels(self, labels):
        self.labels = labels

    def predict(self, point_data: list, k=None) -> int:

        def _get_distance(main_point_data: list, second_point_data: list, point_class_idx: int):
            distance = 0
            for feature_idx in range(len(main_point_data)):
                distance += round(pow(main_point_data[feature_idx] - second_point_data[feature_idx], 2),
                                  self.ROUND_AFTER_COMA)
            return pow(distance, 0.5)

        if k is None:
            k = self.k_nn

        if len(point_data) != len(self.data[0][0]):
            print("Given point cannot be match to model data")
            return -1

        max_heap = self.MaxHeap(max_k=k)
        m_rows = self.get_number_of_rows()

        if m_rows == -1:
            print("Wrong sequence of model fit. Redo")
            return -1

        for point_idx in range(m_rows):
            features = self.data[0][point_idx]
            target = self.data[1][point_idx]
            p2p_distance = _get_distance(point_data, features, target)
            max_heap.insert(p2p_distance, target)

        return max_heap.get_main_class()

    def evaluation(self, data_for_evaluation: (list, list), metric="confusion_matrix"):

        def confusion_matrix(data: (list, list), knn_model: KNN) -> (list, float, float):
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            features, target = data[0], data[1]

            for idx in range(len(features)):
                prediction = knn_model.predict(features[idx])
                if prediction == 1 and target[idx] == 1:
                    true_positive += 1
                elif prediction == 1 and target[idx] == 0:
                    false_positive += 1
                elif prediction == 0 and target[idx] == 1:
                    false_negative += 1
                else:
                    true_negative += 1

            m = len(features)
            matrix = [[round(true_positive / m, knn_model.ROUND_AFTER_COMA),
                       round(false_positive / m, knn_model.ROUND_AFTER_COMA)],
                      [round(false_negative / m, knn_model.ROUND_AFTER_COMA),
                       round(true_negative / m, knn_model.ROUND_AFTER_COMA)]]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            return matrix, precision, recall

        if metric == "confusion_matrix":
            # conf_matrix, precision, recall = confusion_matrix(data_for_evaluation)
            return confusion_matrix(data_for_evaluation, self)

        return -1.0
