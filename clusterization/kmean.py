"""K-Mean

Clustering with K-Mean algorithm.

By default, there are 20 tries of random centroid initialization
with cost function logging to define the best centroids coordinates.

Number of iterations(epoch), number of centroids(k), number of centroids
initialization and more are possible to define by the user.

Model returns best trained coordinates of centroids after 20 tries
(with self.centroids) and cost function logs for each iteration(epoch)
as self.cost_functions

"""


import matplotlib.pyplot as plt


class KMean:

    import random

    ROUND_AFTER_COMA = 4
    NUMBER_OF_CENTROIDS_INITIALIZATION = 20

    def __init__(self):

        self.number_of_centroids = 2
        self.centroids = []
        self.training_data = []
        self.epoch = 1

        self.cost_functions = []
        self._point_centroid = []
        self._temp_centroids = []
        self._min_max_training_ranges = []

    def fit(self):
        """Train model"""
        for i in range(self.NUMBER_OF_CENTROIDS_INITIALIZATION):
            self._initialization()
            self._k_mean_core()

    def predict(self, new_point_coord: list) -> int:
        """Return the most closest centroid to the given point"""
        min_distance = float("inf")
        closest_centroid = None
        for centroid_idx in range(len(self._temp_centroids)):
            distance = self._get_distance(centroid_idx, point_coord=new_point_coord)
            if distance < min_distance:
                min_distance = distance
                closest_centroid = centroid_idx
        return closest_centroid

    def set_training_data(self, training_data: list) -> None:
        """Set training data set internally"""
        self.training_data = training_data

    def set_min_max_ranges(self, min_max_data: list) -> None:
        """Set range of random centroids' coords initialization"""
        self._min_max_training_ranges = min_max_data

    def _k_mean_core(self):
        """Core body of K-Mean algorithm"""

        cost_function_local = []
        for e in range(self.epoch):
            self.draw_centroids()
            self.draw_feature()
            self._assign_centroids_to_point()
            self._recalc_centroids_coord()
            self.draw_centroids(label='bo')
            plt.show()
            cost_function_local.append(max(self.get_cost_functions_local()))

        # take the best trained centroids by min cost function
        if not self.cost_functions or (self.cost_functions[-1] > cost_function_local[-1]):
            self.cost_functions = cost_function_local
            self.centroids = self._temp_centroids.copy()

    def _get_distance(self, centroid_idx: int, point_coord=None, point_idx=None) -> float:
        """Return distance from point to centroid"""
        centroid_coord = self._temp_centroids[centroid_idx]
        if point_idx is not None and point_coord is None:
            point_coord = self.training_data[point_idx]

        distance = 0
        for idx in range(len(point_coord)):
            distance += pow(point_coord[idx] - centroid_coord[idx], 2)
        return round(pow(distance, 0.5), KMean.ROUND_AFTER_COMA)

    def _assign_centroids_to_point(self):
        """Define which centroid is closed to each training point and assign it"""
        for point_idx in range(len(self.training_data)):

            min_distance = float("inf")
            closest_centroid = None

            for centroid_idx in range(len(self._temp_centroids)):
                distance = self._get_distance(centroid_idx, point_idx=point_idx)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = centroid_idx
            self._point_centroid[point_idx] = closest_centroid

    def _recalc_centroids_coord(self):
        """Recalculation of centroids' coordinates based on new assigned points"""
        n_features = self.get_number_of_features()
        n_of_point_for_each_centroid = [0] * len(self._temp_centroids)

        for point_idx in range(len(self._point_centroid)):
            assigned_centroid = self._point_centroid[point_idx]
            point_coord = self.training_data[point_idx]
            for feature_idx in range(len(point_coord)):
                axis_val = point_coord[feature_idx]
                self._temp_centroids[assigned_centroid][feature_idx] += axis_val
            n_of_point_for_each_centroid[assigned_centroid] += 1

        for centroid_idx in range(len(self._temp_centroids)):
            if n_of_point_for_each_centroid[centroid_idx] == 0:
                print("Centroid {} doesn't have any point".format(centroid_idx))
                continue
            for feature_idx in range(n_features):
                self._temp_centroids[centroid_idx][feature_idx] /= \
                    n_of_point_for_each_centroid[centroid_idx]
                # storage[centroid_idx][feature_idx] = \
                # round(storage[centroid_idx][feature_idx] / m, self.ROUND_AFTER_COMA)

    def _initialization(self):
        """Pre-processing"""
        self._centroids_initialization()
        self._additional_initialization()

    def _centroids_initialization(self):
        """Create random centroids in range min-max of each feature"""
        self._temp_centroids = [[0 for x in range(self.get_number_of_features())]
                                for y in range(self.number_of_centroids)]
        for centroid_idx in range(self.number_of_centroids):
            for feature_idx in range(self.get_number_of_features()):
                a = self._min_max_training_ranges[feature_idx][0]  # min
                b = self._min_max_training_ranges[feature_idx][1]  # max
                self._temp_centroids[centroid_idx][feature_idx] = \
                    self.random.randrange(a, b)

    def get_number_of_features(self) -> int:
        """Return number of training features"""
        return len(self.training_data[0])

    def get_number_of_training_points(self) -> int:
        """Return number of training rows/points"""
        return len(self.training_data)

    def _additional_initialization(self):
        """Add 1 column to store assigned centroids to each training point"""
        self._point_centroid = [-1] * self.get_number_of_training_points()

    def get_cost_functions_local(self) -> list:
        """Return array of cost functions, where each index represent centroid index"""
        n_points = self.get_number_of_training_points()
        n_of_point_for_each_centroid = [0] * len(self._temp_centroids)
        cost_function_of_each_centroid_local = [0] * len(self._temp_centroids)

        for point_idx in range(n_points):
            assigned_centroid = self._point_centroid[point_idx]
            distance = self._get_distance(assigned_centroid, point_idx=point_idx)
            n_of_point_for_each_centroid[assigned_centroid] += 1
            cost_function_of_each_centroid_local[assigned_centroid] += distance

        for centroid_idx in range(len(self._temp_centroids)):
            if n_of_point_for_each_centroid[centroid_idx] == 0:
                print("There is no point assigned to this {} centroid".format(centroid_idx))
                cost_function_of_each_centroid_local[centroid_idx] = -1
                continue
            cost_function_of_each_centroid_local[centroid_idx] /= n_of_point_for_each_centroid[centroid_idx]

        return cost_function_of_each_centroid_local

    def draw_centroids(self, label='ro'):
        for centroid in self._temp_centroids:
            x, y = centroid[0], centroid[1]
            plt.plot(x, y, label)

    def draw_feature(self):
        X = []
        Y = []
        for r in range(len(self.training_data)):
            x, y = self.training_data[r]
            X.append(x)
            Y.append(y)
        plt.scatter(X, Y)
