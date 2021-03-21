"""Regression

Follow jupyter notebook workflow for better 
understanding of how to apply data_reader and
Regression model to train your AI.

Link: https://github.com/kotsky/ai-dev/blob/main/regression_workflow.ipynb
Package: https://github.com/kotsky/ai-dev/blob/main/regression.py

"""

"""Logistic Regression

Follow jupyter notebook workflow for better 
understanding of how to apply data_reader and
Logistic Regression model to train your AI.

Link: https://github.com/kotsky/ai-dev/blob/main/logistic_regression_workflow.ipynb
Package: https://github.com/kotsky/ai-dev/blob/main/classification/logistic_regression.py

"""

"""K-Nearest Neighbors

Follow jupyter notebook workflow for better 
understanding of how to apply data_reader and
K-Nearest Neighbors model to train your AI.

Link: https://github.com/kotsky/ai-dev/blob/main/knn_workflow.ipynb
Package: https://github.com/kotsky/ai-dev/blob/main/classification/knn.py

"""


if __name__ == '__main__':

    """K-Mean test"""

    import data_reader as dr
    import clusterization.kmean as km
    import random
    import matplotlib.pyplot as plt
    import helper_methods as hp

    # random.seed(443)

    main_table = dr.DataTable("data/Cust_Segmentation.csv")
    main_table.activate_features(["Edu", "Income"])
    main_table.select_target("Customer Id")
    main_table.split_data(0.7, shuffle=True)
    min_max_info = main_table.get_min_max_features()

    training_data = main_table.get_training_data()
    test_data = main_table.get_testing_data()

    training_data = training_data[0]
    test_data = test_data[0]

    model_kmean = km.KMean()
    model_kmean.epoch = 5
    model_kmean.number_of_centroids = 5
    model_kmean.set_training_data(training_data)
    model_kmean.set_min_max_ranges(min_max_info)

    model_kmean.fit()
    centroids_coord = model_kmean.centroids
    print(centroids_coord)
    cost_function = model_kmean.cost_functions

    axis_x = [x for x in range(model_kmean.epoch)]
    plt.plot(axis_x, cost_function, 'b')
    plt.show()
    #
    # plt.plot(centroids_coord[0][0], centroids_coord[0][1], 'r')
    # plt.plot(centroids_coord[1][0], centroids_coord[1][1], 'g')
    #
    #
