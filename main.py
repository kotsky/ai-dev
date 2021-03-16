import data_reader as dr
import regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper_methods import *


if __name__ == '__main__':

    main_data_table = dr.DataTable("data/FuelConsumption.csv")

    head = main_data_table.head
    main_data_table.activate_features("FUELCONSUMPTION_COMB_MPG")
    # main_data_table.activate_features("ENGINESIZE")
    # for feature in label_data.head:
    #     main_data_table.activate_features(feature)
    # main_data_table.deactivate_feature(head[-1])
    main_data_table.select_target("CO2EMISSIONS")

    # main_data_table.add_new_feature("FUELCONSUMPTION_COMB_MPG", power=2)
    main_data_table.max_scaling()
    main_data_table.split_data(0.6, 0.2, shuffle=True)
    training_data = main_data_table.get_training_data()
    cv_data = main_data_table.get_cv_data()
    test_data = main_data_table.get_testing_data()
    data_labels = main_data_table.get_labels()
    # main_data_table.plot(features2target=True)

    regression_model = regression.Regression()
    regression_model.set_labels(data_labels)
    regression_model.set_training_data(training_data[0], training_data[1])
    regression_model.set_testing_data(test_data[0], test_data[1])
    # regression_model.log_mode()
    regression_model.RANDOM_WEIGHT_INITIALIZATION = 10
    regression_model.ROUND_AFTER_COMA = 6
    regression_model.epoch = 500
    regression_model.alpha = 0.01
    regression_model.regularization = 0
    regression_model.create_coefficients_array(True)
    print(regression_model.coefficients)
    coeff = regression_model.fit()
    error = regression_model.evaluation(cv_data)
    print("Coefficients {} give error {}".format(coeff, error))

    test_features, test_target = test_data
    training_features, training_target = training_data
    predicted = []
    for test in test_features:
        predicted.append(regression_model.predict(test))

    # tr_f = []
    # for line in training_features:
    #     tr_f.append(line[0])

    # fig = plt.figure()
    # ax = Axes3D(fig)

    axis1 = column2list(test_features, 0)
    axis2 = column2list(test_features, 1)

    # ax.scatter(axis1, axis2, test_target)
    # plt.scatter(test_features, test_target)
    # ax.scatter(axis1, axis2, predicted)
    # plt.show()

    axis = main_data_table.get_column_data("ENGINESIZE")

    # x = generate_axis(1, 12, 0.5)
    #
    # y = []
    # for _x in x:
    #     y.append(regression_model.coefficients[1]*_x + regression_model.coefficients[0])

    plt.scatter(axis1, test_target, label="Target")
    plt.scatter(axis1, predicted, label="Predicted")
    plt.title("YO")
    plt.legend(loc="upper right")
    plt.xlabel("FUELCONSUMPTION_COMB_MPG")
    plt.ylabel("CO2EMISSIONS")

    # plt.plot(x, y)
    plt.show()

    """
    ...
    Iteration 1000 done
    Coefficients [-1.55637, 4.81319, 11.87481, 4.5559, 5.51779, 2.46835, 1.13174] give error 17.65038
    
    Iteration 1000 done
    1.15217 -1.91057 * FUELCONSUMPTION_COMB_MPG + 1.02521 * FUELCONSUMPTION_COMB_MPG^2
    Coefficients [1.15217, -1.91057, 1.02521] give error 0.0302
    
    """

