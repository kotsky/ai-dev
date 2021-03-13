import data_reader as dr
import regression
import matplotlib.pyplot as plt

if __name__ == '__main__':

    main_data_table = dr.DataTable("test_data.csv")
    head = main_data_table.head
    main_data_table.activate_features(head[0])
    # for feature in head:
    #     main_data_table.activate_features(feature)
    # main_data_table.deactivate_feature(head[-1])
    main_data_table.select_target(head[-1])
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
    regression_model.alpha = 0.001
    regression_model.create_coefficients_array()
    coeff = regression_model.fit()
    error = regression_model.evaluation(cv_data)
    print("Coefficients {} give error {}".format(coeff, error))

    test_features, test_target = test_data
    predicted = []
    for test in test_features:
        predicted.append(regression_model.predict(test))

    plt.scatter()
    plt.plot(test_features, test_target, test_features, predicted)
    plt.show()

    # Coefficients [21.56, 77.04] give error 352.15

