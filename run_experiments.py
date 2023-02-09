# Chang Wei Tan, Christoph Bergmeir, Francois Petitjean, Geoff Webb
#
# @article{
#   Tan2020TSER,
#   title={Time Series Extrinsic Regression},
#   author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
#   journal={Data Mining and Knowledge Discovery},
#   pages={1--29},
#   year={2021},
#   publisher={Springer},
#   doi={https://doi.org/10.1007/s10618-021-00745-9}
# }

import numpy as np
import pandas as pd
from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from utils.tools2 import create_directory
from utils.traverser import ensemble_traverser, collection

module = "RegressionExperiment"
data_path = "/Users/johnleland/Desktop/TS-Extrinsic-Regression/Monash_UEA_UCR_Regression_Archive/"
problems = ["FloodModeling2"]       # see data_loader.regression_datasets
regressors = ["fcnattention"]    # see regressor_tools.all_models
iterations = [1,2,3,4,5]
norm = "none"               # none, standard, minmax
output_num = [1,2,3,4,5]
output_path = "output/regression/"
if __name__ == '__main__':
    # for each problem
    for problem in problems:
        print("#########################################################################")
        print("[{}] Starting Experiments".format(module))
        print("#########################################################################")
        print("[{}] Data path: {}".format(module, data_path))
        print("[{}] Problem: {}".format(module, problem))

        # set data folder, train & test
        data_folder = data_path + problem + "/"
        train_file = data_folder + problem + "_TRAIN.ts"
        test_file = data_folder + problem + "_TEST.ts"

        # loading the data. X_train and X_test are dataframe of N x n_dim
        print("[{}] Loading data".format(module))
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        X_test, y_test = load_from_tsfile_to_dataframe(test_file)

        print("[{}] X_train: {}".format(module, X_train.shape))
        print("[{}] X_test: {}".format(module, X_test.shape))

        # in case there are different lengths in the dataset, we need to consider that.
        # assume that all the dimensions are the same length
        print("[{}] Finding minimum length".format(module))
        min_len = np.inf
        for i in range(len(X_train)):
            x = X_train.iloc[i, :]
            all_len = [len(y) for y in x]
            min_len = min(min(all_len), min_len)
        for i in range(len(X_test)):
            x = X_test.iloc[i, :]
            all_len = [len(y) for y in x]
            min_len = min(min(all_len), min_len)
        print("[{}] Minimum length: {}".format(module, min_len))

        # process the data into numpy array
        print("[{}] Reshaping data".format(module))
        x_train = process_data(X_train, normalise=norm, min_len=min_len)
        x_test = process_data(X_test, normalise=norm, min_len=min_len)

        print("[{}] X_train: {}".format(module, x_train.shape))
        print("[{}] X_test: {}".format(module, x_test.shape))

        for regressor_name in regressors:
            print("[{}] Regressor: {}".format(module, regressor_name))
            for num in output_num:
                for itr in iterations:
                    # create output directory
                    output_directory = "output_"+str(num)+"/regression/"
                    if norm != "none":
                        output_directory = "output_"+str(num)+"/regression_{}/".format(norm)
                    output_directory = output_directory + regressor_name + '/' + problem + '/itr_' + str(itr) + '/'
                    create_directory(output_directory)

                    print("[{}] Iteration: {}".format(module, itr))
                    print("[{}] Output Dir: {}".format(module, output_directory))

                    # fit the regressor
                    regressor = fit_regressor(output_directory, regressor_name,
                                                        x_train, y_train, x_test, y_test,
                                                        itr=itr)

                    # start testing
                    y_pred = regressor.predict(x_test)
                    df_metrics = calculate_regression_metrics(y_test, y_pred)
                    dataframe = np.concatenate([y_test.reshape(-1,1), y_pred], axis=1)
                    y_prediction = pd.DataFrame(dataframe, columns = ["ytest", "yhat." + str(itr)])
                    print(df_metrics)

                    # save the outputs
                    df_metrics.to_csv(output_directory + 'regression_experiment.csv', index=False)
                    #y_testing.to_csv(output_directory + 'y_testing.csv', index=False) 
                    y_prediction.to_csv(output_directory + 'y_predictions.csv', index=False)               
                # my step
                out_path = "/Users/johnleland/Desktop/TS-Extrinsic-Regression/output_"+ str(num) +"/regression/fcnattention/"
                folder = out_path + problem + "/*/y_predictions.csv"
                new_directory = "output_"+str(num)+"/regression/" + regressor_name + '/' + problem + '/'
                create_directory(new_directory)
                merge,output = ensemble_traverser(folder)
                #outputs = pd.DataFrame(data=np.zeros((1, 1)), index=[0],
                #   columns=['rmse'])
                #outputs['rmse'] = output
                print("Ensemble RMSE:", output)
                merge.to_csv(new_directory+ 'ensemble.csv', index=False)
                output.to_csv(new_directory + 'ensemble_metrics.csv', index=False)
            out = "/Users/johnleland/Desktop/TS-Extrinsic-Regression/*/regression/"+ regressor_name + "/" + problem + "/ensemble_metrics.csv"
            newish_directory = "output/regression/" + regressor_name + '/' + problem + '/'
            create_directory(newish_directory)
            mean_rmse = collection(out)
            mean_rmse.to_csv(newish_directory + "mean_rmse_mae.csv", index = False)





                
