from utils.tools import create_directory
from utils.traverser import ensemble_traverser
out_path = "/Users/johnleland/Desktop/TS-Extrinsic-Regression/output/regression/fcnattention/"
problem  = "AppliancesEnergy"
folder = out_path + problem + "/*/y_predictions.csv"
new_directory = "output/regression/" + regressor_name + '/' + problem + '/'
create_directory(new_directory)
merge,output = ensemble_traverser(folder)
        #outputs = pd.DataFrame(data=np.zeros((1, 1)), index=[0],
         #   columns=['rmse'])
        #outputs['rmse'] = output
print("Ensemble RMSE:", output)
merge.to_csv(new_directory+ 'ensemble.csv', index=False)
output.to_csv(new_directory + 'ensemble_metrics.csv', index=False)