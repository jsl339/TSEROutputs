import glob
import math
import pandas as pd
from functools import reduce
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def ensemble_traverser(path_name):
    files = glob.glob((path_name))
    df1, df2, df3, df4 ,df5 = [pd.read_csv(f) for f in files]
    #df1,df2 = [pd.read_csv(f) for f in files]
    data_frames = [df1, df2, df3, df4 ,df5]
    #data_frames = [df1,df2]
    merged = reduce(lambda  left,right: pd.merge(left,right,on=['ytest'],
                                            how='outer'), data_frames)
    hatlist = ["yhat.1","yhat.2","yhat.3","yhat.4","yhat.5"]
    #hatlist = ["yhat.1","yhat.2"]
    merged["mean"] = merged[hatlist].mean(axis=1)
    merged["sd"] = merged[hatlist].std(axis=1)

    output = pd.DataFrame(data=np.zeros((1, 2)), index=[0],
                       columns=['rmse', 'mae'])
    output['rmse'] = math.sqrt(mean_squared_error(merged["ytest"], merged["mean"]))
    output['mae'] = mean_absolute_error(merged["ytest"], merged["mean"])
    return merged,output


def collection(path_name):
    files = glob.glob((path_name))
    df1,df2,df3,df4,df5 = [pd.read_csv(f) for f in files]
    data_frames = [df1,df2,df3,df4,df5]
    merged = pd.concat(data_frames, ignore_index = True)
    outs = pd.DataFrame(data=np.zeros((1, 2)), index=[0],
                       columns=['total rmse', 'total mae'])
    outs["total rmse"] = merged["rmse"].mean()
    outs["total mae"] = merged["mae"].mean()
    print(outs)
    return outs
