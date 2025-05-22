import pathlib
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

from feature_definitions import feature_build


def load_data(load_path):
    df = pd.read_csv(load_path)
    return df

def split_data(df,test_size,seed):  
    train,test = train_test_split(df, test_size=test_size, random_state=seed)
    return train,test

def save_data(train,test,output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index = False)
    test.to_csv(output_path + '/test.csv',index = False)

if __name__ == '__main__':
    currdir = pathlib.Path(__file__)
    homedir = currdir.parent.parent.parent
    path = sys.argv[1]

    trainpath = path + 'train.csv'
    testpath = path + 'test.csv'
    
    train_data = pd.read_csv(trainpath)
    test_data = pd.read_csv(testpath)
    
    output_path = homedir.as_posix()+'/data/processed'

    train_data = feature_build(train_data)
    test_data = feature_build(test_data)
    
    do_not_train = ['id','pickup_datetime','dropoff_datetime','check_trip_duration','pickup_date','avg_speed_h','avg_speed_m','pickup_lat_bin','pickup_long_bin','center_lat_bin','center_long_bin','pickup_dt_bin','pickup_datetime_group']
    feature_names = [f for f in train_data.columns if f not in do_not_train]
    print("We have %i features to train." %len(feature_names))
    
    train_data = train_data[feature_names]
    test_data = test_data[feature_names] 
    
    save_data(train_data, test_data, output_path)

    