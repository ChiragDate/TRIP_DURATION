import pathlib
import pandas as pd
import numpy as np

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def datetime_feature_fix(df):
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df.loc[:, 'pickup_date'] = df['pickup_datetime'].dt.date
    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
    
def create_datetime_features(df):
    df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
    df.loc[:, 'pickup_minute'] = df['pickup_datetime'].dt.minute
    df.loc[:,'pickup_dt'] = (df['pickup_datetime']- df['pickup_datetime'].min()).dt.total_seconds()
    df.loc[:,'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']
    
    
def create_dist_features(df):
    df.loc[:, 'distance_haversine'] = haversine_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'direction'] = bearing_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    
def test_feature_build(df):  
    
    # just for local checking if it works feature_build to be used in other codes
    
    datetime_feature_fix(df)
    create_dist_features(df)
    create_datetime_features(df)
    print(df.columns)
    
def feature_build(df):
    datetime_feature_fix(df)
    create_dist_features(df)
    create_datetime_features(df)
    print(df.head())
    return df
    
    
if __name__ == '__main__':
    currdir = pathlib.Path(__file__)
    homedir = currdir.parent.parent.parent
    datapath = homedir.as_posix()+'/data/raw/test.csv'
    
    data = pd.read_csv(datapath,nrows = 10)
    test_feature_build(data)