from readAIS import read_AIS_data
import pandas as pd

def read_local_data(time,folder):
    AIS_df=read_AIS_data(folder)
    AIS_df['timestamp'] = pd.to_datetime(AIS_df['timestamp'])
    min_time=time-pd.Timedelta(minutes=15)
    time_filtered_df = AIS_df[(AIS_df['timestamp'] >= min_time)&(AIS_df['timestamp'] <= time)]

    for mmsi,ship_info in time_filtered_df.groupby('mmsi'):
        traj_info = ship_info[["length","width","latitude","longitude","speed","course"]].values
    
    




   

    
    



    

read_local_data(pd.Timestamp("2023-06-03 12:00:00"),"20230603")
