from readAIS import read_AIS_data
import pandas as pd

def read_local_data(time,folder):
    AIS_df=read_AIS_data(folder)
    AIS_df['timestamp']=pd.to_datetime(AIS_df['timestamp'])

    time_window=pd.Timedelta(minutes=15)
    time_filtered_df=AIS_df[(AIS_df['timestamp']>=time-time_window)&(AIS_df['timestamp']<=time+time_window)] #only check for ships with AIS pings within 15 minutes.
    time_filtered_df=time_filtered_df.set_index('timestamp')#index for speed
    result=time_filtered_df.groupby('mmsi').apply(find_postion_mmsi_group,time=time)#apply function to every mmsi group

    return result


def find_postion_mmsi_group(mmsi_group_dataframe,time):
    mmsi_group_dataframe.sort_index(inplace=True)
    index=mmsi_group_dataframe.index.searchsorted(time)

    if index==0:
        return pd.Series({"lat":mmsi_group_dataframe.iloc[0]["latitude"],"lon":mmsi_group_dataframe.iloc[0]["longitude"]})#only transmissions before time provided
    elif index==len(mmsi_group_dataframe):
        return pd.Series({"lat":mmsi_group_dataframe.iloc[index-1]["latitude"],"lon":mmsi_group_dataframe.iloc[index-1]["longitude"]})#only transmissions after time provided
    
    left=mmsi_group_dataframe.iloc[index-1] #closest transmission before time
    right=mmsi_group_dataframe.iloc[index] #closest transmision after time
    left_time=left.name #.name as indexed by timestamp
    right_time=right.name

    if left_time==right_time:
        return pd.Series({"lat":left["latitude"],"lon":left["longitude"]}) #hasnt moved, return either position
    
    ratio=(time-left_time).total_seconds()/(right_time-left_time).total_seconds() #proprotion of time between left and right AIS tranmission timestamps
    calc_lat=left["latitude"]+ratio*(right["latitude"]-left["latitude"])
    calc_lon=left["longitude"]+ratio*(right["longitude"]-left["longitude"]) 

    return pd.Series({"lat":calc_lat,"lon":calc_lon}) #return interpolated position for each mmsi
        

print(read_local_data(pd.Timestamp("2023-06-01 12:00:00"),"20230601"))
