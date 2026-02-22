from readAIS import read_AIS_data
import pandas as pd
from pyproj import geod
geod=geod.Geod(ellps="WGS84")

def update_AIS_data(time,folder):
    AIS_df=read_AIS_data(folder)
    AIS_df['timestamp']=pd.to_datetime(AIS_df['timestamp'])

    time_filtered_df=AIS_df.set_index('timestamp')#index for speed of search

    result_rows=[]
    for mmsi, group in time_filtered_df.groupby('mmsi'):
        result_rows.append(find_postion_mmsi_group(group,time))

    result=pd.DataFrame(result_rows)
    return result



def find_postion_mmsi_group(mmsi_group_dataframe,time):
    mmsi_group_dataframe.sort_index(inplace=True)
    index=mmsi_group_dataframe.index.searchsorted(time)

    if index == 0:
        row = mmsi_group_dataframe.iloc[0].copy()
    elif index == len(mmsi_group_dataframe):
        row = mmsi_group_dataframe.iloc[-1].copy()
    else:
        left=mmsi_group_dataframe.iloc[index-1] #closest transmission before time
        right=mmsi_group_dataframe.iloc[index] #closest transmision after time
        left_time=left.name #.name as indexed by timestamp
        right_time=right.name

        left_speed  = float(left.get("speed", 0)) * 0.514444
        right_speed = float(right.get("speed", left_speed)) * 0.514444

        left_course  = float(left.get("course", 0))
        right_course = float(right.get("course", left_course))

        total_time=(right_time-left_time).total_seconds()

        left_proportion=(time-left_time).total_seconds()/total_time
        right_proportion=1-left_proportion

        reverse_course=(right_course+180)%360

        left_lon,left_lat,_=geod.fwd(left["longitude"],left["latitude"],left_course,left_speed*total_time*left_proportion)
        right_lon,right_lat,_=geod.fwd(right["longitude"],right["latitude"],(reverse_course),(right_speed*total_time*right_proportion))

        angle,_,dist=geod.inv(left_lon,left_lat,right_lon,right_lat)
        calc_lon,calc_lat,_=geod.fwd(left_lon,left_lat,angle,dist)



        row=left.copy()
        row["latitude"]=calc_lat
        row["longitude"]=calc_lon
        if mmsi_group_dataframe[mmsi_group_dataframe["mmsi"]==235100575].shape[0]>0:
            print("found",calc_lat,calc_lon)

    return row
        