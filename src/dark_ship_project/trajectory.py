from dark_ship_project.read_AIS_data import read_AIS_data
import pandas as pd
import numpy as np
from pyproj import geod
geod = geod.Geod(ellps="WGS84")
'''
Reads and processes AIS data.
Parameters:
    time (string): the time the AIS data is needed for
    folder (string): name of folder where AIS data is stored
Returns:
    pandas dataframe with one row per MMSI at predicted location at given time. 

'''


def update_AIS_data(time, folder):
    try:
        AIS_df = read_AIS_data(folder)
        AIS_df['timestamp'] = pd.to_datetime(AIS_df['timestamp'])

        time_filtered_df = AIS_df.set_index(
            'timestamp')  # index for speed of search

        result_rows = []
        for mmsi, group in time_filtered_df.groupby('mmsi'):
            result_rows.append(find_position_mmsi_group(group, time))

        result = pd.DataFrame(result_rows)
        return result
    except ValueError:
        print("Value error encountered, likely invalid file")
        return pd.DataFrame(columns=["mmsi", "ship_type", "navigational_status","timestamp", "latitude", "longitude", "speed", "course"])


'''
Predicts the position of a ship at a given time

Parameters:
    mmsi_group_dataframe (pandas dataframe): the dataframe of AIS pings specific to an mmsi.
    time (pandas datetime): the time the prediction is for.
Returns:
    pandas dataframe row - a single row with the predicted location in the lat/long fields at provided time.

'''

def find_position_mmsi_group(mmsi_group_dataframe, time):
    mmsi_group_dataframe.sort_index(inplace=True)
    index = mmsi_group_dataframe.index.searchsorted(time)

    if index == 0:
        row = mmsi_group_dataframe.iloc[0].copy()
    elif index == len(mmsi_group_dataframe):
        row = mmsi_group_dataframe.iloc[-1].copy()
    else:
        # closest transmission before time
        first = mmsi_group_dataframe.iloc[index-1]
        # closest transmision after time
        second = mmsi_group_dataframe.iloc[index]
        first_time = first.name  # .name as indexed by timestamp
        second_time = second.name

        first_speed = float(first.get("speed", 0)) * \
            0.514444  # convert knots to m/s
        second_speed = float(second.get("speed", first_speed)) * 0.514444
        speed_difference=second_speed-first_speed

        first_course = float(first.get("course", 0))
        second_course = float(second.get("course", first_course))
        course_difference = (second_course - first_course + 180) % 360 - 180 #finds shortest angle between courses
        

        total_time = (second_time-first_time).total_seconds()
        
        time_from_first_ping=(time-first_time).total_seconds()
        

        if total_time<30:
            speed_difference=0


        time_step_length=5
        lon = first["longitude"]
        lat = first["latitude"]
        time_steps=int(time_from_first_ping//time_step_length)
        remainder = total_time % time_step_length

        #lon_lat=[[lon,lat]] add commented lines to return list of points

        for time_step in range(time_steps):
            current_time=(time_step+1)*time_step_length
            current_speed = first_speed + (current_time/total_time) * speed_difference #accelerate by the average difference in speed
            current_course= (first_course + (current_time/total_time) * course_difference)%360 #change course by the average difference in course
            distance=current_speed*time_step_length
            lon,lat,_=geod.fwd(lon,lat,current_course,distance)
            #lon_lat.append([lon,lat])

        if remainder > 0:
            current_time = time_steps * time_step_length + remainder

            current_speed = first_speed + (current_time / total_time) * speed_difference
            current_course = (first_course + (current_time / total_time) * course_difference) % 360

            distance = current_speed * remainder
            lon, lat, _ = geod.fwd(lon, lat, current_course, distance)

            #lon_lat.append([lon, lat])

        row = first.copy()
        row["latitude"] = lat
        row["longitude"] = lon
    return row #, lon_lat





