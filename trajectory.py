from read_AIS_data import read_AIS_data
import pandas as pd
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
    AIS_df = read_AIS_data(folder)
    AIS_df['timestamp'] = pd.to_datetime(AIS_df['timestamp'])

    time_filtered_df = AIS_df.set_index(
        'timestamp')  # index for speed of search

    result_rows = []
    for mmsi, group in time_filtered_df.groupby('mmsi'):
        result_rows.append(find_postion_mmsi_group(group, time))

    result = pd.DataFrame(result_rows)
    return result


'''
Predicts the position of a ship at a given time

Parameters:
    mmsi_group_dataframe (pandas dataframe): the dataframe of AIS pings specific to an mmsi.
    time (pandas datetime): the time the prediction is for.
Returns:
    pandas dataframe row - a single row with the predicted location in the lat/long fields at provided time.

'''


def find_postion_mmsi_group(mmsi_group_dataframe, time):
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

        left_course = float(first.get("course", 0))
        second_course = float(second.get("course", left_course))

        total_time = (second_time-first_time).total_seconds()

        # adjust for times since each AIS ping.
        first_proportion = (time-first_time).total_seconds()/total_time
        second_proportion = 1-first_proportion

        reverse_course = (second_course+180) % 360

        first_lon, first_lat, _ = geod.fwd(
            first["longitude"], first["latitude"], left_course, first_speed*total_time*first_proportion)  # project on globe
        second_lon, second_lat, _ = geod.fwd(second["longitude"], second["latitude"], (
            reverse_course), (second_speed*total_time*second_proportion))

        angle, _, dist = geod.inv(first_lon, first_lat, second_lon, second_lat)
        calc_lon, calc_lat, _ = geod.fwd(first_lon, first_lat, angle, dist)

        row = first.copy()
        row["latitude"] = calc_lat
        row["longitude"] = calc_lon

    return row
