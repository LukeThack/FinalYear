import os
import pandas
import tempfile
import pytest
from datetime import datetime
from dark_ship_project.trajectory import update_AIS_data,find_position_mmsi_group
from dark_ship_project.read_AIS_data import read_AIS_data
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod
geod = Geod(ellps="WGS84")

def test_trajectory_invalid_dir():
    time= datetime(2026, 2, 25, 0, 0, 0)
    result=update_AIS_data(time,"nonexistentfolder")
    assert list(result.columns)==["mmsi", "ship_type", "navigational_status","timestamp", "latitude", "longitude", "speed", "course"]


def test_trajectory_returns_correctly():
    temp_dir=tempfile.TemporaryDirectory()
    data1=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:00:00", 10.0, 20.0, 30, 40]]#index 16 used therefore 17 entries required.
    data2=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:30:00", 20.0, 30.0, 40, 50]]
    file1=os.path.join(temp_dir.name,"data1.csv")
    file2=os.path.join(temp_dir.name,"data2.csv")
    df=pandas.DataFrame(data1)
    df.to_csv(file1,header=False,index=False)
    df=pandas.DataFrame(data2)
    df.to_csv(file2,header=False,index=False)
    time=datetime(2026, 2, 25, 0, 15, 0)
    result=update_AIS_data(time,temp_dir.name)
    assert list(result.columns)==["mmsi", "ship_type", "navigational_status", "latitude", "longitude", "speed", "course"]
    assert len(result)==1
    temp_dir.cleanup()

    
def test_trajectory_math():
    AIS_df = read_AIS_data("202306")  # path to AIS data folder

    open_water_ships = AIS_df[
        (AIS_df['latitude'] >= 53.5) & (AIS_df['latitude'] <= 54.1) &
        (AIS_df['longitude'] >= -5.2) & (AIS_df['longitude'] <= -3.2)
    ]

    
    total_attempts = 0
    successful_attempts = 0
    
    for mmsi, group in open_water_ships.groupby('mmsi'):
        group = group.sort_values("timestamp")
        if len(group) >= 3:
            # pick a random row in the middle of the group
            random_index = random.randint(1, len(group) - 2)
            row_prev = group.iloc[random_index - 1]
            row_mid = group.iloc[random_index]
            row_next = group.iloc[random_index + 1]
            past_time=pandas.to_datetime(row_prev["timestamp"])
            curr_time = pandas.to_datetime(row_mid["timestamp"])
            next_time = pandas.to_datetime(row_next["timestamp"])
            
            if not(curr_time - past_time <= pandas.Timedelta(minutes=15)) or not(next_time - curr_time <= pandas.Timedelta(minutes=15)): 
                continue
            if row_mid["ship_type"]!="MSC" and row_mid["ship_type"]!="TNK" :
                continue
            total_attempts += 1
                
            combined_df = pandas.DataFrame([row_prev, row_next])
            combined_df["timestamp"] = pandas.to_datetime(combined_df["timestamp"])
            combined_df = combined_df.set_index("timestamp")
            curr_time = pandas.to_datetime(row_mid["timestamp"])
            result = find_position_mmsi_group(combined_df, curr_time)
        
            try:
                _, _, dist = geod.inv(
                    float(row_mid["longitude"]),
                    float(row_mid["latitude"]),
                    float(result["longitude"]),
                    float(result["latitude"])
                )

            except (ValueError, KeyError, TypeError) as e:
                continue
            
            if dist<300:
                successful_attempts+=1
                
            '''
            if dist>200:
                plot_attempt(row_prev, row_next, row_mid, result, dist, lon_lat)
                print(row_prev["course"],row_next["course"],row_prev["speed"],row_next["speed"],(next_time-past_time).total_seconds())
            '''

    assert((successful_attempts/total_attempts)>0.9)

def plot_attempt(row_prev, row_next, row_mid, result, dist, trajectory):
    plt.figure(figsize=(5,5))
    plt.scatter(row_prev["longitude"], row_prev["latitude"], label="prev", marker="x")
    plt.scatter(row_next["longitude"], row_next["latitude"], label="next", marker="x")
    plt.scatter(row_mid["longitude"], row_mid["latitude"], label="actual mid", marker="o")
    plt.scatter(result["longitude"], result["latitude"], label="predicted", marker="^")

    if len(trajectory) > 1:
        lons, lats = zip(*trajectory)
        plt.plot(lons, lats, marker=".", linewidth=2, label="trajectory")

    plt.legend()
    plt.title(f"The error is {dist:.1f} m")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    