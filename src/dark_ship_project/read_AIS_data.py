
import pandas
import os

'''
Reads all files in a folder, converting the AIS data into a pandas dataframe with 8 fields.

Parameters:
    folder_path(string):The path to the folder with AIS data

Returns:
    pandas dataframe: a dataframe contianing mmsi, ship_type, navigation status, timestamp, latitude, longitude speed and course.
'''


def read_AIS_data(folder_path):
    columns_to_read = [0, 5, 11, 12, 13, 14, 15, 16]
    file_list = []

    if not os.path.isdir(folder_path):
        raise ValueError("Invalid directory: "+folder_path)

    for file_name in os.listdir(folder_path):
        file = pandas.read_csv(folder_path+"/"+file_name,
                               header=None, usecols=columns_to_read)
        file_list.append(file)
    
    if not file_list:
        return pandas.DataFrame(columns=["mmsi", "ship_type", "navigational_status","timestamp", "latitude", "longitude", "speed", "course"])

    combined_file = pandas.concat(file_list, ignore_index=True)
    combined_file.columns = ["mmsi", "ship_type", "navigational_status",
                             "timestamp", "latitude", "longitude", "speed", "course"]
    combined_file.groupby("mmsi")  # group by mmsi, data already in time order

    return combined_file 
