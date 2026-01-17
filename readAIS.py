
import pandas 
import os


def read_AIS_data(folder_path):
    columns_to_read=[0,5,11,12,13,14]
    file_list=[]

    for file_name in os.listdir(folder_path):
        file=pandas.read_csv(folder_path+"/"+file_name,header=None,usecols=columns_to_read)
        file_list.append(file)

    combined_file=pandas.concat(file_list,ignore_index=True)
    combined_file.columns=["mmsi","ship_type","navigational_status","timestamp","latitude","longitude"]
    combined_file.groupby("mmsi") #group by mmsi, data already in time order

    return combined_file


