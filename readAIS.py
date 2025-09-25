
import pandas 
import os


def read_AIS_data(folder_path):
    columns_to_read=[0,11,12,13,14]
    file_list=[]

    for file_name in os.listdir(folder_path):
        file_path=os.path.join(folder_path,file_name)
        file=pandas.read_csv(folder_path+"/"+file_name,header=None,usecols=columns_to_read)

        file_list.append(file)


    combined_file=pandas.concat(file_list,ignore_index=True)
    combined_file.columns=["mmsi","navigational_status","timestamp","latitude","longitude"]

    mmsi_to_check = 311404000  # replace with an actual MMSI from your data
    print(combined_file[combined_file['mmsi'] == mmsi_to_check])


    combined_file.groupby("mmsi") #group by mmsi, data already in time order so

    return combined_file


