from dark_ship_project.detect_dark_ships import find_dark_ships, get_ship_box_image
from dark_ship_project.process_sar_image import next_id,process_directory
from esa_snappy import ProductIO
from dark_ship_project.read_AIS_data import read_AIS_data
import numpy
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
yolo_model = YOLO("runs/obb/train/weights/best.pt")

ship_type_dict={
    "MER": "Merchant",
    "TNK": "Tanker",
    "FSH": "Fishing",
    "TUG": "Tug",
    "MIL": "Military",
}


def display_ships(path_to_SAR,AIS_directory_path,output_directory,directory_flag,YOLO1_conf_threshold,YOLO2_conf_threshold,classifier_conf_threshold):
    if directory_flag:

        process_directory(path_to_SAR)
        all_files=os.listdir(path_to_SAR)
        many_dark_ships={}
        many_multi_ships={}
        many_found_ships={}

        for file in all_files:
            if file.endswith(".dim"):
                file_name=os.path.join(path_to_SAR,file)
                dark_ships,found_confirmed_ships,multi_ships=find_dark_ships(AIS_directory_path, file,YOLO1_conf_threshold,YOLO2_conf_threshold,classifier_conf_threshold)
                many_dark_ships[file_name]=dark_ships
                many_found_ships[file_name]=found_confirmed_ships
                many_multi_ships[file_name]=multi_ships
    else:
        dark_ships,found_confirmed_ships,multi_ships=find_dark_ships(AIS_directory_path,path_to_SAR,YOLO1_conf_threshold,YOLO2_conf_threshold,classifier_conf_threshold)
        many_dark_ships[path_to_SAR]=dark_ships
        many_found_ships[path_to_SAR]=found_confirmed_ships
        many_multi_ships[path_to_SAR]=multi_ships

    for key in many_found_ships.keys():
        dark_ships=many_dark_ships.get(key)
        found_ships=many_found_ships.get(key)
        label_ships(key, dark_ships,output_directory, AIS_directory_path)
        label_ships(key, found_ships,output_directory, AIS_directory_path)



def write_ships_to_csv(ship_locations,filename):
    with open(filename, "w") as file:
        file.write("lon\tlat\n")
        for lat, lon in ship_locations:
            file.write(f"{lon}\t{lat}\n")


def show_image(key, ship, band, file_name, ais_df):
    if (key==1):
        display_ship_type="Dark ship"
        key=""
    else:
        ship_type = ais_df[ais_df["mmsi"] == int(key)]["ship_type"].values[0]
        display_ship_type = ship_type_dict.get(str(ship_type), "Other")
    
    img_3_channel = get_ship_box_image(ship, band)
    lat = ship.geo_centre[0]
    lon = ship.geo_centre[1]

    plt.figure(figsize=(5, 5))
    plt.imshow(img_3_channel, interpolation="nearest")
    plt.title(f"{key} - {display_ship_type}\nLatitude: {lat:.4f} | Longitude: {lon:.4f}")
    plt.axis('off')
    plt.show()
    acceptable = 2
    while acceptable not in ["0", "1"]:
        acceptable = input("Is this likely an image of a ship? (1- Yes, 0- No): \n")
    acceptable = int(acceptable)

    if acceptable:
        cv2.imwrite("SHIP_CATEGORISATION_IMAGES/dataset/images/" +
                    file_name, img_3_channel)
    return acceptable


def label_ships(image_path, ships, output_dir, ais_folder):
    ais_df = read_AIS_data(ais_folder)
    product = ProductIO.readProduct(image_path)
    band = product.getBand("Gamma0_VV_ocean")
    dark_ships=[]
    found_ships=[]

    if isinstance(ships,dict):
        all_ship_keys = ships.keys()

        for i, key in enumerate(all_ship_keys):
            ship = ships[key]
            file_name = "found_ship_{}.jpg".format(i)
            acceptable = show_image(key, ship, band, file_name, ais_df)

            if not acceptable:
                print("Image not accepted, skipping...")
                continue
            else:
                found_ships.append([ship.geo_centre[0],ship.geo_centre[1]])
    
    else:
        for i,ship in enumerate(ships):
            key=1
            file_name = "dark_ship_{}.jpg".format(i)
            acceptable = show_image(key, ship, band, file_name, ais_df)

            if not acceptable:
                print("Label not accepted, skipping...")
                continue
            else:
                dark_ships.append([ship.geo_centre[0],ship.geo_centre[1]])

            
    write_ships_to_csv(dark_ships,os.path.join(output_dir,"dark_ships.txt"))
    write_ships_to_csv(found_ships,os.path.join(output_dir,"found_ships.txt"))


YOLO1_conf_threshold=0.2
YOLO2_conf_threshold=0.7
classifier_conf_threshold=0.5
AIS_directory_path=""
path_to_SAR=""
output_directory=""
directory_flag=False
display_ships(path_to_SAR,AIS_directory_path,output_directory,directory_flag,YOLO1_conf_threshold,YOLO2_conf_threshold,classifier_conf_threshold)
