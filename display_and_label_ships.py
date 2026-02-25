from detect_dark_ships import find_dark_ships, get_ship_box_image
from process_sar_image import calc_max_id
from esa_snappy import ProductIO
from read_AIS_data import read_AIS_data
import numpy
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
yolo_model = YOLO("runs/obb/train/weights/best.pt")

'''
ship_type_dict={
    "MER": "Merchant",
    "TNK": "Tanker",
    "FSH": "Fishing",
    "TUG": "Tug",
    "MIL": "Military",
}
'''
ship_type_dict = {
    "MER": 0,
    "TNK": 1,
    "FSH": 2,
    "TUG": 3,
    "MIL": 4,
}


def show_image(key, ship, band, file_name, ais_df):
    ship_type = ais_df[ais_df["mmsi"] == int(key)]["ship_type"].values[0]
    display_ship_type = ship_type_dict.get(str(ship_type), 5)
    img_3_channel = get_ship_box_image(ship, band)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_3_channel, interpolation="nearest")
    plt.axis('off')
    plt.show()

    category = 2
    acceptable = 2

    while category not in ["0", "1"]:
        category = input(
            "Enter category for ship (0-Ship, 1-Background Noise): \n")
    category = int(category)
    while acceptable not in ["0", "1"]:
        acceptable = input("Is this image acceptable? (1- Yes, 0- No): \n")
    acceptable = int(acceptable)

    if acceptable:
        cv2.imwrite("SHIP_CATEGORISATION_IMAGES/dataset2/images/" +
                    file_name, img_3_channel)

    return category, acceptable


def label_ships(image_path, confirmed_ships, output_dir, max_image_id, ais_folder):
    ais_df = read_AIS_data(ais_folder)
    product = ProductIO.readProduct(image_path)
    band = product.getBand("Gamma0_VV_ocean")
    all_ship_keys = confirmed_ships.keys()

    for i, key in enumerate(all_ship_keys):
        # for i,ship in enumerate(confirmed_ships):
        ship = confirmed_ships[key]
        # key=1
        file_name = "luke_ship_{}.jpg".format(i+max_image_id)
        category, acceptable = show_image(key, ship, band, file_name, ais_df)

        if not acceptable:
            print("Label not accepted, skipping...")
            continue

        output_file_path = os.path.join(
            output_dir, os.path.basename(file_name).replace(".jpg", ".txt"))
        with open(output_file_path, "w") as output_file:
            output_file.write(str(category))


'''
many_found_ships={}
directory="satelite"
all_files=os.listdir(directory)
for file in all_files:
    if file.endswith(".dim"):
        file_name=os.path.join(directory,file)
        dark_ships,found_confirmed_ships,multi_ships=find_dark_ships("202306",file_name,250,50)
        many_found_ships[file_name]=found_confirmed_ships

for key in many_found_ships.keys():
    found_ships=many_found_ships[key]
    max_image_id=calc_max_id(all_files)
    label_ships(key,found_ships,"SHIP_CATEGORISATION_IMAGES/dataset2/labels/",max_image_id,"202306")
'''


dark_ships, found_confirmed_ships, multi_ships = find_dark_ships(
    "202306", 'satelite/image12.dim', 250, 50)
label_ships('satelite/image12.dim', found_confirmed_ships,
            "SHIP_CATEGORISATION_IMAGES/dataset2/labels/", 0, "202306")
