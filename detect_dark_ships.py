from turtle import width

import cv2
from matplotlib.pylab import size
from readAIS import read_AIS_data
from readSARdata import read_SAR_data
import math
from getting_ship_vectors import get_coastline_vectors
import pandas as pd
from trajectory import update_AIS_data
from ultralytics import YOLO
from esa_snappy import ProductIO, PixelPos, GeoPos
import numpy
yolo_model=YOLO("runs/obb/train/weights/best.pt")

def find_dark_ships(ais_folder,sar_file,thresh_min,min_size):

    ship_locations,time=read_SAR_data(sar_file,thresh_min,min_size)
    
    time_filter=update_AIS_data(time,ais_folder)
    ship_found=dict()
    dark_ships=[]
    multi_ships=dict()

    for i in range(len(ship_locations)):
        ship=ship_locations[i]
        lat=ship.geo_centre[0]
        lon=ship.geo_centre[1]
        min_ship_long=math.floor(lon*100)/100 #position recorded wont be perfect
        min_ship_lat=math.floor(lat*100)/100
        lat_filter=time_filter[(time_filter["latitude"]>min_ship_lat)&(time_filter["latitude"]<min_ship_lat+0.01)] #+0.02 to allow for variance.
        final_filter=lat_filter[(lat_filter["longitude"]>min_ship_long)&(lat_filter["longitude"]<min_ship_long+0.01)]["mmsi"].unique()

        if len(final_filter)>0:
            current_dist=10000 #find closest ship
            closest_mmsi=None
            for mmsi in final_filter:
                ais_ship=time_filter[time_filter["mmsi"]==mmsi]
                discovered_lat = ais_ship.iloc[0]["latitude"]
                discovered_lon = ais_ship.iloc[0]["longitude"]
                dist=math.hypot(discovered_lat-lat,discovered_lon-lon)
                if dist<current_dist:
                    current_dist=dist
                    closest_mmsi=mmsi

            if ship.multiship_group: #if multiple ships flag present
                if ship.multiship_group not in multi_ships.keys():
                    multi_ships[ship.multiship_group]=[1,[ship,closest_mmsi]]
                else:
                    multi_ships[ship.multiship_group][0]+=1
                    multi_ships[ship.multiship_group].append([ship,closest_mmsi])
            else:              
                ship_found[int(closest_mmsi)]=ship #if ship found at coords, then not a dark ship.


            time_filter=time_filter[time_filter["mmsi"]!=closest_mmsi] #remove ship from further searches.
       
        else:
            if ship.multiship_group:
                if multi_ships.get(ship.multiship_group) is None:
                    multi_ships[ship.multiship_group]=[0,[ship,None]]
                else:
                    multi_ships[ship.multiship_group].append([ship,None])
            else:
                dark_ships.append(ship) #if ships not found at coords, then dark ship has been detected.

    for key in list(multi_ships.keys()):
        ship_data=multi_ships[key]

        if len(ship_data)-1==ship_data[0]: #if same number of ships detected as found in AIS, then no dark ship/s present.
            for ship in ship_data[1:]:
                ship_found[ship[1]]=ship[0] #add confirmed ship to found ships
            del multi_ships[key]

        elif len(ship_data)==2:
            dark_ships.append(ship_data[1][0]) #if only one ship detected and not found in AIS, then dark ship list.
            del multi_ships[key]



    return dark_ships,ship_found,multi_ships

'''
found_dark_ships,found_confirmed_ships,multi_ships=find_dark_ships("202306","satelite/image3.dim",250,50)
print(len(found_dark_ships),len(found_confirmed_ships),len(multi_ships))
dark_ships=[]
found_ships=[]

for key in found_confirmed_ships.keys():
    ship=found_confirmed_ships[key]
    found_ships.append([ship.geo_centre[0],ship.geo_centre[1]])

for key in multi_ships.keys():
    for ship in multi_ships[key][1:]:
        if ship[1] is not None:
            found_ships.append([ship[0].geo_centre[0],ship[0].geo_centre[1]])
        else:
            dark_ships.append([ship[0].geo_centre[0],ship[0].geo_centre[1]])

for ship in found_dark_ships:
    dark_ships.append([ship.geo_centre[0],ship.geo_centre[1]])



def write_ships_to_csv(ship_locations,filename):
    with open(filename, "w") as file:
        file.write("lon\tlat\n")
        for lat, lon in ship_locations:
            file.write(f"{lon}\t{lat}\n")



write_ships_to_csv(dark_ships,"ships_snap.txt")
write_ships_to_csv(found_ships,"ships_snap2.txt")
'''