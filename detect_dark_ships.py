from readAIS import read_AIS_data
from readSARdata import read_SAR_data
import math
from getting_shp_vectors import get_coastline_vectors
import pandas as pd

def find_dark_ships(start,end,ais_folder,sar_file,low,high,thresh_min,min_size):

    ship_locations=read_SAR_data(sar_file,low,high,thresh_min,min_size)
    ais_data=read_AIS_data(ais_folder)

    time_filter=ais_data[(ais_data["timestamp"]>=start) & (ais_data["timestamp"]<=end)]

    ship_found=dict()
    dark_ships=[]
    multi_ships=dict()

    for i in range(len(ship_locations)):
        ship=ship_locations[i]
        ship[0].lat
        lon=ship[0].lon
        min_ship_long=math.floor(lon*100)/100 #position recorded wont be perfect
        min_ship_lat=math.floor(lat*100)/100
        lat_filter=time_filter[(time_filter["latitude"]>min_ship_lat)&(time_filter["latitude"]<min_ship_lat+0.02)] #+0.02 to allow for variance.
        final_filter=lat_filter[(lat_filter["longitude"]>min_ship_long)&(lat_filter["longitude"]<min_ship_long+0.02)]["mmsi"].unique()

        if len(final_filter)>0:
            if len(ship)==8: #if multiple ships flag present
                if ship[7] not in multi_ships:
                    multi_ships[ship[7]]=[1,ship]
                else:
                    multi_ships[ship[7]][0]+=1
                    multi_ships[ship[7]].append(ship)
            else:              
                ship_found[int(final_filter[0])]=ship #if ship found at coords, then not a dark ship.

            #need to find closest ship to remove
            time_filter=time_filter[time_filter["mmsi"]!=final_filter[0]] #remove ship from further searches.
       
        else:
            if len(ship)==8:
                if multi_ships.get(ship[7]) is None:
                    multi_ships[ship[7]]=[0,ship]
                else:
                    multi_ships[ship[7]].append(ship)
            else:
                dark_ships.extend(ship) #if ships not found at coords, then dark ship has been detected.

    for key in multi_ships.keys():
        ship_data=multi_ships[key]
        if len(ship_data)-1==ship_data[0]: #if ships detected than found in AIS, then no dark ship/s present.
            del multi_ships[key]




    return dark_ships,ship_found,multi_ships


dark_ships,_=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","mosaic_msk.dim",0.0001516640332, 0.04868127104362205,250,5)

'''
df = pd.read_csv("ships_snap.txt", delim_whitespace=True)
dark_ships = df[["lon", "lat"]].to_numpy()
dark_ships = dark_ships.tolist()
'''
coastline=get_coastline_vectors("coastlines-split-4326")


new_dark_ships=[]
for lat,lon in dark_ships[:]:
    min_ship_long=math.floor(lon*100)/100 #position recorded wont be perfect
    min_ship_lat=math.floor(lat*100)/100
    lat_filter=coastline[(coastline["latitude"]>min_ship_lat-0.035)&(coastline["latitude"]<min_ship_lat+0.035)] #within 0.035, about 2.2km from a coastline, ignore result.
    final_filter=lat_filter[(lat_filter["longitude"]>min_ship_long-0.035)&(lat_filter["longitude"]<min_ship_long+0.035)]

    if len(final_filter)>0 or lat<50.4: #50.4 is minimum latitude for irish sea, remove if finding is close enough to land.
        pass
    else:
        new_dark_ships.append([lat,lon])

print("updated dark ships")

def write_ships_to_csv(ship_locations):
    with open("ships_snap.txt", "w") as file:
        file.write("lon\tlat\n")
        for lat, lon in ship_locations:
            file.write(f"{lon}\t{lat}\n")



write_ships_to_csv(new_dark_ships)
