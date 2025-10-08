from readAIS import read_AIS_data
from readSARdata import read_SAR_data
import math

def find_dark_ships(start,end,ais_folder,sar_file,low,high,thresh_min,min_size):

    ship_locations=read_SAR_data(sar_file,low,high,thresh_min,min_size)
    ais_data=read_AIS_data(ais_folder)

    time_filter=ais_data[(ais_data["timestamp"]>=start) & (ais_data["timestamp"]<=end)]

    ship_found=dict()
    dark_ships=[]

    for i in range(len(ship_locations)):
        lat=ship_locations[i][0]
        long=ship_locations[i][1]
        min_ship_long=math.floor(long*100)/100 #position recorded wont be perfect
        min_ship_lat=math.floor(lat*100)/100
        lat_filter=time_filter[(time_filter["latitude"]>min_ship_lat)&(time_filter["latitude"]<min_ship_lat+0.02)] #+0.02 to allow for variance.
        final_filter=lat_filter[(lat_filter["longitude"]>min_ship_long)&(lat_filter["longitude"]<min_ship_long+0.02)]["mmsi"].unique()

        if len(final_filter)>0:
            ship_found[int(final_filter[0])]=[lat,long]
        else:
            dark_ships.append([lat,long]) #if ships not found at coords, then dark ship has been detected.

    return dark_ships,ship_found


#dark_ships,_=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","subset_1_of_mosaic_msk.dim",0.0001516640332, 0.04868127104362205,220,5)
#print(dark_ships)