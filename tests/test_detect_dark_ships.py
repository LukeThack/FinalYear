from dark_ship_project.detect_dark_ships import get_max_min_xy,get_ship_image,get_ship_box_image,get_closest_mmsi
from dark_ship_project.read_SAR_data import Ship
from dark_ship_project.read_SAR_data import shoelace
from dark_ship_project.trajectory import update_AIS_data
import tempfile
import pandas
import os
import numpy
from datetime import datetime
from esa_snappy import ProductIO
def test_get_max_min_xy():
    ship=Ship([0,0],[0,0],[0,1,1,0,1,1,0,0],None)
    max_x,max_y,min_x,min_y=get_max_min_xy(ship)
    assert(min_x==0)
    assert(min_y==0)
    assert(max_y==1)
    assert(max_x==1)


def test_get_ship_image_centred_edge():
    product = ProductIO.readProduct("satelite/image12.dim")
    ship=Ship([0,0],[0,0],[0,1,1,0,1,1,0,0],None)
    band = product.getBand("Gamma0_VV_ocean")
    img_3_channel, start_x, start_y=ship_image=get_ship_image(ship, band, size=100)
    assert(start_x==0)
    assert(start_y==0)
    assert(img_3_channel.shape==(100,100,3))


def test_get_ship_image_centred_middle():
    product = ProductIO.readProduct("satelite/image12.dim")
    ship=Ship([0,0],[500,500],[0,1,1,0,1,1,0,0],None)
    band = product.getBand("Gamma0_VV_ocean")
    img_3_channel, start_x, start_y=ship_image=get_ship_image(ship, band, size=100)
    assert(start_x==450)
    assert(start_y==450)
    assert(img_3_channel.shape==(100,100,3))


def test_get_ship_box_image_shape():
    product = ProductIO.readProduct("satelite/image12.dim")
    ship=Ship([0,0],[500,500],[0,1,1,0,1,1,0,0],None)
    band = product.getBand("Gamma0_VV_ocean")
    img_3_channel=ship_image=get_ship_box_image(ship, band)
    assert(img_3_channel.shape==(1,1,3))


def test_find_largest_area_math():
    largest_area=0
    largest_bbox=None
    start_x=0
    start_y=0
    rotated_boxes=[[[0,0],[1,0],[1,1],[0,1]],[[0,0],[2,0],[2,2],[0,2]]]
    for rotated_box in rotated_boxes:
        x1 = int(rotated_box[0][0])
        y1 = int(rotated_box[0][1])
        x2 = int(rotated_box[1][0])
        y2 = int(rotated_box[1][1])
        x3 = int(rotated_box[2][0])
        y3 = int(rotated_box[2][1])
        x4 = int(rotated_box[3][0])
        y4 = int(rotated_box[3][1])

        bbox = [start_x+x1, start_y+y1, start_x+x2, start_y +
                y2, start_x+x3, start_y+y3, start_x+x4, start_y+y4]

        ship_area = shoelace(bbox)
        if ship_area > largest_area:
            largest_bbox = bbox
            largest_area = ship_area
    assert(largest_bbox==[0,0, 2,0, 2,2, 0,2])
    assert(largest_area==4)


def test_get_closest_mmsi_empty_folder():
    dt = datetime.strptime("2026-02-25 00:00:00", "%Y-%m-%d %H:%M:%S")
    ais_folder="non_existent"
    time_filter = update_AIS_data(dt, ais_folder)
    final_filter=[]
    result=get_closest_mmsi(final_filter, time_filter, 2, 2)
    assert(result==None)


def test_get_closest_mmsi_on_point():
    temp_dir=tempfile.TemporaryDirectory()
    data1=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:00:00", 10.0, 20.0, 30, 40]]#index 16 used therefore 17 entries required.
    data2=[[2, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:30:00", 20.0, 30.0, 40, 50]]
    file1=os.path.join(temp_dir.name,"data1.csv")
    file2=os.path.join(temp_dir.name,"data2.csv")
    df=pandas.DataFrame(data1)
    df.to_csv(file1,header=False,index=False)
    df=pandas.DataFrame(data2)
    df.to_csv(file2,header=False,index=False)

    dt = datetime.strptime("2026-02-25 00:00:00", "%Y-%m-%d %H:%M:%S")
    ais_folder=temp_dir.name
    time_filter = update_AIS_data(dt, ais_folder)
    final_filter=[1,2]
    result=get_closest_mmsi(final_filter, time_filter,20,30)
    assert(result==2)


def test_get_closest_mmsi():
    temp_dir=tempfile.TemporaryDirectory()
    data1=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:00:00", 10.0, 20.0, 30, 40]]#index 16 used therefore 17 entries required.
    data2=[[2, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:30:00", 20.0, 30.0, 40, 50]]
    file1=os.path.join(temp_dir.name,"data1.csv")
    file2=os.path.join(temp_dir.name,"data2.csv")
    df=pandas.DataFrame(data1)
    df.to_csv(file1,header=False,index=False)
    df=pandas.DataFrame(data2)
    df.to_csv(file2,header=False,index=False)

    dt = datetime.strptime("2026-02-25 00:00:00", "%Y-%m-%d %H:%M:%S")
    ais_folder=temp_dir.name
    time_filter = update_AIS_data(dt, ais_folder)
    final_filter=[1,2]
    result=get_closest_mmsi(final_filter, time_filter,17,27)
    assert(result==2)


def test_get_closest_mmsi_return_middle_of_points():
    temp_dir=tempfile.TemporaryDirectory()
    data1=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:00:00", 10.0, 20.0, 30, 40]]#index 16 used therefore 17 entries required.
    data2=[[2, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:30:00", 20.0, 30.0, 40, 50]]
    file1=os.path.join(temp_dir.name,"data1.csv")
    file2=os.path.join(temp_dir.name,"data2.csv")
    df=pandas.DataFrame(data1)
    df.to_csv(file1,header=False,index=False)
    df=pandas.DataFrame(data2)
    df.to_csv(file2,header=False,index=False)

    dt = datetime.strptime("2026-02-25 00:00:00", "%Y-%m-%d %H:%M:%S")
    ais_folder=temp_dir.name
    time_filter = update_AIS_data(dt, ais_folder)
    final_filter=[1,2]
    result=get_closest_mmsi(final_filter, time_filter,15,25)
    assert(result==1 or result==2)