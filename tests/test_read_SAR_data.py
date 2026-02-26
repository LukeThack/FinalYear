from dark_ship_project.read_SAR_data import read_SAR_data,check_for_ships,process_full_sar_image,shoelace
import sys
from esa_snappy import ProductIO, PixelPos, GeoPos
import pytest
from datetime import datetime

def test_check_for_ships_return_values():
    product = ProductIO.readProduct("satelite/image12.dim")
    band = product.getBand("Gamma0_VV_ocean")
    height=1000
    width=1000
    geoCoding = product.getSceneGeoCoding()
    x=0
    y=0
    multi_ship_group=1
    img = process_full_sar_image(band, height, width, x, y)
    sep_ship_detections, multi_ship_group = check_for_ships(img, x, y, geoCoding, multi_ship_group)
    assert(sep_ship_detections==None)
    assert isinstance(multi_ship_group,int)


def test_read_sar_data_non_existent_directory():
    pytest.raises(SystemExit,read_SAR_data,"non_existent_direcotry")


def test_read_sar_data_real_file_no_product():
    pytest.raises(SystemExit,read_SAR_data,"satelite/image13.dim")


def test_read_sar_data_working_test():
    ships,dt=read_SAR_data("satelite/image12.dim")
    assert isinstance(dt,datetime)
    assert isinstance(ships,list)
    

def test_shoelace_function():
    bbox=[2,7,10,1,8,6,11,7,7,10]
    result=shoelace(bbox)
    assert(int(result)==32)
