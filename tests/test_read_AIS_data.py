import os
import pandas
import tempfile
import pytest
from dark_ship_project.read_AIS_data import read_AIS_data

def test_read_AIS_data_valid_file_multiple_entries():
    temp_dir=tempfile.TemporaryDirectory()
    data1=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:00:00", 10.0, 20.0, 30, 40]]#index 16 used therefore 17 entries required.
    data2=[[2, 0, 0, 0, 0, "TAN", 0, 0, 0, 0, 0, "column", "2026-02-25 00:00:00", 10.0, 20.0, 30, 40]]
    file1=os.path.join(temp_dir.name,"data1.csv")
    file2=os.path.join(temp_dir.name,"data2.csv")
    df=pandas.DataFrame(data1)
    df.to_csv(file1,header=False,index=False)
    df=pandas.DataFrame(data2)
    df.to_csv(file2,header=False,index=False)
    result=read_AIS_data(temp_dir.name)
    assert len(result)==2
    assert list(result.columns)==["mmsi", "ship_type", "navigational_status","timestamp", "latitude", "longitude", "speed", "course"]
    assert int(result.iloc[0]["mmsi"]) == 1
    assert int(result.iloc[1]["mmsi"]) == 2
    temp_dir.cleanup()


def test_read_AIS_data_no_entries():
    temp_dir=tempfile.TemporaryDirectory()
    result=read_AIS_data(temp_dir.name)
    assert list(result.columns)==["mmsi", "ship_type", "navigational_status","timestamp", "latitude", "longitude", "speed", "course"]
    temp_dir.cleanup()

def test_read_AIS_data_value_error():
    pytest.raises(ValueError,read_AIS_data,"non_existent_direcotry")
    


