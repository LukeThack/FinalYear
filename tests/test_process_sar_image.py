import tempfile
import os
from dark_ship_project.process_sar_image import next_id
import pandas
def test_process_sar_image_next_id():
    temp_dir=tempfile.TemporaryDirectory()
    data1=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:00:00", 10.0, 20.0, 30, 40]]#index 16 used therefore 17 entries required.
    data2=[[1, 0, 0, 0, 0, "MSC", 0, 0, 0, 0, 0, "column", "2026-02-25 00:30:00", 20.0, 30.0, 40, 50]]
    file1=os.path.join(temp_dir.name,"data1.csv")
    file2=os.path.join(temp_dir.name,"data2.csv")
    df=pandas.DataFrame(data1)
    df.to_csv(file1,header=False,index=False)
    df=pandas.DataFrame(data2)
    df.to_csv(file2,header=False,index=False)
    
    all_files = os.listdir(temp_dir.name)
    result=next_id(all_files)
    assert(result==3)
    temp_dir.cleanup()

def test_process_sar_image_next_id_empty_directory():
    temp_dir=tempfile.TemporaryDirectory()
    all_files = os.listdir(temp_dir.name)
    result=next_id(all_files)
    assert(result==0)
    temp_dir.cleanup()
