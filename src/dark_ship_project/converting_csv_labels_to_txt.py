import pandas as pd
import os
import math
'''
A file to convert the SSDD dataset csv labels into individual text files
'''
excel_path = "SSDD_RBOX_IMAGES/dataset/labels/test.csv"
output_dir = "SSDD_RBOX_IMAGES/dataset/labels/val/"

df = pd.read_csv(excel_path, header=None, engine="python", sep="\\n")


current_image = None
filename = None
width = None
height = None
file_name = None
output_file = None

for index, row in df.iterrows():

    line = row[0].strip()
    if line.startswith("ship"):
        list_of_values = line.split(",")
        # location of x1,y1 ... x4,y4 for the rotated bounding box in SSDD csv file.
        x1 = int(list_of_values[24])/width
        y1 = int(list_of_values[26])/height
        x2 = int(list_of_values[28])/width
        y2 = int(list_of_values[30])/height
        x3 = int(list_of_values[32])/width
        y3 = int(list_of_values[34])/height
        x4 = int(list_of_values[36])/width
        y4 = int(list_of_values[38])/height
    else:
        # always gets updated to correct image first.
        file_name, dimensions = line.strip().split(",")
        width, height = dimensions.split("x")
        width = int(width)
        height = int(height)
        continue

    output_line = "0 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
        x1, y1, x2, y2, x3, y3, x4, y4)
    output_file_path = os.path.join(
        output_dir, os.path.basename(file_name).replace(".jpg", ".txt"))
    with open(output_file_path, "a") as output_file:
        output_file.write(output_line)
