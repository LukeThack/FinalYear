from detect_dark_ships import find_dark_ships
from esa_snappy import ProductIO
import numpy
import cv2
import json
import matplotlib.pyplot as plt
found_dark_ships,ship_found,multi_ship=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","subset_1_of_mosaic_msk.dim",0.0001516640332, 0.04868127104362205,250,5)
from ultralytics import YOLO
yolo_model=YOLO("runs/detect/SSDD_YOLO/weights/best.pt")


def show_image(low,high,ship,band,file_name):
    x=int(ship[1])
    y=int(ship[2])
    if x<100:
        start_x=0
    else:
        start_x=int(x-100//2)
    if y<100:
        start_y=0
    else:
        start_y=int(y-100//2)
    data=numpy.zeros((100,100))
    band.readPixels(start_x,start_y,100,100,data)
    data=numpy.clip(data,low,high)
    img=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])
    img_with_boxes=img_3_channel.copy()
    results=yolo_model(img_3_channel)
    box=results[0].boxes.xyxy[0]
    x1=int(box[0])
    y1=int(box[1])
    x2=int(box[2])
    y2=int(box[3])
    
    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green box, thickness 2

    plt.imshow(img_with_boxes)
    plt.title("Ship "+file_name)
    plt.axis('off')
    plt.show()
    category=0
    while category not in [1,2,3]:
        category=int(input("Enter category for ship (1- Ship, 2- Likely Infrastructure, 3- Ambiguous): \n"))
    acceptable=2
    while acceptable not in [0,1]:
        acceptable=int(input("Is this label acceptable? (1- Yes, 0- No): \n"))
    if acceptable:
        cv2.imwrite("SSDD_YOLO_IMAGES/dataset/images/train/"+file_name, img_3_channel)

    return results,category,acceptable




def label_ships(image_path,low,high,found_dark_ships,json_file,max_image_id,max_annotation_id):
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Sigma0_VH")

    for i,ship in enumerate(found_dark_ships):
        file_name="luke_ship_{}.jpg".format(i+max_image_id)
        results,category,acceptable=show_image(low,high,ship,band,file_name)
        box=results[0].boxes.xyxy[0]
        x1=float(box[0])
        y1=float(box[1])
        x2=float(box[2])
        y2=float(box[3])
        area=(x2 - x1) * (y2 - y1)
        if not acceptable:
            print("Label not accepted, skipping...")
            continue
        else:
            new_image={
                "file_name":file_name,
                "height":100,
                "width":100,
                "id":max_image_id+i,
            }
            new_annotation={
                "area":area,
                "iscrowd":0,
                "image_id":max_image_id+i,
                "bbox":[x1,y1,x2-x1,y2-y1],
                "category_id":category,
                "id":max_annotation_id+i,
            }
            categories={
                "1":"Ship",
                "2":"Infrastructure",
                "3":"Ambiguous"
            }
            with open(json_file,"r") as f:
                data=json.load(f)
            data["images"].append(new_image)
            data["annotations"].append(new_annotation)
            data["categories"].append(categories)
            with open(json_file,"w") as f:
                json.dump(data,f,indent=4) 

label_ships("subset_1_of_mosaic_msk.dim",0.0001516640332, 0.04868127104362205,found_dark_ships,"SSDD_YOLO_IMAGES/dataset/annotations/instances_train.json",1161,2042)
