import json
import os

coco_json_path="SSDD_YOLO_IMAGES/dataset/annotations/instances_val.json" 
images_dir="SSDD_YOLO_IMAGES/dataset/images"
output_dir="SSDD_YOLO_IMAGES/dataset/labels/val"
os.makedirs(output_dir,exist_ok=True)

with open(coco_json_path) as file:
    data=json.load(file)

image_info={img['id']:img for img in data['images']}

for annotation in data['annotations']:
    id=annotation['image_id']
    img=image_info[id]

    image_width,image_height=img['width'],img['height']
    x,y,w,h=annotation['bbox']
    x_center=(x+w/2)/image_width #format for yolo labels, all require normalised values between 0 and 1
    y_center=(y+h/2)/image_height
    w_norm=w/image_width
    h_norm=h/image_height

    line=f"{annotation['category_id']} {x_center} {y_center} {w_norm} {h_norm}\n" #expects each variable with space separation

    image_filename=os.path.basename(img['file_name']).replace(".jpg",".txt")
    txt_path=os.path.join(output_dir,image_filename)

    with open(txt_path,"a") as f:
        f.write(line)