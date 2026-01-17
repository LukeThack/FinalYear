import numpy
import cv2
import matplotlib.pyplot as plt
from esa_snappy import ProductIO, PixelPos, GeoPos
from ultralytics import YOLO
yolo_model=YOLO("runs/obb/train/weights/best.pt")



def get_centroid(contour):
    moments = cv2.moments(contour)  #dict of weighted pixel intensities
    if moments["m00"] > 0:
        return int(moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"]) #credit geekforgeeks, returns average centre of mass of contours.
    return None


def process_slice(band,start_x,start_y,width,height,low,high,threshold_min,min_area):
    data=numpy.zeros((height,width))
    band.readPixels(start_x,start_y,width,height,data)
    data=numpy.clip(data,low,high)
    img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    binary = cv2.threshold(img, threshold_min, 255, cv2.THRESH_BINARY)[1]
    pre_contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours=[]
    for c in pre_contours:
        if cv2.contourArea(c) > min_area:
            c[:,0,0]+=start_x
            c[:,0,1]+=start_y
            contours.append(c)

    centroids=[]
    for c in contours:
        centroids.append(get_centroid(c))

    return centroids,contours




def read_SAR_data(image_path,low,high,threshold_min,min_area):

    multi_ship_group=0
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Sigma0_VH")
    w=band.getRasterWidth()
    h=band.getRasterHeight()
    geoCoding = product.getSceneGeoCoding()
    centroids=[]
    contours=[]
    slice_size=1000

    for y in range(0,h,slice_size):
        for x in range(0,w,slice_size):
            width=min(slice_size,w-x)
            height=min(slice_size,h-y)
            new_centroids,new_contours=process_slice(band,x,y,width,height,low,high,threshold_min,min_area)
            centroids.extend(new_centroids)
            contours.extend(new_contours)

    merged_contours = []
    skip = set()
    for i,contour in enumerate(contours):
        if i not in skip:
            cx1,cy1=centroids[i]
            same_dot=[contour]
            for j,contour2 in enumerate(contours): #check for every dot to see if it is within dist pixels from eachother.
                if not (j<=i or j in skip):
                    cx2, cy2 = centroids[j]
                    dist = numpy.hypot(cx1-cx2, cy1-cy2)
                    if dist < 50: #merge threshold in pixels.
                        same_dot.append(contour2)
                        skip.add(j)
            merged=numpy.vstack(same_dot) #must be a numpy array for cv.
            merged_contours.append(merged)
            

    location_of_ships=[]
    for c in merged_contours:
        min_enclosing_circle=cv2.minEnclosingCircle(c)
        (x, y) = min_enclosing_circle[0]
        sep_ship_detections,multi_ship_group=check_contour_multiple_ships(band,x,y,300,geoCoding,low,high,multi_ship_group)
        if sep_ship_detections is not None:
            for ship_detection in sep_ship_detections:
                already_exist=False
                mid_x=ship_detection.pixel_centre[0]
                mid_y=ship_detection.pixel_centre[1]

                for location in location_of_ships:
                    dist=numpy.hypot(location.pixel_centre[0]-mid_x,location.pixel_centre[1]-mid_y)
                    if dist<20: #if two detected ships are within 20 pixels of eachother, assume they are the same ship.
                        already_exist=True
                        if location.area<ship_detection.area: #if new detection has a larger bounding box, replace old one.
                            location_of_ships.remove(location)
                            location_of_ships.append(ship_detection)
                        break
                if not already_exist:
                    location_of_ships.append(ship_detection)
            





    return location_of_ships
    


def check_contour_multiple_ships(band,x,y,size,geoCoding,low,high,multi_ship_group):
    if x<size:
        start_x=0
    else:
        start_x=int(x-size//2)
    if y<size:
        start_y=0
    else:
        start_y=int(y-size//2)

    data=numpy.zeros((size,size))
    band.readPixels(start_x,start_y,size,size,data)
    data=numpy.clip(data,low,high)
    img=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])

    results=yolo_model(img_3_channel) #get bounding boxes from yolo model

    ship_detections=results[0].obb
    ship_locations=[]
    for box in ship_detections:
        if hasattr(box.cls,"item"):
            cls_id=int(box.cls.item())
        else:
            cls_id=int(box.cls)
        if cls_id==0:
            rotated_box=box.xyxyxyxy[0]
            x1=int(rotated_box[0][0])
            y1=int(rotated_box[0][1])
            x2=int(rotated_box[1][0])
            y2=int(rotated_box[1][1])
            x3=int(rotated_box[2][0])
            y3=int(rotated_box[2][1])
            x4=int(rotated_box[3][0])
            y4=int(rotated_box[3][1])

            centre_x=(x1+x4)/2 + start_x
            centre_y=(y1+y4)/2 + start_y
            geo = GeoPos()
            geoCoding.getGeoPos(
                PixelPos(float(centre_x), float(centre_y)),
                geo
            )
            ship=Ship([geo.lat,geo.lon],[centre_x,centre_y],[x1,y1,x2,y2,x3,y3,x4,y4],None)
            ship_locations.append(ship) #save geo located centre,pixel centre and bounding box coords.

    if len(ship_locations)==0:
        return None,multi_ship_group
    elif len(ship_locations)>1:
        for i in range (len(ship_locations)):
            ship_locations[i].multiship_group=multi_ship_group #add flag to show multiple ships detected.
        multi_ship_group+=1
        return ship_locations, multi_ship_group
    else:
        return ship_locations,multi_ship_group


    
class Ship:
    def __init__(self,geo_centre,pixel_centre,rbbox,multiship_group):
        self.geo_centre=geo_centre
        self.pixel_centre=pixel_centre
        self.rbbox=rbbox
        self.multiship_group=multiship_group
        self.area=shoelace(rbbox)
        

    
def shoelace(bbox):
    sum1=0
    sum2=0
    for i in range(0,len(bbox),2):
        sum1+=bbox[i-2]*bbox[i+1]
        sum2+=bbox[i-1]*bbox[i]
    return 0.5*(abs(sum1-sum2))
    


#cropped=img[row:row,column:column]
#cv2.cvtcolor(pic,cv2.COLOR_GRAY2RGB)
#cv2.circle(pic,center,radius,(colour))
#pic.shape[1] (width) pic.shape[0](height)
#contours = cv2.findContours(pic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) cv2.RETR_LIST for all, cv2.RETR_EXTERNAL for all cv2.RETR_TREE not needed, could also use CHAIN_APPROX_NONE
#cv2.threshold(pic, min, max, cv2.THRESH_BINARY) anything below min or above max set to black, else set to white.