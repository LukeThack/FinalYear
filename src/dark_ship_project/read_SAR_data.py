import numpy
import cv2
import matplotlib.pyplot as plt
from esa_snappy import ProductIO, PixelPos, GeoPos
from ultralytics import YOLO
from datetime import datetime
import os
import sys
yolo_model = YOLO("runs/obb/train/weights/best.pt")


'''
A function to find all ships within a provided SAR data image.

Parameters:
    image_path(string): the path to file of SAR image.

Returns:
    location of ships(list of ship objects): the ships found in image.
    dt(datetime): datetime when image was taken
'''


def read_SAR_data(image_path):
    multi_ship_group = 0
    if not os.path.exists(image_path):
        sys.exit(f"Error: File not found: {image_path}")
    try:
        product = ProductIO.readProduct(image_path)
    except Exception as e:
        sys.exit("Error: Product could not be read ",e)

    band = product.getBand("Gamma0_VV_ocean")
    if band==None:
        sys.exit("Gamma0_VV_ocean band not found in file")
    w = band.getRasterWidth()
    h = band.getRasterHeight()
    geoCoding = product.getSceneGeoCoding()
    slice_size = 640
    # iterate over smaller slices to avoid missing ships on the border.
    overlap = 128

    location_of_ships = []

    # split image to fit in YOLO memory limitation.
    for y in range(0, h, (slice_size-overlap)):
        for x in range(0, w, (slice_size-overlap)):
            width = min(slice_size, w-x)
            height = min(slice_size, h-y)
            # get processed SAR image.
            img = process_full_sar_image(band, height, width, x, y)
            sep_ship_detections, multi_ship_group = check_for_ships(
                img, x, y, geoCoding, multi_ship_group)  # get list of detections in image.
            if sep_ship_detections is not None:
                for ship_detection in sep_ship_detections:
                    already_exist = False
                    mid_x = ship_detection.pixel_centre[0]
                    mid_y = ship_detection.pixel_centre[1]

                    for location in location_of_ships.copy():
                        dist = numpy.hypot(
                            location.pixel_centre[0]-mid_x, location.pixel_centre[1]-mid_y)
                        if dist < 20:  # if two detected ships are within 20 pixels of eachother, assume they are the same ship.
                            already_exist = True
                            # if new detection has a larger bounding box, replace old one.
                            if location.area < ship_detection.area:
                                location_of_ships.remove(location)
                                location_of_ships.append(ship_detection)
                            break
                    if not already_exist:
                        location_of_ships.append(ship_detection)

    utc_time = product.getStartTime()  # need to return datetime of image.
    time = utc_time.toString()
    dt = datetime.strptime(time, "%d-%b-%Y %H:%M:%S.%f")

    return location_of_ships, dt


'''
Locates ships within a provided image

Parameters:
    img_3_channel(numpy ndarray): numpy array containing the image.
    start_x(int): x position of the top left corner relative to the image as a whole.
    start_y(int): y position of the top left corner relative to the image as a whole.
    geoCoding(esa snap GeoCoding object): geoCoding for image to convert pixel coordinates to longitdue latitude.
    multi_ship_group(int): a variable to give a unique id flag to ships close together

Returns:
    list: list of ship objects located
    int: the updated multi_ship_group flag
'''


def check_for_ships(img_3_channel, start_x, start_y, geoCoding, multi_ship_group):
    conf_threshold = 0.8
    # get bounding boxes from yolo model
    results = yolo_model(img_3_channel, conf=conf_threshold)
    ship_detections = results[0].obb
    ship_locations = []
    for box in ship_detections:
        if hasattr(box.cls, "item"):
            cls_id = int(box.cls.item())
        else:
            cls_id = int(box.cls)

        if cls_id == 0:
            rotated_box = box.xyxyxyxy[0]
            ship = build_ship(rotated_box, start_x, start_y, geoCoding)
            # save geo located centre,pixel centre and bounding box coords.
            ship_locations.append(ship)

    if len(ship_locations) == 0:
        return None, multi_ship_group
    elif len(ship_locations) > 1:
        for i in range(len(ship_locations)):
            # add flag to show multiple ships detected.
            ship_locations[i].multiship_group = multi_ship_group
        multi_ship_group += 1
        return ship_locations, multi_ship_group
    else:
        return ship_locations, multi_ship_group


'''
A class to contain all information relevant to a ship.
'''


class Ship:
    def __init__(self, geo_centre, pixel_centre, rbbox, multiship_group):
        self.geo_centre = geo_centre
        self.pixel_centre = pixel_centre
        self.rbbox = rbbox
        self.multiship_group = multiship_group
        self.area = shoelace(rbbox)


'''
Calculates the area of a polygon in space using the shoelace method (only works on 4 or more even number of points).

Parameters:
    bbox(list): a list containing the points x1,y1.. x4,y4

Returns:
    float: area within the 4 points.
'''


def shoelace(bbox):
    sum1 = 0
    sum2 = 0
    for i in range(0, len(bbox), 2):
        sum1 += bbox[i-2]*bbox[i+1]
        sum2 += bbox[i-1]*bbox[i]
    return 0.5*(abs(sum1-sum2))


'''
Read the subsection of a SAR image and process it for YOLO detection.

Parameters:
    band(esa snap band object): the image data.
    height(int): height of the SAR image
    width(int): width of the SAR image
    start_x(int): x position of the top left corner relative to the image as a whole.
    start_y(int): y position of the top left corner relative to the image as a whole.

Returns:
    numpy ndarray: the processed image.
'''


def process_full_sar_image(band, height, width, start_x, start_y):
    data = numpy.zeros((height, width))
    band.readPixels(start_x, start_y, width, height, data)
    data = numpy.log1p(data)  # YOLO trained on log1p data.
    nan_replace = numpy.percentile(data, 10)
    # replace nans with small value. make normalise less aggressive.
    data = numpy.nan_to_num(data, nan_replace)
    img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel = cv2.merge([img, img, img])
    return img_3_channel


'''
Instantiates the ship object.

Parameters:
    rotated_box(int 2D list): contains the points of the 4 corners of the bounding box.
    start_x(int): x position of the top left corner relative to the image as a whole.
    start_y(int): y position of the top left corner relative to the image as a whole.
    geoCoding(esa snap GeoCoding object): geoCoding for image to convert pixel coordinates to longitdue latitude.
'''


def build_ship(rotated_box, start_x, start_y, geoCoding):
    x1 = int(rotated_box[0][0])
    y1 = int(rotated_box[0][1])
    x2 = int(rotated_box[1][0])
    y2 = int(rotated_box[1][1])
    x3 = int(rotated_box[2][0])
    y3 = int(rotated_box[2][1])
    x4 = int(rotated_box[3][0])
    y4 = int(rotated_box[3][1])
    centre_x = (x1+x4)/2 + start_x
    centre_y = (y1+y4)/2 + start_y
    geo = GeoPos()
    geoCoding.getGeoPos(PixelPos(float(centre_x), float(centre_y)), geo)
    ship = Ship([geo.lat, geo.lon], [centre_x, centre_y], [start_x+x1, start_y+y1,
                start_x+x2, start_y+y2, start_x+x3, start_y+y3, start_x+x4, start_y+y4], None)
    return ship

