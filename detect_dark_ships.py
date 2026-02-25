import cv2
from read_SAR_data import read_SAR_data, shoelace
import math
import pandas
from trajectory import update_AIS_data
from ultralytics import YOLO
from esa_snappy import ProductIO
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy
yolo_model = YOLO("runs/obb/train/weights/best.pt")
ship_model = keras.models.load_model("ship_classifier_final.keras")

'''
Finds the dark ships within a SAR image

Parameters:
    ais_folder(string): path to the folder where the AIS data is held
    sar_file(string): path to the SAR image file (.dim)

Returns:
    dark_ships(ship class list): list of dark ships
    ships_found(ship class dictionary): dictionary mmsi:ship object
    multi_ships(ship class dictionary): multi_ship_group:number of ships in group, list of ships in group
'''


def find_dark_ships(ais_folder, sar_file):
    ship_locations, time = read_SAR_data(sar_file)
    time_filter = update_AIS_data(time, ais_folder)
    ship_found = dict()
    dark_ships = []
    multi_ships = dict()
    product = ProductIO.readProduct(sar_file)
    band = product.getBand("Gamma0_VV_ocean")
    size = 100
    ships_found = []

    for ship in ship_locations:
        img_3_channel, start_x, start_y = get_ship_image(ship, band, size)
        conf_threshold = 0.7
        # get bounding boxes from yolo model
        results = yolo_model(img_3_channel, conf=conf_threshold)
        ship_detections = results[0].obb
        largest_area, largest_bbox = find_largest_area(
            ship_detections, start_x, start_y)

        if ship_detections is not None and len(ship_detections) > 0 and largest_area > 0:
            processed_image = pre_process_ship_image(ship, band)
            prob_not_ship = ship_model.predict(processed_image)[0][0]
            if (1-prob_not_ship) > 0.5:  # if most likely a ship
                ship.rbbox = largest_bbox
                ship.area = largest_area
                ships_found.append(ship)

    for ship in ships_found:
        lat = ship.geo_centre[0]
        lon = ship.geo_centre[1]
        # position recorded wont be perfect
        min_ship_long = math.floor(lon*100)/100
        min_ship_lat = math.floor(lat*100)/100
        lat_filter = time_filter[(time_filter["latitude"] > min_ship_lat) & (
            time_filter["latitude"] < min_ship_lat+0.01)]  # +0.01 to allow for non perfect trajectory tracking
        final_filter = lat_filter[(lat_filter["longitude"] > min_ship_long) & (
            lat_filter["longitude"] < min_ship_long+0.01)]["mmsi"].unique()

        if len(final_filter) > 0:
            # df passed by reference.
            closest_mmsi = get_closest_mmsi(
                final_filter, time_filter, lat, lon)

            if ship.multiship_group:  # if multiple ships flag present
                if ship.multiship_group not in multi_ships.keys():
                    multi_ships[ship.multiship_group] = [
                        1, [ship, closest_mmsi]]
                else:
                    multi_ships[ship.multiship_group][0] += 1
                    multi_ships[ship.multiship_group].append(
                        [ship, closest_mmsi])
            else:
                # if ship found at coords, then not a dark ship.
                ship_found[int(closest_mmsi)] = ship

            # remove ship from further searches.
            time_filter = time_filter[time_filter["mmsi"] != closest_mmsi]

        else:
            if ship.multiship_group:
                if multi_ships.get(ship.multiship_group) is None:
                    multi_ships[ship.multiship_group] = [0, [ship, None]]
                else:
                    multi_ships[ship.multiship_group].append([ship, None])
            else:
                # if ships not found at coords, then dark ship has been detected.
                dark_ships.append(ship)

    for key in list(multi_ships.keys()):
        ship_data = multi_ships[key]

        # if same number of ships detected as found in AIS, then no dark ship/s present.
        if len(ship_data)-1 == ship_data[0]:
            for ship in ship_data[1:]:
                # add confirmed ship to found ships
                ship_found[ship[1]] = ship[0]
            del multi_ships[key]

        elif len(ship_data) == 2:
            # if only one ship detected and not found in AIS, then dark ship list.
            dark_ships.append(ship_data[1][0])
            del multi_ships[key]

    return dark_ships, ship_found, multi_ships


'''
Rotates the pixels within a ship's bounding box to be upright.

Parameters:
    ship(ship object): The ship to rotate
    image(numpy ndarray): The image where the ship is located.

Returns:
    numpy ndarry: the rotated image
'''


def rotate_image(ship, image):
    max_x, max_y, min_x, min_y = get_max_min_xy(ship)
    height = max_y-min_y
    width = max_x-min_x
    x1, y1, x2, y2, x3, y3, x4, y4 = ship.rbbox

    points = numpy.array([[x1, y1], [x2, y2], [x3, y3],
                         [x4, y4]], dtype=numpy.float32)
    points[:, 0] -= min_x
    points[:, 1] -= min_y
    centre, (rect_width, rect_height), angle = cv2.minAreaRect(points)

    if rect_height < rect_width:  # rotate image to be vertical.
        angle += 90
        rect_width, rect_height = rect_height, rect_width

    rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    rad_angle = math.radians(angle)
    sin = abs(math.sin(rad_angle))
    cos = abs(math.cos(rad_angle))
    # need to be adjusted after rotation to prevent cut-off
    new_width = int((height*sin)+(width*cos))
    new_height = int((height*cos)+(width*sin))

    centre_offset_x = (new_width/2)-centre[0]
    centre_offset_y = (new_height/2)-centre[1]
    # move image to centre of new canvas, 2x2 rotation, 2x1 translation to make 2x3 matrix.
    rotation_matrix[0, 2] += centre_offset_x
    rotation_matrix[1, 2] += centre_offset_y

    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height))

    h, w = rotated_image.shape[:2]
    min_x_crop = max(0, int((w-rect_width)/2))  # avoid negatives
    max_x_crop = min(w, int((w+rect_width)/2))
    min_y_crop = max(0, int((h-rect_height)/2))
    max_y_crop = min(h, int((h+rect_height)/2))
    rotated_image = rotated_image[min_y_crop:max_y_crop, min_x_crop:max_x_crop]
    return rotated_image


'''
Get the maximum and minimum x,y value from a ships bounding box

Parameters:
    ship(ship object)

Returns:
    max_x(int)
    max_y(int)
    min_x(int)
    min_y(int)
'''


def get_max_min_xy(ship):
    bbox = ship.rbbox
    x1, y1 = bbox[0], bbox[1]
    max_x = x1
    min_x = x1
    max_y = y1
    min_y = y1

    for i in range(0, len(bbox), 2):
        x = bbox[i]
        y = bbox[i+1]
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y
    return max_x, max_y, min_x, min_y


'''
Gets the pixels within a the bounding box of a ship and applies filters for model prediction.

Parameters:
    ship(ship object)
    band(ESA snappy band object): pixel data which contains the provided ship.
Returns:
    numpy ndarray - rotated and processed image of pixels within bounding box.
'''


def get_ship_box_image(ship, band):
    max_x, max_y, min_x, min_y = get_max_min_xy(ship)
    height = max_y-min_y
    width = max_x-min_x
    # has to be a flat array for read pixels of small values.
    flat_array = numpy.zeros(width*height, dtype=numpy.float32)
    band.readPixels(min_x, min_y, width, height, flat_array)
    data = flat_array.reshape((height, width))
    # replace nans with small value. make normalise less aggressive.
    data = numpy.nan_to_num(data, nan=1e-6)
    data = numpy.log1p(data)
    mean, std = numpy.mean(data), numpy.std(data)
    high, low = mean+std, mean
    data = numpy.clip(data, low, high)  # remove extreme values
    img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    rotated_image = rotate_image(ship, img)
    img_3_channel = cv2.merge([rotated_image, rotated_image, rotated_image])
    return img_3_channel


'''
Produces a sizexsize image centred at the centre of a ship for YOLO model prediction. 

Parameters:
    ship(ship object)
    band(ESA snappy band object): pixel data which contains the provided ship.
    size: size of image function needs to produce.

Returns:
    img_3_channel (numpy ndarray): image in format for YOLO
    start_x(int): x coordinate of top left corner relative to whole image
    start_y(int): y coordinate of top left corner relative to whole image
'''


def get_ship_image(ship, band, size=100):
    x = int(ship.pixel_centre[0])
    y = int(ship.pixel_centre[1])
    if x < size:
        start_x = 0
    else:
        start_x = int(x-size//2)
    if y < size:
        start_y = 0
    else:
        start_y = int(y-size//2)
    data = numpy.zeros((size, size))
    band.readPixels(start_x, start_y, size, size, data)
    nan_replace = numpy.percentile(data, 10)
    # replace nans with small value. make normalise less aggressive.
    data = numpy.nan_to_num(data, nan_replace)
    data = numpy.log1p(data)
    mean, std = numpy.mean(data), numpy.std(data)
    high, low = mean+std, mean
    data = numpy.clip(data, low, high)  # remove extreme values
    img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel = cv2.merge([img, img, img])
    return img_3_channel, start_x, start_y


'''
Finds the detection with the largest area (most likely the ship)

Parameters:
    ship_detections - results from YOLO model
    start_x(int): x coordinate of top left corner relative to whole image of the image containing all the detections.
    start_y(int): y coordinate of top left corner relative to whole image of the image containing all the detections.

Returns:
    lagest_area(float): area or largest bbox.
    largest_box(int array): pixel coordinates of largest boundary box.
'''


def find_largest_area(ship_detections, start_x, start_y):
    largest_area = 0
    largest_bbox = None
    for box in ship_detections:
        if hasattr(box.cls, "item"):
            cls_id = int(box.cls.item())
        else:
            cls_id = int(box.cls)

        if cls_id == 0:
            rotated_box = box.xyxyxyxy[0]
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
    return largest_area, largest_bbox


'''
Retrieves ship bounding box image and applies filters .

Parameters:
    ship(ship object)
    band(ESA snappy band object): pixel data which contains the provided ship.

Returns:
    processed_image (numpy ndarray): image in format for model to predict.
'''


def pre_process_ship_image(ship, band):
    processed_image = get_ship_box_image(ship, band)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_image = cv2.resize(processed_image, (224, 224))
    processed_image = preprocess_input(processed_image)
    processed_image = processed_image.astype(numpy.float32)
    processed_image = numpy.expand_dims(processed_image, axis=0)
    return processed_image


'''
Finds the mmsi closest to the provided position.
    
Parameters:
    final_filter(list): list of mmsi close to point
    time_filter(pandas dataframe): contains location of each mmsi at the time SAR image was taken.
    lat(float): latitude of position
    lon(float): longitude of position

Returns:
    string - mmsi closest to point provided,
'''


def get_closest_mmsi(final_filter, time_filter, lat, lon):
    closest_mmsi = None
    current_dist = 10000  # set to ridiculously large value
    for mmsi in final_filter:
        ais_ship = time_filter[time_filter["mmsi"] == mmsi]
        discovered_lat = ais_ship.iloc[0]["latitude"]
        discovered_lon = ais_ship.iloc[0]["longitude"]
        dist = math.hypot(discovered_lat-lat, discovered_lon-lon)
        if dist < current_dist:
            current_dist = dist
            closest_mmsi = mmsi
    return closest_mmsi


'''
found_dark_ships,found_confirmed_ships,multi_ships=find_dark_ships("202306","satelite/image12.dim",250,50)
print(len(found_dark_ships),len(found_confirmed_ships),len(multi_ships))
dark_ships=[]
found_ships=[]

for key in found_confirmed_ships.keys():
    ship=found_confirmed_ships[key]
    found_ships.append([ship.geo_centre[0],ship.geo_centre[1]])

for key in multi_ships.keys():
    for ship in multi_ships[key][1:]:
        if ship[1] is not None:
            found_ships.append([ship[0].geo_centre[0],ship[0].geo_centre[1]])
        else:
            dark_ships.append([ship[0].geo_centre[0],ship[0].geo_centre[1]])

for ship in found_dark_ships:
    dark_ships.append([ship.geo_centre[0],ship.geo_centre[1]])



def write_ships_to_csv(ship_locations,filename):
    with open(filename, "w") as file:
        file.write("lon\tlat\n")
        for lat, lon in ship_locations:
            file.write(f"{lon}\t{lat}\n")



write_ships_to_csv(dark_ships,"ships_snap.txt")
write_ships_to_csv(found_ships,"ships_snap2.txt")
'''
