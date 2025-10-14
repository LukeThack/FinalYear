import numpy
import cv2
import matplotlib.pyplot as plt
from esa_snappy import ProductIO, PixelPos



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
        pixel_position = PixelPos(float(x), float(y))
        long_lat_position = geoCoding.getGeoPos(pixel_position, None)
        location_of_ships.append([long_lat_position.lat, long_lat_position.lon])


    return location_of_ships
    


    


#cropped=img[row:row,column:column]
#cv2.cvtcolor(pic,cv2.COLOR_GRAY2RGB)
#cv2.circle(pic,center,radius,(colour))
#pic.shape[1] (width) pic.shape[0](height)
#contours = cv2.findContours(pic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) cv2.RETR_LIST for all, cv2.RETR_EXTERNAL for all cv2.RETR_TREE not needed, could also use CHAIN_APPROX_NONE
#cv2.threshold(pic, min, max, cv2.THRESH_BINARY) anything below min or above max set to black, else set to white.