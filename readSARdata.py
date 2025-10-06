import esa_snappy as snappy
import numpy
import cv2
import matplotlib.pyplot as plt


product = snappy.ProductIO.readProduct("subset_1_of_mosaic_msk.dim")
band=product.getBand("Sigma0_VH")
w=band.getRasterWidth()
h=band.getRasterHeight()

data=numpy.zeros((h,w)) #h x w matrix of 0s.
band.readPixels(0,0,w,h,data) 
low, high = 0.0001516640332, 0.0486812710436 #min/max value provided by snap.

clipped = numpy.clip(data, low, high) #sets all values to low as minimum and high as maximum.
img = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

pre_contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
min_area = 5   #adjust depending on expected dot size, removes noise.
contours=[]
for c in pre_contours:
    if cv2.contourArea(c) > min_area:
        contours.append(c)


def get_centroid(c):
    moments = cv2.moments(c)  #dict of weighted pixel intensities
    if moments["m00"] > 0:
        return int(moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"]) #credit geekforgeeks, returns average centre of mass of contours.
    return None


centroids=[]
for c in contours:
    centroids.append(get_centroid(c))

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
        
print("Merged detections:", len(merged_contours))

coords=[[]]
output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
for c in merged_contours:
    min_enclosing_circle=cv2.minEnclosingCircle(c)
    (x, y) = min_enclosing_circle[0]
    coords.append([x,y])
    radius = min_enclosing_circle[1]
    center = (int(x), int(y))
    radius = int(radius*20)
    cv2.circle(output, center, radius, (0, 0, 255))


plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Specs with circles")
plt.show()



#cropped=img[row:row,column:column]
#cv2.cvtcolor(pic,cv2.COLOR_GRAY2RGB)
#cv2.circle(pic,center,radius,(colour))
#pic.shape[1] (width) pic.shape[0](height)
#contours = cv2.findContours(pic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) cv2.RETR_LIST for all, cv2.RETR_EXTERNAL for all cv2.RETR_TREE not needed, could also use CHAIN_APPROX_NONE
#cv2.threshold(pic, min, max, cv2.THRESH_BINARY) anything below min or above max set to black, else set to white.