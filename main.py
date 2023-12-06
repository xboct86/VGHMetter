# import the necessary packages
import argparse

# import imutils
import cv2
import numpy as np
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

imheigth = 900

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def TestShow(image, wname, hsize):
    resized = ResizeWithAspectRatio(image, height=hsize)
    cv2.imshow(wname, resized)
    cv2.waitKey(0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
                help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
image = ResizeWithAspectRatio(image, height=1000)
TestShow(image, "Image", 900)

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
TestShow(gray, "Gray", 900)

gray = cv2.GaussianBlur(gray, (5, 5), 0)
#gray = cv2.GaussianBlur(gray, (15, 15), 3)
TestShow(gray, "Gaus", 900)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 10, 100, L2gradient=False)
TestShow(edged, "Edged", 900)

edged = cv2.dilate(edged, None, iterations=2)
TestShow(edged, "dilate", 900)

edged = cv2.erode(edged, None, iterations=2)
TestShow(edged, "Erode", 900)

# find contours in the edge map
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw finded contours on image
cv2.drawContours(image, cnts, -1, (0,255,255), 3, cv2.LINE_AA)
pixelsPerMetric = None

# Find maximum contour
max=0
max_cnt = None
# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    print(cv2.contourArea(c))
    if cv2.contourArea(c) > 1000:
        if c.shape[0]>max:
            max_cnt=c
            max=c.shape[0]
            print(max)

cv2.drawContours(image, max_cnt, -1, (0,0,255), 3, cv2.LINE_AA)

box = cv2.minAreaRect(max_cnt)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
TestShow(image, "Result", 900)


# order the points in the contour such that they appear
# in top-left, top-right, bottom-right, and bottom-left
# order, then draw the outline of the rotated bounding
# box

box = perspective.order_points(box)
cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 120), 2)
TestShow(image, "Result", 900)

for (x, y) in box:
    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

TestShow(image, "Result", 900)

'''
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)


    rad = np.arctan2((tltrY - blbrY), (tltrX - blbrX))
    angle = 180-(rad * 180/3.141592)
#    if angle > 180:
#    angle = 360 - angle
#    print(rad)
    print(angle)
    angle = 270 - angle
    print(angle)

    line = line+1
    # draw the midpoints on the image
    cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
#    cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
#    cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
#    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)


    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]
        cv2.putText(image, str(pixelsPerMetric), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255),2 )


    # compute the size of the objecttd
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    cv2.putText(image, str(angle), (50, (70 + (line * 20))), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # draw the object sizes on the image
    cv2.putText(image, "{:.1f}mm".format(dimA),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
    cv2.putText(image, "{:.1f}mm".format(dimB),
        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)

    cv2.imshow("size", image)
    cv2.waitKey(0)


    (h, w) = image.shape[:2]
    center = (h / 2, w / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (h, w))
    cv2.imshow("rotated", image)
    cv2.waitKey(0)


    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
'''