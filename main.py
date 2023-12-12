# import the necessary packages
import argparse
# test

# import imutils
import cv2
import numpy as np
# from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

# vars
imheigth = 600
Test = False


# functions
def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
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
ap.add_argument("-o", "--output", required=True,
                help="path to the output image")
ap.add_argument("-d", "--debug", type=bool, nargs='?', const=True, default=False,
                help="Enable debug mode. Show all images")
ap.add_argument("-b", "--barcode", required=True,
                help="Barcode of SKU on image")
args = vars(ap.parse_args())

if args["debug"]:
    Test = True


# load the image, convert it to grayscale, and blur it slightly
orig = cv2.imread(args["image"])
# resized
resized = ResizeWithAspectRatio(orig, height=1000)

(h_o, w_o) = orig.shape[:2]
(h_r, w_r) = resized.shape[:2]

k_X = w_o / w_r
k_Y = h_o / h_r

if Test:
    TestShow(resized, "Gray", imheigth)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
if Test:
    TestShow(gray, "Gray", imheigth)

gray = cv2.GaussianBlur(gray, (5, 5), 0)
if Test:
    TestShow(gray, "Gaus", imheigth)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 10, 100, L2gradient=False)
if Test:
    TestShow(edged, "Edged", imheigth)

edged = cv2.dilate(edged, None, iterations=2)
if Test:
    TestShow(edged, "dilate", imheigth)

edged = cv2.erode(edged, None, iterations=2)
if Test:
    TestShow(edged, "Erode", imheigth)

# find contours in the edge map
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find maximum contour
max_s = 0
max_cnt = None
# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) > 1000:
        if c.shape[0] > max_s:
            max_cnt = c
            max_s = c.shape[0]

box = cv2.minAreaRect(max_cnt)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
cv2.drawContours(resized, [box.astype("int")], -1, (0, 255, 0), 2)
if Test:
    TestShow(resized, "Maximum", imheigth)

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
angle = round((180 - (rad * 180 / 3.141592)), 2)
angle = 270 - angle

# compute the Euclidean distance between the midpoints
dY = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dX = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

if dY > dX:
    kaima = int(dX * 0.2)
else:
    kaima = int(dY * 0.2)

center = midpoint((tltrX, tltrY), (blbrX, blbrY))
M = cv2.getRotationMatrix2D(center, angle, 1.0)
resized = cv2.warpAffine(resized, M, (w_r, h_r))
resized = resized[int(round((center[1] - dY/2 - kaima), 0)):int(round((center[1] + dY/2 + kaima), 0)),
          int(round((center[0] - dX/2 - kaima), 0)):int(round((center[0] + dX/2 + kaima), 0))]
if Test:
    TestShow(resized, "Rotated_resized", imheigth)

center = (center[0] * k_X, center[1] * k_Y)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_orig = cv2.warpAffine(orig, M, (w_o, h_o))
rotated_crop = rotated_orig[int(round((center[1] - (dY/2 + kaima) * k_Y), 0)):int(round((center[1] + (dY/2 + kaima) * k_Y), 0)),
       int(round((center[0] - (dX/2 + kaima) * k_X), 0)):int(round((center[0] + (dX/2 + kaima) * k_X), 0))]
if Test:
    TestShow(rotated_orig, "Rotated_orig", imheigth)
if Test:
    TestShow(rotated_crop, "Rotated_crop", imheigth)

print(int(round(dX * k_X, 0)), int(round(dY * k_Y, 0)))

cv2.imwrite(args["output"], orig)
