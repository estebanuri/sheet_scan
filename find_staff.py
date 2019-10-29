import cv2
import numpy as np

img = cv2.imread('samples/capture.jpg')
#img = cv2.imread('samples/torcida.jpg')
#img = cv2.imread('samples/capture2.jpg')
img = cv2.imread('samples/capture3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
_, edges = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

#cv2.imshow('edges', edges)
#cv2.waitKey()


lines = cv2.HoughLines(edges, rho=1, theta=0.5 * np.pi/180, threshold=300)
#lines = cv2.HoughLinesP(edges, rho=1, theta=0.5 * np.pi/180, threshold=500, minLineLength=1, maxLineGap=2)

angles = []

# horizontal lines
angles.append(np.pi / 2)

# vertical
angles.append(0)

for line in lines:

    rho = line[0][0]
    theta = line[0][1]

    eps = 0.01
    #shift = -0.01
    #shift = 0.015
    #shift = 0.01
    shift = 0.0

    #if not (abs(theta - np.pi / 2) < eps):
    #if not (abs(theta - 0) < eps):

    ok = False
    for angle in angles:
        if (abs(theta - (shift + angle)) < eps):
            ok = True

    if not ok:
        continue

    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)

cv2.imshow('results', img)
cv2.waitKey()

# cv2.imwrite('houghlines3.jpg',img)