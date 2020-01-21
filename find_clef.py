import cv2
import imutils
import numpy as np
#from matplotlib import pyplot as plt



template = cv2.imread('res/bass_clef.jpg')
#template = cv2.imread('res/treble_clef.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


#image = cv2.imread('samples/capture.jpg')
#image = cv2.imread('samples/capture2.jpg')
#image = cv2.imread('samples/capture3.jpg')
#image = cv2.imread('samples/torcida.jpg')
image = cv2.imread('samples/cotton_fields_0.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

found = None
visualize = False

# detect edges in the resized, grayscale image and apply template
# matching to find the template in the image
#edged = cv2.Canny(gray, 50, 200)
#cv2.imshow('edged', edged)

# loop over the scales of the image
scales = np.linspace(0.1, 1.0, 20)[::-1]
for scale in scales:

    scaled = imutils.resize(template, width=int(template.shape[1] * scale))
    scaled_template = cv2.Canny(scaled, 50, 200)
    (tH, tW) = scaled_template.shape[:2]

    # # resize the image according to the scale, and keep track
    # # of the ratio of the resizing
    # #resized = resize(gray, width=int(gray.shape[1] * scale))
    r = template.shape[1] / float(scaled_template.shape[1])
    #
    # # if the resized image is smaller than the template, then break
    # # from the loop
    if scaled_template.shape[0] < tH or gray.shape[1] < tW:
        print("Scale skipped:", scale)
        continue

    #result = cv2.matchTemplate(edged, scaled_template, cv2.TM_CCOEFF)
    result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)

    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    threshold = 0.6
    if maxVal < threshold:
        continue

    loc = np.where(result >= threshold)
    score = np.median(result[loc])

    # check to see if the iteration should be visualized
    if visualize:

        #print("size:", tH, tW, maxVal, maxLoc, r)
        #cv2.imshow('template', scaled_template)

        # draw a bounding box around the detected region
        clone = np.copy(gray)
        #x_offset = y_offset = 0
        #clone[y_offset:y_offset + scaled.shape[0], x_offset:x_offset + scaled.shape[1]] = scaled
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)

        for maxLoc in zip(*loc[::-1]):
            cv2.rectangle(clone, maxLoc, (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)

        #cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
        #              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)

        cv2.imshow("Visualize", clone)
        cv2.waitKey(0)

    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or score > found[0]:
        found = (score, loc, scale)

# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, loc, scale) = found
print(found)
#startX = maxLoc[0]
#startY = maxLoc[1]
#endX = maxLoc[0] + int(template.shape[1] * scale)
#endY = maxLoc[1] + int(template.shape[0] * scale)

# draw a bounding box around the detected result and display the image
#cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

tW = int(template.shape[1] * scale)
tH = int(template.shape[0] * scale)
for maxLoc in zip(*loc[::-1]):
    cv2.rectangle(image, maxLoc, (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.resizeWindow("Image", 1024, 768)
cv2.waitKey(0)

#template = cv2.resize(template, (256, 437))
#template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#w, h = template.shape[::-1]
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold )
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

#cv2.imshow('template', template)
#cv2.imshow('res', img_rgb)
#cv2.waitKey()
