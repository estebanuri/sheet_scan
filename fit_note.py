import cv2
import numpy as np
#from cv2 import ellipse as cvellipse
from find_staff import rectify


def pix(np_array):
    return tuple(np.round(np_array).astype(int))



def fit_note(note_h):

    canvas_h = 2 * note_h
    canvas_w = 3 * note_h

    p = 0.63
    angle = 65
    theta = angle * np.pi / 180.0

    pos = np.array((canvas_w/2.0, canvas_h/2.0))

    min_b = note_h
    max_b = 2 * note_h
    itt = 0
    debug = False
    ret = None
    while itt < 100:

        curr_h = (min_b + max_b) / 2.0

        ma = curr_h
        MA = curr_h * p
        ellipse = (pos, (MA, ma), angle)

        img = np.zeros((canvas_h, canvas_w), dtype='uint8')
        cv2.ellipse(img, ellipse, color=(255), thickness=-1)
        cnts, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        note_contour = max(cnts, key=lambda c: cv2.contourArea(c))

        rect = cv2.boundingRect(note_contour)
        bx, by, bw, bh = rect

        if debug:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, [note_contour], -1, (255, 0, 255), 8)

            x, y = pos
            pt1 = np.array((x, y - note_h/2.0))
            pt2 = np.array((x, y + note_h/2.0))
            cv2.line(vis, pix(pt1), pix(pt2), color=(255, 255, 0), thickness=8)
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color=(0,255,0), thickness=1)
            cv2.imshow('note', vis)
            k = cv2.waitKey()
            print(note_h, bh, curr_h, itt)

        if note_h == bh:
            ret = img[by:by + bh, bx:bx + bw]
            break
        elif note_h < bh:
            max_b = curr_h
        else:
            min_b = curr_h

        itt += 1

    # img = np.zeros((bh, bw), dtype='uint8')
    # pos = np.array((bw/2.0, bh/2.0))
    # ma = curr_h
    # MA = curr_h * p
    # ellipse = (pos, (MA, ma), angle)
    # cv2.ellipse(img, ellipse, color=(255), thickness=-1)

    return ret


def draw_note(img, pos, h):

    #p = 0.63
    #angle = 65
    p = 0.2
    angle = 80

    theta = angle * np.pi / 180.0

    use_h = h * 0.95
    h2 = use_h * use_h
    p2 = p * p
    sin = np.sin(theta)
    sin2 = sin*sin
    d = (sin2 + 1) * p2 + 1
    b = 2.0 * np.sqrt(h2 / d)
    a = p * b
    MA = a
    ma = b

    ellipse = (pos, (MA, ma), angle)

    cv2.ellipse(img, ellipse, color=(255, 0, 255), thickness=1)

    x, y = pos
    pt1 = np.array((x, y - h/2.0))
    pt2 = np.array((x, y + h/2.0))
    cv2.line(vis, pix(pt1), pix(pt2),color = (255, 255, 0), thickness=1)
    cv2.line(vis, (0, 0), (0, h), color = (255, 255, 0), thickness=1)


def fit_from_img():

    img = cv2.imread('res/note.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ret = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    cnts, h = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    note_contour = max(cnts, key=lambda c: cv2.contourArea(c))

    vis = img.copy()
    cv2.drawContours(vis, [note_contour], -1, (255, 0, 255), 1)
    #(x, y), (MA, ma), angle = cv2.fitEllipse(note_contour)
    ellipse = cv2.fitEllipse(note_contour)
    (x, y), (MA, ma), angle = ellipse
    print("x, y", x, y)
    print("M, m", MA, ma)
    print("angle", angle)
    print("prop", MA / ma)


    #center = (int(x), int(y))
    #axes = (int(MA), int(ma))
    #cvellipse(vis, center, axes, angle, startAngle=0, endAngle=360, color=(0, 255, 0))
    cv2.ellipse(vis, ellipse, thickness=1, color=(0, 255, 0))

    theta = angle * np.pi / 180.0
    #a = MA
    #b = ma
    a = MA
    b = ma
    a2 = a*a
    b2 = b*b
    cos2 = np.cos(theta) * np.cos(theta)
    sin2 = np.sin(theta) * np.sin(theta)

    eH = np.sqrt(2 * (a2 + b2 - (a2 - b2) * cos2))/2.0
    eh = np.sqrt(2 * (a2 + b2 - (b2 - a2) * sin2))/2.0



    #pt1 = np.array((x - h/2.0, y))
    #pt2 = np.array((x + h/2.0, y))
    pt1 = np.array((x, y - eh/2.0))
    pt2 = np.array((x, y + eh/2.0))
    cv2.line(vis, pix(pt1), pix(pt2),color = (0, 255, 255), thickness=2)
    pt3 = np.array((x - eH/2.0, y))
    pt4 = np.array((x + eH/2.0, y))
    cv2.line(vis, pix(pt3), pix(pt4),color = (255, 255, 0), thickness=2)


    w, h = img.shape[1], img.shape[0]
    note_pos = (w/2.0, h/2.0)
    note_h = 88
    draw_note(vis, note_pos, note_h)

    cv2.namedWindow('note', cv2.WINDOW_NORMAL)
    cv2.imshow('note', vis)
    cv2.waitKey()

    #vis = img.copy()
    #cv2.drawContours(vis, cnts, -1, (0, 255, 0), -1)
    #cv2.imshow('note', vis)
    #cv2.waitKey()

    #vis = img.copy()
    #for i in range(len(cnts)):


    #    cv2.drawContours(vis, cnts, i, (0, 255, 0), -1)

        #(x, y), (MA, ma), angle = cv2.fitEllipse(cnts[i])
        #center = (int(x), int(y))
        #axes = (int(MA), int(ma))
        #cvellipse(vis, center, axes, angle, startAngle=0, endAngle=360, color=(0, 255, 0))

    #    cv2.imshow('note', vis)
    #    cv2.waitKey()


img, ns = cv2.imread('samples/cotton_fields_0.jpg'), 13
#img, ns = cv2.imread('samples/torcida.jpg'), 9

img = rectify(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, edges = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

note_img= fit_note(ns)
note_w, note_h = note_img.shape[1], note_img.shape[0]

#filtered = cv2.filter2D(edges, -1, note_img)
#cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(edges, note_img, cv2.TM_CCOEFF_NORMED)
max = np.max(result)
poss = np.argwhere(result >= 0.75 * max)

vis = img.copy()
for pos in poss:
    note_pos = np.array((pos[1] + note_w/2.0, pos[0] + note_h/2.0))
    cv2.circle(vis, pix(note_pos), 3, (0,255,0), 1)

cv2.namedWindow("note", cv2.WINDOW_NORMAL)
cv2.imshow("note", note_img)
cv2.imshow("image", edges)
cv2.imshow("filt", result)
cv2.imshow("found", vis)
cv2.waitKey()