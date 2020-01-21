import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt
#from hough import hough_lines
from hough import hough_lines_, hough_lines_2


def draw_line(img, line, color=(255, 0, 0)):

    h, w = img.shape[0], img.shape[1]
    rho = line[0]
    theta = line[1]

    sinT = np.sin(theta)
    cosT = np.cos(theta)

    if abs(sinT) < 0.5:
        # vertical ish line
        y1 = 0
        x1 = int((rho - y1 * sinT) / cosT)
        y2 = h
        x2 = int((rho - y2 * sinT) / cosT)
    else:
        # horizontal ish line
        x1 = 0
        y1 = int((rho - x1 * cosT) / sinT)
        x2 = w
        y2 = int((rho - x2 * cosT) / sinT)

    cv2.line(img, (x1, y1), (x2, y2), color, 2)


def draw_lines(img, lines, color=(255, 0, 0)):


    for line in lines:

        draw_line(img, line, color)



def degrees_to_radians(angle):
    """
    """
    return (angle * np.pi / 180.0)

def radians_to_degrees(angle):
    """
    """
    return (angle * 180.0 / np.pi)

def find_lines(edges, angle, tolerance, min_votes):

    rho_resolution = 1
    theta_resolution = 0.5 * (np.pi / 180)
    #theta_resolution = (np.pi / 180)
    #theta_resolution = np.pi / 180 * 2
    #min_votes = 30

    min_theta = degrees_to_radians(angle - tolerance)
    max_theta = degrees_to_radians(angle + tolerance)

    # lines = cv2.HoughLines(
    #     edges,
    #     rho_resolution,
    #     theta_resolution,
    #     threshold=min_votes
    # )

    lines = cv2.HoughLines(
        edges,
        rho_resolution,
        theta_resolution,
        threshold=min_votes,
        min_theta=min_theta,
        max_theta=max_theta
    )
    lines = np.reshape(lines, (-1, 2))

    return lines


def filter_outliers(lines):

    thetas = lines[:, 1]
    mu = np.median(thetas)
    var = np.var(thetas)
    thresh = 0.5 * var
    lines = [line for line in lines if abs(line[1] - mu) <= thresh]
    return lines, mu

def find_staves(img, lines):

    w, h = img.shape[1], img.shape[0]

    lines = np.sort(lines, axis=0)
    rhos = lines[:, 0]

    # sort lines by rho
    #lines = sorted(lines, key=lambda line: line[0])
    #rhos = lines[:, 0]
    #rhos.sort()
    seq0 = np.insert(rhos, 0, 0)
    seq1 = np.insert(rhos, len(rhos), h)

    diff = seq1 - seq0

    #n_bins = int(h/10)
    #n_bins = h
    #bins = range(0, int(h/2), 3)
    bins = range(0, h)
    hist, bins = np.histogram(diff, bins=bins)

    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # plt.bar(center, hist, align='center', width=width)
    # plt.show()

    max_bin = np.argmax(hist)
    max = bins[max_bin]

    staves = []
    staff = []
    for i in range(len(lines) - 1):
        line1 = lines[i]
        line2 = lines[i+1]
        rho1 = line1[0]
        rho2 = line2[0]
        rhoDiff = np.abs(rho2 - rho1)
        if (rhoDiff - max) < 2:
            if len(staff) < 3:
                staff.append(line1)
            else:
                staff.append(line1)
                staff.append(line2)
                staves.append(staff)
                staff = []
        else:
            staff = []



    return staves, max



def rectify(img):

    w, h = img.shape[1], img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) negative threshold of image
    _, edges = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2) finds horizontal lines
    hough_thresh = int(0.5 * w)
    h_lines = find_lines(edges, angle=90, tolerance=5, min_votes=hough_thresh)

    # 3) obtains page rotation (mu)
    _, mu = filter_outliers(h_lines)

    # 4) de-rotates image
    angle = radians_to_degrees(mu)
    angle = (90 - angle)
    R = cv2.getRotationMatrix2D((0, 0), -angle, 1.0)
    dsize = (img.shape[1], img.shape[0])
    rot = cv2.warpAffine(img, R, dsize, borderValue=(255,255,255))

    return rot


def map_line_func(img, line, func):
    w, h = img.shape[1], img.shape[0]
    rho = line[0]
    theta = line[1]
    sinT = np.sin(theta)
    cosT = np.cos(theta)
    # horizontal ish line
    y1 = int((rho - 0 * cosT) / sinT)
    y2 = int((rho - w * cosT) / sinT)
    return func(y1, y2)


def process_staves(img, staves):

    #i = 0
    #vis = img.copy()
    #for staff in staves:
    #    if i % 2 == 0:
    #        color = (255, 0, 0)
    #    else:
    #        color = (0, 0, 255)
    #    draw_lines(vis, staff, color=color)
    #    i+= 1
    # cv2.imshow('staff', vis)
    w, h = img.shape[1], img.shape[0]

    for staff in staves:
        top = staff[0]
        bottom = staff[-1]

        y_from = map_line_func(img, top, min)
        y_to = map_line_func(img, bottom, max)


        sub = img[y_from:y_to, 0:w]
        sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

        hough_thresh = int(0.97 * sub.shape[0])
        _, edges = cv2.threshold(sub, 200, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('sub', edges)
        #cv2.waitKey()
        v_lines = find_lines(edges, angle=0, tolerance=1, min_votes=hough_thresh)

        vis = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
        draw_lines(vis, v_lines, color=(0, 0, 255))
        cv2.imshow('vlines', vis)
        cv2.imshow('sub', sub)
        cv2.waitKey()

def process(file):

    img = cv2.imread(file)

    print("file: ", file, " shape:", img.shape)
    #img = imutils.resize(img, width=1024)

    # 1) Rectifies the music sheet
    img = rectify(img)
    #cv2.imshow('rect', img)
    #cv2.waitKey()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray,50,150,apertureSize = 3)
    _, edges = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    w, h = img.shape[1], img.shape[0]
    hough_thresh = int(0.5 * w)
    h_lines = find_lines(edges, angle=90, tolerance=1, min_votes=hough_thresh)
    h_lines, mu = filter_outliers(h_lines)
    staves, note_si = find_staves(img, h_lines)
    print("note size:", note_si)
    process_staves(img, staves)

    vis = img.copy()
    draw_lines(vis, h_lines)
    cv2.imshow('filtered', vis)
    cv2.waitKey()



    # plt.imshow(img)
    # plt.xticks([]), plt.yticks([])
    # plt.show()


def test():

    base_path = 'samples/'

    files = [f for f in glob.glob(base_path + "/*.jpg", recursive=False)]
    for file in files:
        process(file)
        cv2.waitKey()

    #file = 'samples/cotton_fields_0.jpg'
    #file = 'samples/cotton_fields_1.jpg'
    #file = 'samples/cotton_fields_2.jpg'
    #file = 'samples/capture.jpg'
    file = 'samples/torcida.jpg'
    #file = 'samples/capture2.jpg'
    #file = 'samples/capture3.jpg'

    process(file)
    cv2.waitKey()




# def test2():
#     file = 'samples/cotton_fields_0.jpg'
#     # file = 'samples/capture.jpg'
#     #file = 'samples/torcida.jpg'
#     # file = 'samples/capture2.jpg'
#     # file = 'samples/capture3.jpg'
#     img = cv2.imread(file)
#
#     print("file: ", file, " shape:", img.shape)
#     #w, h = img.shape[1], img.shape[0]
#     #new_w = 320
#     #new_h = int(h * (new_w / w))
#     #img = cv2.resize(img, (new_w, new_h))
#
#     w, h = img.shape[1], img.shape[0]
#
#     # Pipeline Idea
#     # 0. Gray, Thresholding (img -> edges)   
#     #    0.0 test using vertical Sobel?
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # edges = cv2.Canny(gray,50,150,apertureSize = 3)
#     _, edges = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#
#     cv2.imshow('edges', edges)
#
#     # 1. Hough -> lines     
#     #     1.1 adaptive thresh
#
#     #hough_thresh = int(w / 2.0)
#     # lines = hough_lines(img)
#     #lines = cv2.HoughLines(edges, rho=1, theta=0.5 * np.pi / 180, threshold=hough_thresh)
#     lines = cv2.HoughLinesP(edges, rho=1, theta=0.5 * np.pi/180, threshold=500, minLineLength=1, maxLineGap=2)
#     #h = hough_lines_(edges)
#     #h = np.array(255 * h / np.max(h), dtype='uint8')
#
#     #h = h.T
#     #res = cv2.resize(h, (1920, 480))
#     #cv2.imshow('spectrum', res)
#     #cv2.waitKey()
#     plt.imshow(h)
#     plt.xticks([]), plt.yticks([])
#     plt.show()

if __name__ == "__main__":
    test()