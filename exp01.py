# Barcodes experiment

import cv2 as cv
import numpy as np
import matplotlib.pyplot as pl

def image_edges(im):
    """
    Convert the image to grey scale, clean, and prepare it for contour-finding.
    Current method is just to threshold and then morph-open to separate features.
    """
    #kernel = np.ones((5,5),np.uint8) # structuring element for morphology.

    grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    work_sample = cv.threshold(grey, 200, 255, cv.THRESH_BINARY)[1]
    # The threshold value was selected by trial and error.
    # A future version should probably do more elaborate segmentation.
    
    return work_sample

def find_long_contours(edges_image):
    """
    Turn an image of edges (by derivative/Canny) to a list of rectangles
    fitted to the contour. Only long rectangles (aspect >= given aspect) 
    are taken, and angles are edited to make height always the larger quantity.
    """
    contours, _ = cv.findContours(edges_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    filtered_rects = []
    for cnt in contours:
        # Fit a box to it. Determine if part of barcode.
        rect = cv.minAreaRect(cnt)
        width = rect[1][0]
        height = rect[1][1]
        
        # Ensure that height is always the larger quantity.
        if width > height:
            width, height = height, width
            angle = rect[2] + 90 # angle. It's the same rect after the width-height change.
            rect = (rect[0], (width, height), angle)
            
        if width <= 0 or height/width < 5:
            continue
        
        filtered_rects.append(rect)
    
    return filtered_rects

def find_long_contours_cluster(long_boxes):
    """
    Finds among the long contours a cluster that's supposed to represent the barcode.
    """
    cluster_boxes = []
    heights = np.array([r[1][1] for r in filtered_rects])
    angles = np.array([r[2] for r in filtered_rects])
    counts, xedges, yedges = np.histogram2d(heights, angles)
    
    print(counts)
    print(yedges)
    
    # 30 bars in a barcode, but some images get more...
    # https://brokensecrets.com/2010/04/30/every-upc-barcode-has-30-bars/
    bar_cluster = np.nonzero((counts >= 30) & (counts < 32))
    
    min_height = xedges[bar_cluster[0][0]]
    max_height = xedges[bar_cluster[0][0] + 1]
    min_ang = yedges[bar_cluster[1][0]]
    max_ang = yedges[bar_cluster[1][0] + 1]

    for rect in filtered_rects:
        #print(rect)
        #print(min_height)
        
        height = rect[1][1]
        ang = rect[2]
        if not (min_height <= height <= max_height and min_ang <= ang <= max_ang):
            continue
        
        cluster_boxes.append(rect)
    
    return cluster_boxes

def plot_boxes(box_list):
    for box in box_list:
        rect = cv.boxPoints(box)
        rect = np.vstack((rect, rect[0]))
        pl.plot(rect[:,0], rect[:,1], 'r')

if __name__ == "__main__":
    for im_name in ['2021-04-08_19-42-00.jpg', '2021-04-08_19-42-42.jpg'] :
        pl.figure()
        
        im = cv.imread(im_name)
        # pl.imshow(im[...,::-1]) # OpenCV is BGR
        
        edges = image_edges(im)
        pl.imshow(edges, 'gray')
        
        filtered_rects = find_long_contours(edges)
        cluster_boxes = find_long_contours_cluster(filtered_rects)
        #print(cluster_boxes)
        plot_boxes(cluster_boxes)
        
    pl.show()
