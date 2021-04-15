# Barcodes experiment
#
# References:
# [1] https://brokensecrets.com/2010/04/30/every-upc-barcode-has-30-bars/
# [2] https://courses.cs.washington.edu/courses/cse370/01au/minirproject/TeamUPC/UPC.html
# [3] https://en.wikipedia.org/wiki/Universal_Product_Code

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as pl
from sklearn.cluster import DBSCAN

def image_edges(im):
    """
    Convert the image to grey scale, clean, and prepare it for contour-finding.
    Current method is just to threshold and then morph-open to separate features.
    """
    #kernel = np.ones((5,5),np.uint8) # structuring element for morphology.

    grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    work_sample = cv.threshold(grey, 200, 255, cv.THRESH_BINARY_INV)[1]
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
    
    Arguments:
    long_boxes - a list of boxes (as represented by OpenCV box fitting).
    
    Returns:
    a list of boxes that belong to a 30-boxes tight cluster in height/angle plane.
    """
    cluster_boxes = []
    heights = np.array([r[1][1] for r in long_boxes])
    angles = np.array([r[2] for r in long_boxes])
    
    # Rescale to [0,1] for use with DBSCAN:
    heights_scaled = (heights - heights.min())/(heights.max() - heights.min())
    angles_scaled = angles/360
    
    clustering = DBSCAN(eps=0.05, min_samples=3).fit(np.c_[heights_scaled, angles_scaled])
    
    # 30 bars in a barcode [1]
    cluster_ids = np.unique(clustering.labels_)
    counts = [(clustering.labels_ == l).sum() for l in np.unique(clustering.labels_)]
    bars_label = np.nonzero(np.r_[counts] == 30)[0][0] - 1 
    # Becauwse one label is '-1' for unclustered points
    
    for rid, rect in enumerate(long_boxes):
        if clustering.labels_[rid] == bars_label:        
            cluster_boxes.append(rect)
    
    return cluster_boxes

def barcode_run_lengths(boxes):
    """
    Converts a barcode from a series of black boxes to run lengths of
    the black and white parts, alternating. 
    Assumes all boxes are in the same angle bin, so there are no 180-deg.
    differences.
    """
    boxes.sort(key=lambda b: b[0][0]) # For now assume they're kind-of vertical.
    
    run_lengths = []
    for spaceIx in range(len(boxes) - 1):
        interbox_vec = np.r_[boxes[spaceIx + 1]] - np.r_[boxes[spaceIx]]
        space_len = np.linalg.norm(interbox_vec) - (boxes[spaceIx + 1][1][0] + boxes[spaceIx][1][0])/2.
        #print(np.linalg.norm(interbox_vec), boxes[spaceIx][1][0], boxes[spaceIx + 1][1][0], space_len)
        
        
        run_lengths.extend([boxes[spaceIx][1][0], space_len])
    
    run_lengths.append(boxes[-1][1][0])
    return run_lengths
    
def plot_boxes(box_list):
    for box in box_list:
        rect = cv.boxPoints(box)
        rect = np.vstack((rect, rect[0]))
        pl.plot(rect[:,0], rect[:,1], 'r')

if __name__ == "__main__":
    for im_name in ['2021-04-08_19-42-00.jpg', '2021-04-08_19-42-42.jpg']:
        pl.figure()
        
        im = cv.imread(im_name)
        # pl.imshow(im[...,::-1]) # OpenCV is BGR
        
        edges = image_edges(im)
        pl.imshow(edges, 'gray')
        
        filtered_rects = find_long_contours(edges)
        cluster_boxes = find_long_contours_cluster(filtered_rects)
        plot_boxes(cluster_boxes)
        
        rls = barcode_run_lengths(cluster_boxes)
        module_len = np.sum(rls) / 95
        print (rls/module_len)
        
        rl_modules = np.int_(np.round(rls/module_len))
        print(rl_modules, rl_modules.sum())
        
        
    pl.show()
