# Barcodes experiment
#
# References:
# [1] https://brokensecrets.com/2010/04/30/every-upc-barcode-has-30-bars/
# [2] https://courses.cs.washington.edu/courses/cse370/01au/minirproject/TeamUPC/UPC.html
# [3] https://en.wikipedia.org/wiki/Universal_Product_Code
# [4] https://en.wikipedia.org/wiki/International_Article_Number
# [5] http://www.danacode.com/danacode.htm

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as pl

from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

# See [2]
upc_codes_key = {
    (3, 2, 1, 1): 0,
    (2, 2, 2, 1): 1,
    (2, 1, 2, 2): 2, 
    (1, 4, 1, 1): 3,
    (1, 1, 3, 2): 4,
    (1, 2, 3, 1): 5,
    (1, 1, 1, 4): 6,
    (1, 3, 1, 2): 7,
    (1, 2, 1, 3): 8,
    (3, 1, 1, 2): 9,
}

# How the 5 digits after the 1st are encoded in EAN-13 [4]:
ean_parity_patterns = [
    'LLLLLL', 'LLGLGG', 'LLGGLG', 'LLGGGL', 'LGLLGG', 
    'LGGLLG', 'LGGGLL', 'LGLGLG', 'LGLGGL', 'LGGLGL'
]

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
            
        if width <= 0 or height/width < 4.5:
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
    
    clustering = DBSCAN(eps=0.15, min_samples=3).fit(np.c_[heights_scaled, angles_scaled])
    
    # 30 bars in a barcode [1]
    cluster_ids = np.unique(clustering.labels_)
    #pl.figure()
    #pl.scatter(heights, angles, c=clustering.labels_)
    #pl.show()
    counts = [(clustering.labels_ == l).sum() for l in cluster_ids]
    bars_label = np.nonzero((np.r_[counts] == 30) | (np.r_[counts] == 46))[0][0]
    
    # Becauwse one label can be '-1' for unclustered points:
    uncluster_adj = 0
    if cluster_ids[0] == -1:
        bars_label -= 1
        uncluster_adj = 1
        
    for rid, rect in enumerate(long_boxes):
        if clustering.labels_[rid] == bars_label:        
            cluster_boxes.append(rect)
    
    # If the cluster is 46-long, it means a UPC-E barcode (6 digits, 16 bars) 
    # is attached to the UPC-A barcode. Get rid of it for now:
    if counts[bars_label + uncluster_adj] == 46:
        cluster_boxes.sort(key=lambda b: b[0][0]) # For now assume they're kind-of vertical.
    
    return cluster_boxes[:30]

def barcode_run_lengths(boxes):
    """
    Converts a barcode from a series of black boxes to run lengths of
    the black and white parts, alternating. 
    Assumes all boxes are in the same angle bin, so there are no 180-deg.
    differences.
    """
    boxes.sort(key=lambda b: b[0][0]) # For now assume they're kind-of vertical.
    
    # Find th common line they're on. We can't naively use box centers
    # because the limit bars can be longer than the digit bars.
    lin_reg = LinearRegression()
    centers = np.array([b[0] for b in boxes])
    lin_reg.fit(centers[:,0,None], centers[:,1,None])
    
    line_ang = np.arctan(lin_reg.coef_[0][0])
    line_direct = np.r_[np.cos(line_ang), np.sin(line_ang)]
    
    run_lengths = []
    for spaceIx in range(len(boxes) - 1):
        interbox_vec = centers[spaceIx + 1] - centers[spaceIx]
        interbox_vec = line_direct*np.dot(interbox_vec, line_direct)
        space_len = np.linalg.norm(interbox_vec) - (boxes[spaceIx + 1][1][0] + boxes[spaceIx][1][0])/2.
        
        run_lengths.extend([boxes[spaceIx][1][0], space_len])
    
    run_lengths.append(boxes[-1][1][0])
    return run_lengths

def ean13_to_digits(bar_widths):
    """
    See [4].
    
    Receives a list of standardized bar widths (thinnest = 1, thickest = 4),
    including delimiter bars (1-1-1 at both ends and 1-1-1-1-1 in middle).
    Returns the digits corresponding to non-delimiter bars, or throws exception if 
    the format is wrong.
    
    Arguments:
    bar_widths - 59-length NumPy array of ints. Alternating black-white standardizd
        bar width.
    
    Returns:
    13-length array of digits.
    """
    assert(np.all(bar_widths[:3]) == 1) # left delim
    assert(np.all(bar_widths[-3:]) == 1) # right delim
    assert(np.all(bar_widths[27:32]) == 1) # middle delimiter.
    
    digit_codes = np.r_[bar_widths[3:27], bar_widths[32:-3]].reshape(-1,4)
    digits = np.empty(13, dtype=np.int8)
    
    # First digit determined byparity pattern:
    parities = ''.join(['G' if (code[1] + code[3]) % 2 == 0 else 'L' for code in digit_codes])
    digits[0] = ean_parity_patterns.index(parities[:6])
    
    for code_ix, code in enumerate(digit_codes):
        if parities[code_ix] == 'G':
            code = code[::-1]
        digits[code_ix + 1] = upc_codes_key[tuple(code)]
        
    return digits
    
def plot_boxes(box_list):
    for box in box_list:
        rect = cv.boxPoints(box)
        rect = np.vstack((rect, rect[0]))
        pl.plot(rect[:,0], rect[:,1], 'r')

if __name__ == "__main__":
    img_names = ['2021-04-08_19-42-00.jpg', '2021-04-08_19-42-22.jpg', '2021-04-08_19-42-42.jpg']
    for im_name in img_names:
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
        rl_modules = np.int_(np.round(rls/module_len))
        
        digits = ean13_to_digits(rl_modules)        
        pl.title(''.join([str(d) for d in digits]))
        
    pl.show()
