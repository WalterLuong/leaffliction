# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Transformation.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: wluong <wluong@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/07/05 14:13:14 by wluong            #+#    #+#              #
#    Updated: 2023/07/06 11:04:51 by wluong           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import sys
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import numpy as np

class Options:
    def __init__(self):
        self.image = "./img/original_image.jpg"
        self.debug = "plot"
        self.writeimg= False
        self.result = "vis_tutorial_results.json"
        self.outdir = "." # Store the output to the current directory

if __name__ == "__main__":

    pcv.params.debug = 'plot'
    img, path, filename = pcv.readimage(filename=sys.argv[1])
    pcv.params.debug = None

    #Image to grayscale then binary threshold
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, max_value=255, object_type='light')
    
    #Medium and Gaussian blur
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    pcv.params.debug = 'plot'
    gaussian_img = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)
    pcv.params.debug = None
  
    #Better threshold for the mask    
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    b_thresh = pcv.threshold.binary(gray_img=b, threshold=160, max_value=255, object_type='light')
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
    masked = pcv.apply_mask(img=img, mask=bs, mask_color='white')
    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')
    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=115, max_value=255, object_type='dark')
    maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135, max_value=255, object_type='light')
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128, max_value=255, object_type='light')
    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
    opened_ab = pcv.opening(gray_img=ab)
    ab_fill = pcv.fill(bin_img=ab, size=200)
 
    pcv.params.debug = 'plot'
    masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')
    pcv.params.debug = None
 

    id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)
    roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=0, y=0, h=256, w=256)
    pcv.params.debug = 'plot'
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1, roi_hierarchy=roi_hierarchy, object_contour=id_objects, obj_hierarchy=obj_hierarchy, roi_type='partial')
    pcv.params.debug = None
    
    #Analyzing the image with a mask
    obj, mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy3)
    pcv.params.debug = 'plot'
    analysis_image = pcv.analyze_object(img=img, obj=obj, mask=mask, label="default")
    color_histogram = pcv.analyze_color(rgb_img=img, mask=kept_mask, colorspaces='all', label="default")
    pcv.print_image(img=color_histogram, filename="vis_tutorial_color_hist.jpg")
    top_x, bottom_x, center_v_x = pcv.x_axis_pseudolandmarks(img=img, obj=obj, mask=mask, label="default")
    top_y, bottom_y, center_v_y = pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask)