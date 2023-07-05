# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Transformation.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: wluong <wluong@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/07/05 14:13:14 by wluong            #+#    #+#              #
#    Updated: 2023/07/05 15:56:37 by wluong           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import sys
from plantcv import plantcv as pcv

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    gray = pcv.rgb2gray(rgb_img=img)
    gaussian_img = pcv.gaussian_blur(img=gray, ksize=(7, 7), sigma_x=0, sigma_y=None)
    threshold_img = cv2.threshold(gaussian_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    masked_img = pcv.apply_mask(img=img, mask=threshold_img, mask_color='white')
    # cv2.imshow("img",threshold_img)
    cv2.imshow("img",masked_img)
    cv2.waitKey(0)