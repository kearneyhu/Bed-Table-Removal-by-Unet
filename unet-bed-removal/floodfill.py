import cv2
import numpy as np
import os
'''
Instruction:
For binary image, 255 is white target, 0 is black background,
this function is meant to fill the blackhole inside the white target
'''
in_path = 'data/training/mask/'
out_path = 'data/training/full_mask/'
def fillHole(im_in):
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255)

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv
	return im_out

for fn in os.listdir(in_path):
	image = cv2.imread(in_path+fn, cv2.IMREAD_GRAYSCALE)
	filled = fillHole(image)
	cv2.imwrite(out_path+fn, filled)
