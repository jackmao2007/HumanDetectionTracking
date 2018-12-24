# Helper libraries
import numpy as np
from os import listdir
import cv2

paths = ['googleData/cropped_stand',
         'googleData/cropped_walk',
         'googleData/cropped_run']
i = 1
for path in paths:
    for files in listdir(path):
        if files[-4:] == '.png':
            img = cv2.imread(path + '/' + files)
            print(img.shape)
            resize = cv2.resize(img, (64, 128))
            flip = im = np.fliplr(resize)
            cv2.imwrite(path + '/processed/' + str(i) + 'a.png', resize)
            cv2.imwrite(path + '/processed/' + str(i) + 'b.png', flip)
            i += 1
