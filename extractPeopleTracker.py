# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from google.colab import drive
import random
from skimage import data
from skimage.transform import pyramid_gaussian
import time

# drive.mount('/content/gdrive')
gdrive = ''  # "/content/gdrive/My Drive/ProjectHumanDetection/"
runpath = 'googleData/download_run'
walkpath = 'googleData/download_walk'
standpath = 'googleData/download_stand'


def get_hog_descriptors(images):
    """ gets the hog descriptors for each of the images, and returns a
      ndarray containing all the descriptors
      """
    img_count = images.shape[0]
    hog = cv2.HOGDescriptor()
    results = np.zeros((img_count, 3780, 1))
    for i in range(img_count):
        results[i, :, :] = hog.compute(images[i, :, :].astype(np.uint8))
    return results


def non_max_supression(boxes, thres=0.5):
    """ input is a list of boxes with detection scores
    list(list(y_start, y_end, x_start, x_end, prediction_score))
    return a list of boxes with overlaping supressed
    """
    def take_score(lst):
        return lst[4]
    top_score = sorted(boxes, key=take_score)
    good_boxes = []
    while top_score != []:
        # get the top score box
        good = top_score.pop()
        good_area = (good[1] - good[0]) * (good[3] - good[2])
        i = len(top_score) - 1
        while i >= 0:
            check = top_score[i]
            check_area = (check[1] - check[0]) * (check[3] - check[2])
            xx1 = max(good[2], check[2])
            yy1 = max(good[0], check[0])
            xx2 = min(good[3], check[3])
            yy2 = min(good[1], check[1])
            overlap = max(0, (yy2 - yy1)) * max(0, (xx2 - xx1))
            if overlap / (good_area + check_area - overlap) > thres:
                top_score.pop(i)
            i -= 1
        good_boxes.append(good)
    return good_boxes


def find_detections(img, human_model, pose_model, image_scaling):
    """ draw_the detection boxes on the img, detect using human_model and
    pose_model. Use sliding box on image pyrimid with total of 8 layers
    with image_scaling
    """
    box_l = 128
    box_w = 64
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    l, w = gray.shape
    # detection boxes
    people = []
    # scalling of the image pyrimid
    scalor = image_scaling
    # get one bigger layer to detect smaller people
    big = cv2.resize(gray, (int(w * scalor), int(l * scalor)))
    pyrimid = [big] + list(pyramid_gaussian(gray, max_layer=6, downscale=scalor))
    # sliding box
    for i in range(len(pyrimid)):
        y_max, x_max = pyrimid[i].shape
        normalize = scalor ** (i - 1)
        y = 0
        while y < y_max - box_l:
            x = 0
            while x < x_max - box_w:
                sample = pyrimid[i][y:y + box_l, x:x + box_w]
                hog_des = get_hog_descriptors(np.array([sample * 255]))
                predi = human_model.predict(hog_des)[0, 1]
                # threshold for detection model
                if predi > 0.75:
                    # detection box: [y_start, y_end, x_start, x_end, prediction_score]
                    box = [int(x * normalize) for x in (y, y + box_l, x, x + box_w)] + [predi]
                    people.append(box)
                # shift 12 pixels
                x += 12
            y += 12

    good_boxes = non_max_supression(people, 0.45)
    clas = {0: 'stand', 1: 'walk', 2: 'run'}

    for box in good_boxes:
        crop = cv2.resize(gray[box[0]:box[1], box[2]:box[3]], (64, 128))
        hogg = get_hog_descriptors(np.array([crop]))
        predi2 = pose_model.predict(hogg)[0].tolist()
        pose = clas[predi2.index(max(predi2))]
        # detection box: [y_start, y_end, x_start, x_end, prediction_score, pose]
        box.append(pose)
    return good_boxes


def draw_detections(img, good_boxes):
    """ draw the detections onto the img with the good_boxes returned
    by find_detections
    """
    for box in good_boxes:
        cv2.rectangle(img, (box[2], box[0]),
                      (box[3], box[1]),
                      (0, 0, 255), 1)

        cv2.putText(img, box[5], (box[2], box[0]),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
    return img


# image name
# files = listdir(gdrive + runpath)
model = keras.models.load_model(gdrive + 'humanDectectionModel')
model2 = keras.models.load_model(gdrive + 'poseDectectionModel')
# name = files[2]
# img = cv2.imread(gdrive + runpath + '/' + name)
#
# img_plt = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# plt.imshow(img_plt)
# plt.show()
#
# boxe = find_detections(img, model, model2, 1.2)
# draw_detections(img, boxe)
# img_plt = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# plt.imshow(img_plt)
# plt.show()

vcap = cv2.VideoCapture(gdrive + 'TownCentreXVID.avi')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(gdrive + 'output2.avi', fourcc, 25.0, (680, 383), True)

counter = 0
good_boxes = []
multiTracker = cv2.MultiTracker_create()
start = time.time()
while counter < 650:
    # Capture frame-by-frame
    ret, frame = vcap.read()
    if frame is None:
        print("Frame is None")
        break

    frame = cv2.resize(frame, (680, 383))
    counter += 1
    # detect people every 15 frames
    if counter % 25 == 0:
        print("detect", counter)
        good_boxes = find_detections(frame, model, model2, 1.2)
        multiTracker = cv2.MultiTracker_create()
        for box in good_boxes:
            bbox = (box[2], box[0], box[3] - box[2], box[1] - box[0])
            multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)
    sucess, boxes = multiTracker.update(frame)
    for i, newbox in enumerate(boxes):
        good_boxes[i][0] = int(newbox[1])
        good_boxes[i][1] = int(newbox[1] + newbox[3])
        good_boxes[i][2] = int(newbox[0])
        good_boxes[i][3] = int(newbox[0] + newbox[2])
    draw_detections(frame, good_boxes)

    out.write(frame)
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    # Press q to close the video windows before it ends if you want
    if cv2.waitKey(22) & 0xFF == ord('q'):
        break

# When everything done, release the capture
vcap.release()
out.release()
cv2.destroyAllWindows()
end = time.time()
print("Video stop, time used to compute:", end - start)
