import time, socket, sys, requests
import PIL

import keyboard
import os

from itertools import product
from PIL import Image
import os.path
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import sys

import numpy as np
import tensorflow as tf

sys.path.append("..")

# Import utilites
from the_utils import label_map_util
from the_utils import visualization_utils as vis_util

IMAGE_NAME = 'sample_image.jfif'

# Number of classes the object detector can identify
NUM_CLASSES = 7

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = "models/frozen_inference_graph.pb"


# Path to label map file
PATH_TO_LABELS = "models\label_map.pbtxt"

# Path to image
PATH_TO_IMAGE = IMAGE_NAME

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value



print("Initialising....\n")
time.sleep(1)

s = socket.socket()
host = socket.gethostname()
ip = socket.gethostbyname(host)
port = 1234
s.bind((host, port))
print(host, "(", ip, ")\n")
name = "me"

s.listen(1)
print("\nWaiting for incoming connections...\n")
conn, addr = s.accept()
print("Received connection from ", addr[0], "(", addr[1], ")\n")

s_name = "stav"


while True:
    message = conn.recv(1024)
    message = message.decode()
    what_tofind = message
    print(what_tofind)
    while "http" not in message:
        message = conn.recv(1024)
        message = message.decode()
    response = requests.get(message)
    file = open("sample_image.jfif", "wb")
    file.write(response.content)
    file.close()
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    # Draw the results of the detection (aka 'visulaize the results')
    if what_tofind == "7":
        occur = 0.3
    else:
        if what_tofind == "5":
            occur = 0.2
        else:
            if what_tofind == "4":
                occur = 0.25
            else:
                if what_tofind == "1":
                    occur = 0.8
                else:
                    occur = 0.01
    coordinates = vis_util.coordinates_find(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=occur)
    print(coordinates)
    squares = []
    for cord in coordinates:
        rows = []

        if cord[4][0].split(":")[0] == what_tofind:
            for i in range(4):
                print(cord[i] % 0.25)
                if (i == 0) or (i == 1) or (cord[i] % 0.25 > 0.03) or (0 == int(cord[i] / 0.25)):
                    if ((i == 0) or (i == 1)) and (cord[i] % 0.25 > 0.23):
                        rows.append((int(cord[i] / 0.25) + 2))
                    else:
                        rows.append((int(cord[i] / 0.25) + 1))
                else:
                    rows.append(int(cord[i] / 0.25))

            for j in range(rows[2] - rows[0] + 1):
                for t in range(rows[3] - rows[1] + 1):
                    squares.append((5 - rows[0] - j) * 4 - rows[1] - t + 1)

        if (squares == []) and (what_tofind=="3"):
            if (cord[4][0].split(":")[0] == "2") or (cord[4][0].split(":")[0]) == "1":
                for i in range(4):
                    print(cord[i] % 0.25)
                    if (i == 0) or (i == 1) or (cord[i] % 0.25 > 0.03) or (0 == int(cord[i] / 0.25)):
                        if ((i == 0) or (i == 1)) and (cord[i] % 0.25 > 0.23):
                            rows.append((int(cord[i] / 0.25) + 2))
                        else:
                            rows.append((int(cord[i] / 0.25) + 1))
                    else:
                        rows.append(int(cord[i] / 0.25))

                for j in range(rows[2] - rows[0] + 1):
                    for t in range(rows[3] - rows[1] + 1):
                        squares.append((5 - rows[0] - j) * 4 - rows[1] - t + 1)


        if (squares == []) and (what_tofind == "4"):
            if cord[4][0].split(":")[0] == "7":
                for i in range(4):
                    print(cord[i] % 0.25)

                    if (i == 0) or (i == 1) or (cord[i] % 0.25 > 0.03) or (0 == int(cord[i] / 0.25)):
                        if ((i == 0) or (i == 1)) and (cord[i] % 0.25 > 0.23):
                            rows.append((int(cord[i] / 0.25) + 2))
                        else:
                            rows.append((int(cord[i] / 0.25) + 1))
                    else:
                        rows.append(int(cord[i] / 0.25))

                for j in range(rows[2] - rows[0] + 1):
                    for t in range(rows[3] - rows[1] + 1):
                        squares.append((5 - rows[0] - j) * 4 - rows[1] - t + 1)

        if (squares == []) and (what_tofind == "5"):
            if cord[4][0].split(":")[0] == "1":
                for i in range(4):
                    print(cord[i] % 0.25)
                    if (i == 0) or (i == 1) or (cord[i] % 0.25 > 0.03) or (0 == int(cord[i] / 0.25)):
                        if ((i == 0) or (i == 1)) and (cord[i] % 0.25 > 0.23):
                            rows.append((int(cord[i] / 0.25) + 2))
                        else:
                            rows.append((int(cord[i] / 0.25) + 1))
                    else:
                        rows.append(int(cord[i] / 0.25))

                for j in range(rows[2] - rows[0] + 1):
                    for t in range(rows[3] - rows[1] + 1):
                        squares.append((5 - rows[0] - j) * 4 - rows[1] - t + 1)
        print(rows)
        print(squares)

    final_arr = [0] * 16  # makes an empty array for number of object in each tile

    #os.remove("sample_image.jfif")
    for i in squares:  # puts the number of object in each tile to the array
        final_arr[i-1] = 1



    print(final_arr)

    conn.send(str(final_arr).encode())





    if message == "[e]":
        message = "Left chat room!"
        conn.send(message.encode())
        print("\n")
        break