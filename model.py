import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime

#define where onnx-file is
ort_session = onnxruntime.InferenceSession("efficientnet-lite4-11.onnx")

#load labels
labels = json.load(open("labels_map.txt","r"))

# set image file dimensions to 224x224 by resizing and cropping image from center
def preprocess(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def read_img(img_name):
    # read the image
    fname = img_name
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pre-process the image like mobilenet and resize it to 224x224
    img = preprocess(img, (224, 224, 3))

    # create a batch of 1 (that batch size is buned into the saved_model)
    img_batch = np.expand_dims(img, axis=0)
    return img_batch

#run model
def run_model(img):
    results = ort_session.run(["Softmax:0"], {"images:0": read_img(img)})[0]
    result = reversed(results[0].argsort()[-5:])
    resultStr=""
    first=True
    for r in result:
        if first:
            first=False
            resultStr=labels[str(r)]+ " ("+ str(results[0][r])+")"
        else:
            resultStr+="<br>"+labels[str(r)]+ " ("+ str(results[0][r])+")"
        
        print(r, labels[str(r)], results[0][r])
    return resultStr
