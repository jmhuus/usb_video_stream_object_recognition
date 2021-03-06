# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def get_labels(image, model_path, labels_path, filter_for_labels, roi):

  interpreter = tflite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # Original image shape
  orig_height = image.height
  orig_width = image.width

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = image.resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - 127.5) / 127.5

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  
  locations = interpreter.get_tensor(output_details[0]['index'])[0]
  classes = interpreter.get_tensor(output_details[1]['index']).astype(int)[0]
  scores = interpreter.get_tensor(output_details[2]['index'])[0]
  number_of_detections = interpreter.get_tensor(output_details[3]['index'])[0]

  labels = load_labels(labels_path)

  # # DEBUGGING CODE
  # for i in range(len(classes)):
  #   if scores[i] >= 0.65:
  #     print("{} {} {} {} {}".format(labels[classes[i]+1], locations[i][0], locations[i][1], locations[i][2], locations[i][3]))

  width_compression = orig_width/width
  height_compression = orig_height/height
  objects = []
  for i in range(len(classes)):
    detected_object = {}

    # Filter out objects that are obviously too large for this security setup
    object_height = abs((locations[i][0]*orig_height)-(locations[i][2]*orig_height))
    object_width = abs((locations[i][1]*orig_width)-(locations[i][3]*orig_width))
    if labels[classes[i]+1] == "person" and (object_width >= 75 or object_height >= 100):
      continue
    elif object_width >= 400 or object_height >= 400:
      continue

    # Track objects that meet the final filtering criteria
    if scores[i] >= 0.48 and labels[classes[i]+1] in filter_for_labels:
      detected_object["class"] = labels[classes[i]+1]
      detected_object["score"] = scores[i]
      detected_object["location"] = \
        (
          locations[i][1]*orig_width, \
          locations[i][0]*orig_height, \
          locations[i][3]*orig_width, \
          locations[i][2]*orig_height \
        )
      
      # Check if the image is in the selected region of interest
      # roi = left, top, right, bottom
      x1 = detected_object["location"][0]
      y1 = detected_object["location"][1]
      x2 = detected_object["location"][2]
      y2 = detected_object["location"][3]
      if x1<roi[0] or y1<roi[1] or x2>(image.width-roi[2]) or y2>(image.height-roi[3]):
        continue

      # # DEBUGGING CODE
      # object_height = (locations[i][0]*orig_height)-(locations[i][2]*orig_height)
      # object_width = (locations[i][1]*orig_width)-(locations[i][3]*orig_width)
      # print("{} height: {} width: {}".format(detected_object["class"], abs(object_height), abs(object_width)))

    if detected_object:
      objects.append(detected_object)


  return objects
    
    

  
