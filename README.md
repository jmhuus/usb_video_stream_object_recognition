### USB Video Object Detection
This is a simple application that uses a USB video stream, such as a webcam, and tensorflow to detect objects such as cars, people, animals, etc. A batch of object labels can be provided to the program for additional logging details to effectively timestamp each instance and it's associated video stream.


#### Stack
* Python
* Tensorflow Lite
* OpenCv
    * Video streaming module
* Local OS - Lubuntu


* Be sure to change tflite_runtime in requirements.txt to match your target OS. I'm using Ubuntu to run this program on an edge device.