from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
from label_image import get_labels
import time
import os

def start_video_stream(camera_id, image_recognizer):
    cap = cv2.VideoCapture(camera_id)
        
    while(True):
        # Capture - frame-by-frame
        ret, frame = cap.read()
        
        # our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(np.uint8(frame))
        img = pre_process_image(img)
        img = np.array(image_recognizer.classify_image(img))
        
        # Display the resulting frame
        cv2.imshow('frame', img)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def pre_process_image(image):
    # Color
    color_enhancer = ImageEnhance.Color(image)
    img = color_enhancer.enhance(0.5)
    
    # Sharpness
    sharp_enhancer = ImageEnhance.Sharpness(img)
    img = sharp_enhancer.enhance(1.5)
    
    # Brightness 
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(1.0)
    
    # Contrast
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(0.8)

    return img        

class ImageRecognizer:
    def __init__(self, filter_for_labels, region_of_interest):
        # self.output_path = output_path
        self.filter_for_labels = filter_for_labels
        self.model_path = "./detect.tflite"
        self.labels_path = "./labelmap.txt"
        self.object_font = ImageFont.truetype("Roboto-Bold.ttf", size=15)

        # Region of Interest provides the padding size from each edge (left, top, right, bottom)
        # It does not provide the x,y location of the padding
        self.roi = region_of_interest

    def classify_image(self, image):
        # Retrieve labels
        detected_objects = get_labels(image, self.model_path, self.labels_path, self.filter_for_labels, self.roi)

        # Draw the region of interest
        # left, top, right, bottom
        draw = ImageDraw.Draw(image)
        draw.rectangle((self.roi[0], self.roi[1], image.width-self.roi[2], image.height-self.roi[3]), fill=None, outline=2, width=3)  # Top left
        # draw.text((self.roi[0], image.height-self.roi[3]), "X", (0,0,0), font=self.object_font)  # Bottom Left
        # draw.text((image.width-self.roi[2], self.roi[1]), "X", (0,0,0), font=self.object_font)  # Top Right
        # draw.text((image.width-self.roi[2], image.height-self.roi[3]), "X", (0,0,0), font=self.object_font)  # Bottom Right

        # Draw the label on the image
        for detected_object in detected_objects:
            # Run
            text_location = (detected_object["location"][0], detected_object["location"][1]-10)
            rectangle = \
                (detected_object["location"][0], detected_object["location"][1]+10, \
                 detected_object["location"][2], detected_object["location"][3])
            display_text = detected_object["class"] + " " + str(int(detected_object["score"]*100))+"%"
            draw.text(text_location, display_text, \
                      (255, 255, 255), font=self.object_font)
            draw.rectangle(rectangle, fill=None, outline=3, width=3)

            # Save footage
            self.save_frame(image)

        return image

    def save_frame(self, image):
        # Save the image
        time_now = time.gmtime(time.time())
        str_time_now = \
            str(time_now.tm_year)+"_"+\
            str(time_now.tm_mon)+"_"+\
            str(time_now.tm_mday)+"_"+\
            str(time_now.tm_hour)+"_"+\
            str(time_now.tm_min)+"_"+\
            str(time_now.tm_sec)
        image.save("./footage/security_west_camera_"+str_time_now+".jpeg",format="jpeg")
        
        

        
if __name__ == "__main__":
    filter_for_labels = [
        "person",
        "bicycle", 
        "car", 
        "motorcycle", 
        "airplane", 
        "bus", 
        "truck", 
        "bird", 
        "cat", 
        "dog"
    ]
    image_recognizer = ImageRecognizer(filter_for_labels, (50, 200, 50, 5))
    start_video_stream(2, image_recognizer)
