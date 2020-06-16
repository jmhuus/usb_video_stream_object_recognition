from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from label_image import get_labels

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
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(0.5)
    
    # Sharpness
    sharp_enhancer = ImageEnhance.Sharpness(img)
    img = sharp_enhancer.enhance(1.5)
    
    # Brightness 
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(1.5)
    
    # Contrast
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(0.8)

    return img        

class ImageRecognizer:
    def __init__(self, filter_for_labels):
        # self.output_path = output_path
        self.filter_for_labels = filter_for_labels
        self.model_path = "./detect.tflite"
        self.labels_path = "./labelmap.txt"
        self.object_font = ImageFont.truetype("Roboto-Bold.ttf", size=15)

    def classify_image(self, image):
        # Retrieve labels
        detected_objects = get_labels(image, self.model_path, self.labels_path, self.filter_for_labels)

        # Draw the label on the image
        draw = ImageDraw.Draw(image)
        for detected_object in detected_objects:
            text_location = (detected_object["location"][0], detected_object["location"][1]-10)
            rectangle = \
                (detected_object["location"][0], detected_object["location"][1]+10, \
                 detected_object["location"][2], detected_object["location"][3])
            draw.text(text_location, detected_object["class"], \
                      (255, 255, 255), font=self.object_font)
            draw.rectangle(rectangle, fill=None, outline=3, width=3)

        return image
        

        
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
    image_recognizer = ImageRecognizer(filter_for_labels)
    start_video_stream(2, image_recognizer)
