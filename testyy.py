from PIL import Image
import cv2
import numpy as np


cap = cv2.VideoCapture(2)

while(True):
    # Capture - frame-by-frame
    ret, frame = cap.read()

    # our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(np.uint8(frame))
    img.save("test.jpg")
    break
    
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
