
import numpy as np
import cv2
import pyttsx3
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load pre-trained MobileNet SSD model and classNames
prototxt = "MobileNetSSD_deploy.prototxt"
caffe_model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Function to speak asynchronously
def speak_async(text):
    engine.setProperty('rate', 240)  # Speed of speech
    engine.say(text)
    engine.runAndWait()

def calculate_distance_to_object(object_height, object_type, focal_lengths):
    # Calculate distance using simple geometric formula
    real_heights = {
    'background': 2.0,
    'aeroplane': 5.0,
    'bicycle':1.2,
    'bird': 0.3,
    'boat': 1.6,
    'bottle': 0.25,
    'bus': 3.6,
    'car': 1.8,  
    'cat': 0.3,
    'chair': 0.8,
    'cow': 1.6,
    'diningtable': 0.7,
    'dog': 0.5,
    'horse': 1.6,
    'motorbike': 0.9,
    'person': 1.7, 
    'pottedplant': 0.4,
    'sheep': 0.7,
    'sofa': 0.8,
    'train': 3.5,
    'tvmonitor':0.7
}

    real_height = real_heights.get(object_type, 1.0)  # Default height if object type is unknown
    focal_length = focal_lengths.get(object_type, 333)  # Using the calculated focal length
    #Improvements: Load camera calibration data or perform camera calibration
    
    if object_height == 0:
        return float('inf')  # Return infinity if object height is zero 
    
    distance = real_height * focal_length / object_height
    return distance




cap = cv2.VideoCapture("walking2.mp4")
# cap = cv2.VideoCapture("http://10.135.76.96:4747/video")
# cap=cv2.VideoCapture(1)  #for live video



while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended")
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(300, 300),
                                 mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])  
            if class_id in classNames:
                label = classNames[class_id]
                x_top_left = int(detections[0, 0, i, 3] * width)
                y_top_left = int(detections[0, 0, i, 4] * height)
                x_bottom_right = int(detections[0, 0, i, 5] * width)
                y_bottom_right = int(detections[0, 0, i, 6] * height)
                object_height = y_bottom_right - y_top_left
                distance = calculate_distance_to_object(object_height, object_type=label, focal_lengths={})
                label_with_distance = label + ", Distance: " + "{:.2f}".format(distance) + " meters"
                cv2.putText(frame, label_with_distance, (x_top_left, y_top_left - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)

                if distance < 4.0:
                    speech_text = "There is a " + label_with_distance + " from you"
                    # Speak asynchronously
                    threading.Thread(target=speak_async, args=(speech_text,)).start()

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
