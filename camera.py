import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing import image
from keras_preprocessing.image import img_to_array

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        age_model = load_model("age_cnn.h5")
        success, frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray=cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = roi_gray / 255.0
    
            age_predict = age_model.predict(roi_gray.reshape(1, 128, 128, 1))
            age = round(age_predict[0][0])
            age_label_position = (x+h, y+h)
            cv2.putText(frame, "Age="+str(age), age_label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()