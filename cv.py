import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound
import time

# Load the trained model
model = tf.keras.models.load_model('eye_state_detector.h5')

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
alarm_triggered = False
eye_closed_time = 0
start_time = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face)

        for (ex, ey, ew, eh) in eyes:
            eye = face[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (24, 24))
            eye = eye / 255.0
            eye = np.expand_dims(eye, axis=0)

            prediction = model.predict(eye)
            print(prediction)

            if prediction < 0.9:  # Closed eye
                if not alarm_triggered:
                    if eye_closed_time == 0:
                        start_time = time.time()
                    eye_closed_time = time.time() - start_time
                    print("eye_closed")
                    print(eye_closed_time )
                    
                    if eye_closed_time > 3 :
                        playsound('mixkit-classic-alarm-995.wav')
                        alarm_triggered = True
            else:
                alarm_triggered = False
                eye_closed_time = 0

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
