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

current_prediction = "Eyes Open"

while True:
        ret, frame = cap.read()
        if not ret:
            break

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

                if prediction[0][0] < 0.6:  # Closed eye
                    current_prediction = "Eyes Closed"
                    
                    # START OF CHANGE: Ensure start_time is set when eyes are first detected as closed
                    if start_time == 0:  # If this is the first detection of closed eyes
                        start_time = time.time()
                    
                    # START OF CHANGE: Continuously update eye_closed_duration
                    eye_closed_time = time.time() - start_time

                    # Trigger alarm if eyes closed for more than 5 seconds
                    if eye_closed_time > 1.5:
                        playsound('mixkit-classic-alarm-995.wav')  # Play alarm sound
                        alarm_triggered = True
                        start_time = 0  # Reset start_time after alarm is triggered
                else:
                    current_prediction = "Eyes Open"
                    
                    # START OF CHANGE: Reset start_time and eye_closed_duration when eyes open
                    alarm_triggered = False
                    start_time = 0  # Reset start_time when eyes are open
                    eye_closed_duration = 0  # Reset eye_closed_duration as well
        cv2.putText(frame, str(eye_closed_time), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, current_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
