from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import time
import torch
from playsound import playsound

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('eye_state_detector.h5')

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# Initialize global variables to store prediction and eye closed time
current_prediction = "Eyes Open"
eye_closed_duration = 0
start_time =0

def detect_drowsiness():
    global current_prediction, eye_closed_duration,start_time
    cap = cv2.VideoCapture(0)
    alarm_triggered = False
   

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

                if prediction[0][0] < 0.5:  # Closed eye
                    current_prediction = "Eyes Closed"
                    
                    # START OF CHANGE: Ensure start_time is set when eyes are first detected as closed
                    if start_time == 0:  # If this is the first detection of closed eyes
                        start_time = time.time()
                    
                    # START OF CHANGE: Continuously update eye_closed_duration
                    eye_closed_duration = time.time() - start_time

                    # Trigger alarm if eyes closed for more than 5 seconds
                    if eye_closed_duration > 3:
                        playsound('mixkit-classic-alarm-995.wav')  # Play alarm sound
                        alarm_triggered = True
                        start_time = 0  # Reset start_time after alarm is triggered
                else:
                    current_prediction = "Eyes Open"
                    
                    # START OF CHANGE: Reset start_time and eye_closed_duration when eyes open
                    alarm_triggered = False
                    start_time = 0  # Reset start_time when eyes are open
                    eye_closed_duration = 0  # Reset eye_closed_duration as well

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global current_prediction, eye_closed_duration,start_time
    return jsonify(prediction=current_prediction, duration=eye_closed_duration,time=start_time)

if __name__ == '__main__':
    app.run(debug=True)
