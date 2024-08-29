from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import time
from playsound import playsound

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('eye_state_detector.h5')

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    alarm_triggered = False
    eye_closed_time = 0
    start_time = 0

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

                if prediction < 0.5:  # Closed eye
                    if not alarm_triggered:
                        if eye_closed_time == 0:
                            start_time = time.time()
                        eye_closed_time = time.time() - start_time

                        if eye_closed_time >4:
                            playsound('mixkit-classic-alarm-995.wav')  # Play alarm sound
                            alarm_triggered = True
                else:
                    alarm_triggered = False
                    eye_closed_time = 0

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

if __name__ == '__main__':
    app.run(debug=True)
