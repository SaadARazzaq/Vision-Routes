print('Setting UP')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import keras.losses  # Import Keras losses module for potential debugging

# Initialize Socket.IO server
sio = socketio.Server(logger=True, engineio_logger=True)  # Enable logging
# Initialize Flask app
app = Flask(__name__)

# Maximum speed for the car
maxSpeed = 10

def preProcess(img):
    """Preprocess the input image."""
    print("Preprocessing image...")
    img = img[60:135, :, :]  # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian Blur
    img = cv2.resize(img, (200, 66))  # Resize the image
    img = img / 255.0  # Normalize the image
    print("Image preprocessed.")
    return img

print("Loading model...")
model = None
try:
    # Load the pre-trained model
    model = load_model('models/nvidia_model.h5', compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

@sio.on('telemetry', namespace='/')
def telemetry(sid, data):
    """Handle telemetry data."""
    print("Received telemetry data from session id:", sid)
    speed = float(data['speed'])
    # Decode and preprocess the image
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    # Predict the steering angle
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed  # Calculate the throttle
    print(f'Steering: {steering}, Throttle: {throttle}, Speed: {speed}')
    sendControl(steering, throttle)

@sio.on('connect', namespace='/')
def connect(sid, environ):
    """Handle a new connection."""
    print(f'Connected: session id {sid}')
    sendControl(0, 0)

@sio.on('disconnect', namespace='/')
def disconnect(sid):
    """Handle a disconnection."""
    print(f'Disconnected: session id {sid}')

def sendControl(steering, throttle):
    """Send control commands to the car."""
    print("Sending control...")
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    }, namespace='/')
    print("Control sent.")

if __name__ == '__main__':
    print("Starting server...")
    # Wrap the Flask application with Socket.IO middleware
    app = socketio.WSGIApp(sio, app)
    # Start the WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
