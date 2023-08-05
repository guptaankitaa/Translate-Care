from flask import Flask, render_template, Response, send_from_directory
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import time

from tensorflow import keras
from generate_csv import get_connections_list, get_distance

from keypoint_detection import mp_hands, mediapipe_detection, draw_landmarks, extract_keypoints
from telegram_send import save_image, send_msg
from vis_prediction import visualize_prediction
from model import model as md

from tensorflow.keras.models import load_model


app = Flask(__name__)

# handGesture

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

def handGesture_gen_frames():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read each frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)

        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handGesture')
def handGesture():
    return render_template('index1.html')

@app.route('/handGesture_video_feed')
def handGesture_video_feed():
    return Response(handGesture_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Character ASL 
def get_sign_list():
    # Function to get all the values in SIGN column
    df = pd.read_csv('connections.csv', index_col=0)
    return df['SIGN'].unique()

sign_list = get_sign_list()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
connections_dict = get_connections_list()
model1 = keras.models.load_model('ann_model.h5')

def gen_frames2():  
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            # Get image from webcam, change color channels and flip
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Get result
            results = hands.process(image)
            if not results.multi_hand_landmarks:
                # If no hand detected, then just display the webcam frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
            else:
                # If hand detected, superimpose landmarks and default connections
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                # Get landmark coordinates & calculate length of connections
                coordinates = results.multi_hand_landmarks[0].landmark
                data = []
                for _, values in connections_dict.items():
                    data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                
                # Scale data
                data = np.array([data])
                data[0] /= data[0].max()

                # Get prediction
                pred = np.array(model1(data))
                pred = sign_list[pred.argmax()]

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # Display text showing prediction
                image = cv2.putText(
                    image, pred, (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, 
                    (255, 0, 0), 2
                )

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/tmp')
def tmp():
    """Video streaming home page."""
    return render_template('index2.html')

def process_image(image):
    # Write code here to process the image
    pass

def gen():
    while True:
        frame = yield
        process_image(frame)

@app.route('/video_feed22')
def video_feed22():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tmp2')
def tmp2():
    return render_template('index3.html')


# patient Gesture
def gen3():
    md.load_weights('skripsi.h5')
    sequence = []
    predictions = []
    threshold = 0.7
    output_label_counter = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, hands)
            if results.multi_hand_landmarks:
                draw_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-10:]

                if len(sequence) == 10:
                    res = md.predict(np.expand_dims(sequence, axis=0))[0]
                    output_label = np.argmax(res)
                    predictions.append(output_label)
                    visualize_prediction(image, res)

                    if res[np.argmax(res)] > threshold:
                        for label in range(24):
                            if output_label == label:
                                output_label_counter += 1
                                if output_label_counter >= 50:
                                    cv2.putText(image, 'PESAN DIKIRIM', (250, 200),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                                    files = save_image(image)
                                    send_msg(output_label + 1, files)
                                    output_label_counter = 0
                    else:
                        output_label_counter = 0

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                        cv2.LINE_AA)
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            key = cv2.waitKey(20)
            if key == 27:
                break



@app.route('/templates/<filename>')
def serve_image(filename):
    return send_from_directory('templates', filename, mimetype='image/jpeg')


@app.route('/video_feed33')
def video_feed33():
    return Response(gen3(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)