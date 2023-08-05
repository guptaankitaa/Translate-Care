import os

import cv2
import numpy as np

from keypoint_detection import draw_landmarks, mp_hands
from keypoint_detection import extract_keypoints
from keypoint_detection import mediapipe_detection
from keras.utils import to_categorical

DATA_PATH = os.path.join('C:/Rafif/SKRIPSI/Proyek Skripsi - Pycharm/DATA')
gestures = np.array(['pain or ill', 'nurse, call bell', 'toilet', 'change that', 'hot', 'doctor, nurse', 'lie down',
                     'turn over', 'medication', 'frightened, worried', 'sit', 'wash', 'cold', 'food', 'drink',
                     'teeth, dentures', 'fetch, need that', 'home', 'spectacles', 'book or magazine',
                     'stop, finish that', 'yes, good', 'help me', 'no, bad'])
no_sequence = 15
sequence_length = 10
start_folder = 0


def collect_data():
    isdir = os.path.isdir(DATA_PATH)

    if isdir:
        for gesture in gestures:
            dir_max = np.max(np.array(os.listdir(os.path.join(DATA_PATH, gesture))).astype(int))
            for sequence in range(1, no_sequence + 1):
                try:
                    os.makedirs(os.path.join(DATA_PATH, gesture, str(dir_max + sequence)))
                except:
                    pass

    else:
        for gesture in gestures:
            for sequence in range(no_sequence):
                try:
                    os.makedirs(os.path.join(DATA_PATH, gesture, str(sequence)))
                except:
                    pass

    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)

    # Set mediapipe model
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for gesture in gestures:
            for sequence in range(start_folder, start_folder + no_sequence):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, hands)

                    draw_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'MENGAMBIL FRAME: {} VIDEO NOMOR: {}'.format(gesture, sequence), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Hand Gesture Recognition', image)


                    else:
                        cv2.putText(image, 'MENGAMBIL FRAME {} VIDEO NOMOR: {}'.format(gesture, sequence), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Hand Gesture Recognition', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, gesture, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()
    return


def append_data():
    no_sequences = 190
    label_map = {label: num for num, label in enumerate(gestures)}
    # print(label_map)
    sequences, labels = [], []
    for gesture in gestures:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, gesture, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[gesture])

    return np.array(sequences), to_categorical(labels).astype(int)


if __name__ == "__main__":
    collect_data()
