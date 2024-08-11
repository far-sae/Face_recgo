import dlib
import cv2
import pickle
import os
import numpy as np
import logging

logging.basicConfig(level=logging.ERROR)

# Load the pre-trained face detection and face recognition models from dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('/Users/farazsaeed/Face-Auth/script/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('/Users/farazsaeed/Face-Auth/script/dlib_face_recognition_resnet_model_v1.dat')


def load_face_encodings(model_path='models/face_encodings.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    with open(model_path, 'rb') as model_file:
        data = pickle.load(model_file)
    return data


def encode_face(rgb_image, face_location):
    shape = shape_predictor(rgb_image, face_location)
    face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_image, shape))
    return face_encoding


def authenticate_user(data):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot access the webcam.")

    logging.info("Starting authentication...")

    while True:
        ret, frame = cam.read()
        if not ret:
            logging.warning("Failed to grab frame.")
            break

        # Convert the image from BGR (OpenCV format) to RGB (dlib format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        face_locations = detector(rgb_frame, 1)

        for face_location in face_locations:
            try:
                # Encode the face
                face_encoding = encode_face(rgb_frame, face_location)

                # Compare the face encoding to the known faces
                matches = np.linalg.norm(data['encodings'] - face_encoding, axis=1)
                name = "Unknown"

                if len(matches) > 0:
                    min_distance_index = np.argmin(matches)
                    if matches[min_distance_index] < 0.6:  # 0.6 is a common threshold
                        name = data['names'][min_distance_index]
                        logging.info(f"Authenticated: {name}")

                # Draw a rectangle around the face and display the name
                top, right, bottom, left = (
                face_location.top(), face_location.right(), face_location.bottom(), face_location.left())
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

            except Exception as e:
                logging.error(f"Error processing face location {face_location}: {str(e)}")

        # Display the resulting frame
        cv2.imshow('Authentication', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_data = load_face_encodings()
    authenticate_user(model_data)
