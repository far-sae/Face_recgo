import os
import face_recognition
import pickle
from pathlib import Path

def encode_faces(dataset_dir='dataset', model_dir='models', model_filename='face_encodings.pkl'):
    if not Path(dataset_dir).exists():
        raise ValueError(f"Dataset directory '{dataset_dir}' does not exist.")

    known_encodings = []
    known_names = []

    for user_id in os.listdir(dataset_dir):
        user_path = os.path.join(dataset_dir, user_id)
        if not os.path.isdir(user_path):
            continue

        for img_file in os.listdir(user_path):
            img_path = os.path.join(user_path, img_file)
            if img_path.endswith('.jpg'):
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(user_id)

    if not known_encodings:
        raise RuntimeError("No valid face encodings found. Ensure that face images are available.")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    with open(model_path, 'wb') as model_file:
        pickle.dump({'encodings': known_encodings, 'names': known_names}, model_file)

    print(f"Model trained successfully and saved at '{model_path}'.")

if __name__ == "__main__":
    encode_faces()
