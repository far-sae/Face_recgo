
# Face Recognition System

This repository contains a Face Recognition System designed to capture facial images, train a model on those images, and authenticate users using the trained model. The system uses the `dlib` library along with `opencv` for facial detection and recognition.

## Features

- **Capture Faces**: The `capture_faces.py` script captures face images from a webcam and saves them for training.
- **Train Model**: The `train_model.py` script trains a face recognition model using the captured face images.
- **Authenticate Users**: The `Authenticate.py` script uses the trained model to authenticate users by recognizing their faces.

## Installation

To set up the environment, clone this repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/far-sae/Face_recgo.git
cd Face_recgo
pip install -r requirements.txt
```

## Download Required Models

To use this system, you need to download the following models:

1. **shape_predictor_68_face_landmarks.dat**: [Download here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
   - After downloading, extract the file and place it in the project directory.

2. **dlib_face_recognition_resnet_model_v1.dat**: [Download here](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2).
   - After downloading, extract the file and place it in the project directory.

## Usage

### Step 1: Capture Faces

Use the `capture_faces.py` script to capture face images that you want to use for training.

```bash
python capture_faces.py
```

### Step 2: Train the Model

After capturing the images, use the `train_model.py` script to train the face recognition model.

```bash
python train_model.py
```

### Step 3: Authenticate Users

Use the `Authenticate.py` script to authenticate users based on the trained face recognition model.

```bash
python Authenticate.py
```

## Files and Directories

- **`capture_faces.py`**: Script for capturing face images using a webcam.
- **`train_model.py`**: Script for training the face recognition model using the captured images.
- **`Authenticate.py`**: Script for authenticating users based on the trained face recognition model.
- **`requirements.txt`**: File listing the required Python packages.
- **`shape_predictor_68_face_landmarks.dat`**: Pre-trained model for facial landmark detection (to be downloaded).
- **`dlib_face_recognition_resnet_model_v1.dat`**: Pre-trained model for face recognition (to be downloaded).

## Dependencies

The project relies on several Python libraries, which are listed in the `requirements.txt` file. You can install these dependencies using the command mentioned above.
