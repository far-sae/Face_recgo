
# Face Recognition Authentication System

This repository contains a face recognition-based authentication system that captures facial images, trains a model on those images, and then uses the trained model to authenticate users. The system leverages the `dlib`, `opencv`, and `face-recognition` libraries to perform these tasks.

## Features

- **Capture Faces**: The `capture_faces.py` script allows you to capture face images from a webcam and save them for training.
- **Train Model**: The `train_model.py` script processes the captured images and trains a face recognition model.
- **Authenticate Users**: The `Authenticate.py` script uses the trained model to authenticate users by recognizing their faces.

## Installation

To set up the environment, clone this repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/far-sae/face-recognition-authentication.git
cd face-recognition-authentication
pip install -r requirements.txt
```

## Usage

### Step 1: Capture Faces

Use the `capture_faces.py` script to capture and save images of the faces you want to train the model on.

```bash
python capture_faces.py
```

### Step 2: Train the Model

After capturing the images, use the `train_model.py` script to train the face recognition model.

```bash
python train_model.py
```

### Step 3: Authenticate Users

Finally, use the `Authenticate.py` script to authenticate users by recognizing their faces.

```bash
python Authenticate.py
```

## Files and Directories

- **`capture_faces.py`**: Script to capture face images using a webcam.
- **`train_model.py`**: Script to train the face recognition model using captured face images.
- **`Authenticate.py`**: Script to authenticate users based on the trained face recognition model.
- **`requirements.txt`**: File listing the required Python packages.
- **`face_encodings.pkl`**: Serialized face encodings used by the authentication script.

## Dependencies

The project relies on the following libraries:

- `opencv-python==4.5.4.58`
- `face-recognition==1.3.0`
- `numpy==1.21.4`
- `dlib==19.22.0`

You can install these dependencies using the `requirements.txt` file provided.
