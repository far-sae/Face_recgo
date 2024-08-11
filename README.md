
# Custom Image Feature Analysis (CIFA) System

This repository contains a Custom Image Feature Analysis (CIFA) system designed to process and analyze images, extracting meaningful features, and making predictions based on a trained model. The system includes scripts for both feature extraction and prediction.

## Features

- **Custom Image Feature Extraction**: The `CIFA.py` script is designed to process images and extract custom features, which can be used for training and analysis.
- **Prediction**: The `predict.py` script utilizes a trained model to make predictions on new images based on the extracted features.

## Installation

To set up the environment, clone this repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/far-sae/Face_recgo_both_train_run.git
cd Face_recgo_both_train_run
pip install -r requirements.txt
```

## Download Required Models

To use this system, you need to download the following models:

1. **shape_predictor_68_face_landmarks.dat**: [Download here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
   - After downloading, extract the file and place it in the project directory.

2. **dlib_face_recognition_resnet_model_v1.dat**: [Download here](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2).
   - After downloading, extract the file and place it in the project directory.

3. **frozen_inference_graph.pb**: [Download here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz).
   - After downloading, extract the file and place it in the project directory.

## Usage

### Step 1: Feature Extraction

Use the `CIFA.py` script to extract features from your image dataset.

```bash
python CIFA.py --input_dir /path/to/your/images --output_dir /path/to/save/features
```

### Step 2: Prediction

After extracting the features, use the `predict.py` script to make predictions using a trained model.

```bash
python predict.py --model /path/to/your/model --features /path/to/extracted/features
```

## Files and Directories

- **`CIFA.py`**: Script for extracting custom image features from a dataset.
- **`predict.py`**: Script for making predictions based on extracted features and a trained model.
- **`requirements.txt`**: File listing the required Python packages.
- **`shape_predictor_68_face_landmarks.dat`**: Pre-trained model for facial landmark detection (to be downloaded).
- **`dlib_face_recognition_resnet_model_v1.dat`**: Pre-trained model for face recognition (to be downloaded).
- **`frozen_inference_graph.pb`**: Pre-trained TensorFlow model (to be downloaded).

## Dependencies

The project relies on several Python libraries, which are listed in the `requirements.txt` file. You can install these dependencies using the command mentioned above.
