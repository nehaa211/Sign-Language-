🤟 Sign Language Recognition System

A computer vision-based system that captures hand gestures via webcam and classifies them using machine learning. Built to assist communication for the deaf and hard-of-hearing community.

Project Structure

- `collect.py` – Captures hand gesture images and stores them in labeled folders
- `createDataSet.py` – Extracts hand landmarks using MediaPipe and prepares dataset
- `trainModel.py` – Trains a Random Forest classifier on the extracted features
- `runModel.py` – Runs real-time prediction using webcam and displays results

 Technologies Used

- Python
- OpenCV
- MediaPipe
- scikit-learn
- pickle
- NumPy

 Dataset

Images are collected manually using `collect.py`. Each class is stored in a separate folder under `./Data`. Hand landmarks are extracted and normalized for training.
(use dataset from kaggle dataset)

 Getting Started

### Installation

git clone
https://github.com/nehaa211/Sign-Language-.git
cd sign-language-recognition
pip install -r requirements.txt


Run Data Collection
python collect.py


Create Dataset
python createDataSet.py


Train Model
python trainModel.py

Run Real-Time Prediction
python runModel.py


 Model
- Classifier: Random Forest
- Accuracy: ~90% (based on test split)
- Features: Normalized hand landmark coordinates














