"""
os - Use for interacting with the operating system
pickle- use to load the pre-tained ML model
cv2- use to capture video, images
mediapipe- provide prebuild ML solutions

"""

import os
import pickle
import mediapipe as mp
import cv2
# import matplotlib.pyplot as plt

# Initialize Mediapipe modules for hand tracking
mp_hands = mp.solutions.hands  

# Provides utilities to visualize hand landmarks
mp_drawing = mp.solutions.drawing_utils  

# Includes predefined styles for hand landmarks
mp_drawing_styles = mp.solutions.drawing_styles  

# Configure Mediapipe hand tracking settings
hands = mp_hands.Hands(
    static_image_mode=True,            # initialize model for video streams
    min_detection_confidence=0.3       # confidence threshold 
)

# Directory containing image dataset
DATA_DIR = './Data'

# Lists to store extracted features and corresponding labels

 # Stores extracted hand landmark coordinates
data = [] 

# Stores label for each image
labels = []  

# Iterate through all subdirectories in the dataset folder
for dir_ in os.listdir(DATA_DIR):
    
    # Loop through each image inside the subdirectory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store normalized hand landmark positions
        x_ = []        # Stores x-coordinates of detected landmarks
        y_ = []        # Stores y-coordinates of detected landmarks

        # Read image from file
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  
        
        # Convert image from BGR to RGB format for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        # Detect hand landmarks using Mediapipe
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            
            # Process each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Store raw x and y coordinates for normalization
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)  
                    y_.append(y)

                # Normalize hand landmark coordinates to make them relative
                for i in range(len(hand_landmarks.landmark)):
                    
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize x by subtracting the minimum value
                    data_aux.append(y - min(y_))  # Normalize y by subtracting the minimum value

            # Append the extracted features and labels
            data.append(data_aux)
            labels.append(dir_)

# Save extracted features and labels into a serialized file using pickle
f = open('data.pickle', 'wb')

 # Store data in dictionary format
pickle.dump({'data': data, 'labels': labels}, f) 

 # Close file after writing
f.close() 