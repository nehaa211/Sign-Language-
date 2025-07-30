""" 
pickle- use to load the pre-tained ML model
cv2- use to capture video, images
mediapipe- provide prebuild ML solutions
numpy- use to perform numerical operation
time- use for measure intervals or adding delays 

"""

import pickle 
import cv2
import mediapipe as mp
import numpy as np
import time

# string and formating parameters
def wrap_text(text, max_width, font, font_scale, thickness):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

# Load pre-trained model (Machine Initialize)
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Open the webcam for live video processing
cap = cv2.VideoCapture(0)  


#Mediapipe Setup
mp_hands = mp.solutions.hands
# visualize the hand landmarks on the video
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,          # initialize model for video streams
    max_num_hands=1,                  # Use one hand
    min_detection_confidence=0.3      # Confidence threshold
)


# This dictionary maps model outputs  acting on the predicted letter.
labels_dict = { 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z','backspace': 'backspace'
}

# To store word
current_word = []   

# To store the full sentence
sentence = []       


 # Stores the current letter detected as candidate.
candidate_letter = None   
 # Timestamp when the candidate letter was first recognized 
candidate_start_time = None 
# Duration (in seconds) for confirming a stable letter
stability_threshold = 1.5   

# Timer for current word when no hand is detected
last_hand_detection_time = time.time()
# Timeout for finalizing the word if the hand disappears
no_hand_timeout = 0.8  


# Font style
font = cv2.FONT_HERSHEY_SIMPLEX          # Font type
font_scale = 1                           # font size
thickness = 2                            # Thickness of the font


# capture and process video frames
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    # ret - is a Boolean value indicating if the frame was successfully captured
    if not ret:
        break  # Stop if there is an issue with the webcam

    # Flip the frame mirror-like view
    frame = cv2.flip(frame, 1)
    
    # Frame dimensions
    H, W, _ = frame.shape  
    
    # Convert BGR to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    # current time used for later timing check
    current_time = time.time()
    
    # Processes the RGB frame through the Mediapipe hand tracking module
    results = hands.process(frame_rgb)

    # variable to store current detected letter
    letter_from_frame = None

    # detection and prediction of guesture
    if results.multi_hand_landmarks:
        last_hand_detection_time = current_time 
         
        # select the first detected hand and store its landmarks
        hand_landmarks = results.multi_hand_landmarks[0]

        # draw hand landmarks using mediapipe default styles
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # use to store features from landmarks
        temp_data = []
        
        # extract all x and y coordinates for each landmark
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        # minimum x and y values to normalize the feature set
        x_min = min(x_coords)
        y_min = min(y_coords)
        
        # subtracting the minimum coordinate values to normalize the data of each landmark
        for lm in hand_landmarks.landmark:
            temp_data.append(lm.x - x_min)
            temp_data.append(lm.y - y_min)

        # converts the list of feature into a numpy array
        data_aux = np.array(temp_data)
        
        # this model matches what the model expects
        expected_feature_size = 84
        
        # if the vector in shorter then pads with 0 
        if len(data_aux) < expected_feature_size:
            data_aux = np.pad(data_aux, (0, expected_feature_size - len(data_aux)), mode='constant')
            
            # if longer trims it to the correct size this ensures compatibility with the pre-trained model
        elif len(data_aux) > expected_feature_size:
            data_aux = data_aux[:expected_feature_size]

        # Predict the letter using the model
        prediction = model.predict([data_aux])
        predicted_letter = str(prediction[0])
        letter_from_frame = labels_dict.get(predicted_letter, "?")

        # clear current word if "backspace" is detected
        if letter_from_frame == "backspace":
             # Clear only the current word
            current_word = [] 
            
            candidate_letter = None
            candidate_start_time = None
            # Skip the rest of the frame processing
            continue  

        if letter_from_frame != "?":
            if candidate_letter != letter_from_frame:
                candidate_letter = letter_from_frame
                candidate_start_time = current_time
            else:
                if current_time - candidate_start_time >= stability_threshold:
                    current_word.append(candidate_letter)
                    candidate_letter = None
                    candidate_start_time = None

    else:
        # Ffnalize the word when no hand is detected
        if (current_time - last_hand_detection_time > no_hand_timeout) and current_word:
            sentence.append("".join(current_word))
            current_word = []
            candidate_letter = None
            candidate_start_time = None

    # constructs strings that represent the detected letter, the current in-progress word, and the full sentence made of finalized words
    detected_word_text = "Detected Word: " + (letter_from_frame if letter_from_frame else "")
    word_text = "Word: " + "".join(current_word)
    combined_sentence = "Sentence: " + " ".join(sentence)

    # call function wrap_text to break the sentence into multiple lines so that it fits within the frame
    wrapped_sentence_lines = wrap_text(combined_sentence, W - 40, font, font_scale, thickness)

    # the needed height for a background rectangle 
    rectangle_height = 20 + len(wrapped_sentence_lines) * 30 + 80
    overlay = frame.copy()
    
    # blends the overlay with the frame to create a semi-transparent effect (using an alpha value of 0.6)
    cv2.rectangle(overlay, (0, H - rectangle_height), (W, H), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # vertical coordinate
    y_start = H - rectangle_height + 40
    
    # display "Detected Word"
    cv2.putText(frame, detected_word_text, (10, y_start), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # increase it again
    y_start += 40
    
    # display "Word"
    cv2.putText(frame, word_text, (10, y_start), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    # Display the wrapped sentence
    y_start += 40
    for line in wrapped_sentence_lines:
        cv2.putText(frame, line, (10, y_start), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_start += 30

    # display frame
    cv2.imshow('Sign Language', frame)

    # press "q" to quit
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        print("Exiting program due to 'q' key press.")
        break

    # press "c" to clear sentence
    if key == ord('c'):
        print("Clearing sentence with 'c' key!")
        # Clear only the sentence
        sentence = []

# Release resources

# relesed the webcam
cap.release()

# close all onencv window thar opened
cv2.destroyAllWindows()