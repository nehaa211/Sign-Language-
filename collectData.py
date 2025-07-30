import os
import cv2 as cv
import string

# Creation of the 'DataDemo' directory to store the collected data/images
DATA_DIR = './Data'

# Checking if any folder exists with the 'DataDemo' name
if not os.path.exists(DATA_DIR):
    # If the folder does not exist, create it
    os.makedirs(DATA_DIR)

# Define the number of subfolders for different classes of data
number_of_classes = 1  

# Set the number of images to be collected for each class
dataset_size = 10  

# Initialize the webcam using OpenCV and store the video capture object in 'cap'
cap = cv.VideoCapture(0)

# Loop through each class to create its corresponding subfolder
for label in range(number_of_classes):
    class_path = os.path.join(DATA_DIR, str(label))  
    if not os.path.exists(class_path):
        
        # Create the subfolder for the given class
        os.makedirs(class_path)

    print('Collecting data for class {}'.format(label))

    done = False
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        frame = cv.resize(frame, (1080, 720))  

        # Display instructions on the screen for the user
        cv.putText(frame, 'To start collecting data - press "y"', (150, 700), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

        # Show the current webcam feed
        cv.imshow('Sign Language', frame)

        # Wait for the user to press 'y' to start data collection
        if cv.waitKey(25) == ord('y'):
            break  

    counter = 0
    while counter < dataset_size:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        frame = cv.resize(frame, (720, 480))  

        # Display the collection process information on the screen
        cv.imshow('Collecting Data for : {}'.format(label), frame)

        # Pause briefly before proceeding
        cv.waitKey(25)

        # Save the captured image in the respective class folder
        cv.imwrite(os.path.join(DATA_DIR, str(label), '{}.jpg'.format(counter)), frame)

        # Increment the counter for image collection
        counter += 1  

        # Press "q" to quit data collection early
        key = cv.waitKey(25) & 0xFF
        if key == ord('q'):
            print("Exiting program due to 'q' key press.")
            break  


# Release the webcam resource after completion
cap.release()
cv.destroyAllWindows()