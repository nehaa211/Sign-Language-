"""

pickle- used to load the pre-tained ML model
numpy- used to perform numerical operation
sklearn.ensemble - contains the RandomForestClassifier for training ML models
sklearn.model_selection - used to split the dataset for training and testing
sklearn.metrics - used to evaluate model performance using accuracy score

"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset stored in pickle format
with open('./data.pickle', 'rb') as f:
    
    # Load the serialized data dictionary
    data_dict = pickle.load(f)  
    
# Extract feature vectors from loaded data
data_list = data_dict['data']

# Ensure all feature vectors have a uniform length for model compatibility
max_length = max(len(item) if isinstance(item, list) else 0 for item in data_list)

def standardize_vector(item, target_length):

    if isinstance(item, list):
        
          # if the vector in shorter then pads with 0 
        if len(item) < target_length:
            item.extend([0] * (target_length - len(item)) )  
            
        # truncate if too long
        return np.array(item[:target_length])  
    
    # convert to array if not already a list
    return np.array(item)  

# Standardize all feature vectors to maintain uniform dimensions
data = np.array([standardize_vector(item, max_length) for item in data_list], dtype=np.float32)

 # Store labels associated with the data
labels = np.array(data_dict['labels']) 

# Split data into training and testing sets for model evaluation
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model using the training dataset
model.fit(x_train, y_train)  

# Predict on the test dataset to evaluate model performance
y_predict = model.predict(x_test)

# Compute accuracy
score = accuracy_score(y_test, y_predict)  

# Print classification accuracy in percentage format
print(f"{score * 100:.2f}% of data were classified correctly!")

# Save the trained model for future use
with open('model.p', 'wb') as f:
    
    # Serialize and store the trained model
    pickle.dump({'model': model}, f)  