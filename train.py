import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Function to pad sequences to the same length
def pad_sequences(data, pad_value=0):
    max_length = max(len(sublist) for sublist in data)
    padded_data = [sublist + [pad_value] * (max_length - len(sublist)) for sublist in data]
    return np.array(padded_data)

# Load data from the pickle file
data_dict = pickle.load(open('./dataset_processed_all_data.pickle', 'rb'))

# Pad the sequences in data
data = pad_sequences(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file
with open('dataset_processed_all_data.p', 'wb') as f:
    pickle.dump({'model': model}, f)
    