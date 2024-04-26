import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from preprocessing import ClipsCaptions

# Function to create word_index and index_word dictionaries
def create_word_index(dataset):
    word_index = {}
    index = 1  # Start index from 1, 0 will be used for padding

    for clip in dataset:
        captions = clip.captions
        for caption in captions:
            words = caption.split()
            for word in words:
                if word not in word_index:
                    word_index[word] = index
                    index += 1

    # Reverse mapping from index to word
    index_word = {index: word for word, index in word_index.items()}
    
    return word_index, index_word

# Load the pickled dataset
# Specify the directory containing the dataset files
dataset_dir = '../data/preprocessed_10frames_5captions'
# Get the paths of all dataset files
dataset_paths = []
for file in os.listdir(dataset_dir):
    if file.endswith('.pkl'):
        dataset_paths.append(os.path.join(dataset_dir, file))

# Load the pickled dataset
dataset = []
for path in dataset_paths:
    with open(path, 'rb') as f:
        dataset.append(pickle.load(f))

# Shuffle the dataset
np.random.shuffle(dataset)

# Split the dataset into train, test, and validation sets
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))

train_data = dataset[:train_size]
val_data = dataset[train_size:train_size + val_size]
test_data = dataset[train_size + val_size:]

print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")
print(f"Number of testing samples: {len(test_data)}")

# Create word_index and index_word dictionaries
word_index, index_word = create_word_index(train_data + val_data + test_data)

# Save the word_index and index_word dictionaries
with open('word_index.pkl', 'wb') as f:
    pickle.dump(word_index, f)

with open('index_word.pkl', 'wb') as f:
    pickle.dump(index_word, f)

# Save the split datasets
with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('val_dataset.pkl', 'wb') as f:
    pickle.dump(val_data, f)

with open('test_dataset.pkl', 'wb') as f:
    pickle.dump(test_data, f)
