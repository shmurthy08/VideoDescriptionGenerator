import pickle
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
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
from preprocessing import output_dir as dataset_dir
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
# set a seed
np.random.seed(42)
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

# Pickle a set with ALL captions
all_captions = []
for clip in dataset:
    all_captions.extend(clip.captions)

with open('all_captions.pkl', 'wb') as f:
    pickle.dump(all_captions, f)
    
print(all_captions[:5])