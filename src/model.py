import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Input, Conv3D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from preprocessing import ClipsCaptions

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
        dataset.append(pkl.load(f))


with open('all_captions.pkl', 'rb') as f:
    all_captions = pkl.load(f)

with open('word_index.pkl', 'rb') as f:
    word_index = pkl.load(f)
    
with open('index_word.pkl', 'rb') as f:
    index_word = pkl.load(f)
    
vocab_size = len(word_index) + 1

with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pkl.load(f)

with open('val_dataset.pkl', 'rb') as f:
    val_dataset = pkl.load(f)

# Preprocess Video Frames
def preprocess_frames(frames):
    # Resize frames to 224x224 and normalize pixel values
    resized_frames = np.array([frame / 255.0 for frame in frames])
    return resized_frames

# Preprocess Captions
def preprocess_captions(captions):
    '''
    Preprocess list of captions for training. Lowercase, tokenize, pad sequences.
    
    Args:
    captions: List of captions for each video sample
    
    Returns:
    padded_seqs: Padded sequences of tokenized captions
    '''
    # Make all captions lowercase
    captions = [[caption.lower() for caption in sample] for sample in captions]
    # Tokenize captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    caption_seqs = tokenizer.texts_to_sequences(captions)
    padded_seqs = pad_sequences(caption_seqs, padding='post')
    return padded_seqs

# Prepare training and validation data
train_frames = np.array([preprocess_frames(sample.frames) for sample in train_dataset])
train_captions = preprocess_captions([sample.captions for sample in train_dataset])


val_frames = np.array([preprocess_frames(sample.frames) for sample in val_dataset])
val_captions = preprocess_captions([sample.captions for sample in val_dataset])

# Parameters
batch_size = 5
from preprocessing import frames as num_frames
frame_height = 224
frame_width = 224
num_channels = 3

# Batch the frames
num_of_batches = len(train_frames) // batch_size
train_frames = train_frames[:num_of_batches * batch_size]
train_captions = train_captions[:num_of_batches * batch_size]
print("\nNum of samples: ", len(train_frames))
print("Num of samples: ", len(train_captions))

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# Get max length
max_length = max(len(caption.split()) for caption in all_captions)

print(max_length)

print(f"\nTrain frames shape: {train_frames.shape}")
print(f"Val frames shape: {val_frames.shape}")
print("\n")

# Next-Frame Prediction Model using ConvLSTM to get Hidden Representations

input = Input(shape=(num_frames, frame_height, frame_width, num_channels))
convlstm = ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(input)
convlstm = BatchNormalization()(convlstm)
convlstm = ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(convlstm)
output = Conv3D(filters=3, kernel_size=(3,3,3), activation='sigmoid', padding='same')(convlstm)

model = Model(inputs=input, outputs=output)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Model Fit

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(train_frames, train_frames, batch_size=batch_size, epochs=10, validation_data=(val_frames, val_frames), callbacks=[early_stopping])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')  # Save the plot as a PNG image
plt.show()