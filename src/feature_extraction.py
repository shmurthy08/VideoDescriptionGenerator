import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Input, Conv3D, LayerNormalization, Conv2D, Flatten, Dense, Embedding, BatchNormalization, Lambda, MultiHeadAttention, GlobalAveragePooling2D, Reshape, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import pickle as pkl
import os
from preprocessing import ClipsCaptions
from tensorflow.keras.utils import plot_model


# Preprocess Video Frames
def preprocess_frames(frames):
    # Resize frames to 224x224 and normalize pixel values
    resized_frames = np.array([frame / 255.0 for frame in frames])
    return resized_frames

dataset_dir = '../data/preprocessed_10frames_5captions'
# Get the paths of all dataset files
dataset_paths = []
for file in os.listdir(dataset_dir):
    if file.endswith('.pkl'):
        dataset_paths.append(os.path.join(dataset_dir, file))


with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pkl.load(f)

with open('val_dataset.pkl', 'rb') as f:
    val_dataset = pkl.load(f)





train_frames = np.array([preprocess_frames(sample.frames) for sample in train_dataset])
val_frames = np.array([preprocess_frames(sample.frames) for sample in val_dataset])



# Parameters
batch_size = 7
num_frames = 5
frame_height = 224
frame_width = 224
num_channels = 3

# Encoder
inputs = Input(shape=(num_frames, frame_height, frame_width, num_channels))
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(inputs)
x1 = BatchNormalization()(x)  # First block output for residual connection

x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x1)
x = BatchNormalization()(x)
x = Add()([x, x1])  # Add input and output of the block (residual connection)

x2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)  # 
x = BatchNormalization()(x2)

x3 = ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x3)


# Decoder
x = ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = Add()([x, x3])  # Add input from the second block output

x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = Add()([x, x2])  # Add input and output of the block (residual connection)

x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = Add()([x, x1])  # Add input from the first block output

outputs = Conv3D(filters=3, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')(x)  # Output layer to match input channels

# Create model
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# Print model summary
autoencoder.summary()

# Model Checkpoint and LR Scheduler
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('fe_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss', mode='min', verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, verbose=1)

autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
autoencoder.fit(train_frames, train_frames, batch_size=10, epochs=25, validation_data=(val_frames, val_frames), callbacks=[model_checkpoint, early_stopping, reduce_lr], verbose=1)

plot_model(autoencoder, to_file='architecture.png', show_shapes=True, show_layer_names=True)
autoencoder.save('Feature_extract.h5')