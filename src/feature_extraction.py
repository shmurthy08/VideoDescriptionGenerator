import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Input, Conv3D, BatchNormalization, Add
import numpy as np
import pickle as pkl
import os
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt


# Preprocess Video Frames
def preprocess_frames(frames):
    # Resize frames to 224x224 and normalize pixel values
    resized_frames = np.array([frame / 255.0 for frame in frames])
    return resized_frames

dataset_dir = '../data/preprocessed_5frames_5captions'
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
num_frames = 5
frame_height = 224
frame_width = 224
num_channels = 3





# Create model
# Encoder
inputs = Input(shape=(num_frames, frame_height, frame_width, num_channels))
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(inputs)
x1 = BatchNormalization()(x)  # First block output for residual connection

x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x1)
x = BatchNormalization()(x)
x = Add()([x, x1])  # Add input and output of the block (residual connection)

x2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x2)

x3 = ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x3)

x4 = ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x4)


# Decoder
x = ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = Add()([x, x4])  # Add input from the second block output

x = ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = Add()([x, x3])  # Add input and output of the block (residual connection)

x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = Add()([x, x2])  # Add input and output of the block (residual connection)

x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = Add()([x, x1])  # Add input from the first block output

outputs = Conv3D(filters=3, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')(x)  # Output layer to match input channels
autoencoder = Model(inputs, outputs)    
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Print model summary
autoencoder.summary()

def train_generator():
    for features in train_frames:
        yield (features, features)
def val_generator():
    for features in val_frames:
        yield (features, features)

output_types = (tf.float32, tf.float32)
output_shapes = (tf.TensorShape([5, 224, 224, 3]), tf.TensorShape([5, 224, 224, 3]))


# Create the dataset from the generator
train_data = tf.data.Dataset.from_generator(
    train_generator,
    output_types=output_types,
    output_shapes=output_shapes
).batch(15)
val_data = tf.data.Dataset.from_generator(
    val_generator,
    output_types=output_types,
    output_shapes=output_shapes
).batch(15)


# Prefetch
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)


# Model Checkpoint and LR Scheduler
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('fe_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, verbose=1)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1)
history = autoencoder.fit(train_data, epochs=1, validation_data=val_data, callbacks=[model_checkpoint, reduce_lr, earlystop], verbose=1)

plot_model(autoencoder, to_file='architecture.png', show_shapes=True, show_layer_names=True)
autoencoder.save('Feature_extract.h5')

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model: Mean Absolute Error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('fe_accuracy_plot.png')  # Save the plot as a PNG image
