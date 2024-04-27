import tensorflow as tf
from tensorflow.keras.layers import Layer, ConvLSTM2D, LayerNormalization, Dense, Dropout, LSTM, Flatten, Embedding, Conv2D, Permute, Multiply, Reshape, Input, Concatenate, MultiHeadAttention, Add, LayerNormalization, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import numpy as np
import pickle as pkl


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


print(train_frames[0].shape)  
print(train_captions[0].shape)
print(type(train_frames))
print(type(train_captions))

# Parameters
batch_size = 128
num_frames = 10
frame_height = 224
frame_width = 224
num_channels = 3

# Batch the frames
num_of_batches = len(train_frames) // batch_size
train_frames = train_frames[:num_of_batches * batch_size]
train_captions = train_captions[:num_of_batches * batch_size]
print("Num of samples: ", len(train_frames))
print("Num of samples: ", len(train_captions))



# Self-Attention ConvLSTM Network for Feature Extraction
def self_attention(inputs):
    # Assuming inputs shape is (batch_size, samples, height, width, channels)
    x = Permute((1, 2, 3, 4))(inputs)  # Permute to (samples, height, width, channels, batch_size)
    f = Conv2D(filters=1, kernel_size=1, padding='same')(x)  # Learnable weights
    g = Conv2D(filters=1, kernel_size=1, padding='same')(x)  # Learnable weights
    h = Conv2D(filters=inputs.shape[-1], kernel_size=1, padding='same')(x)  # Learnable weights

    s = tf.keras.activations.softmax(f)  # Attention map
    o = Multiply()([s, h])  # Element-wise multiplication
    return o

def self_attention_convLSTM_model(input_shape):
    inputs = Input(shape=input_shape)

    # Apply self-attention mechanism
    attended_inputs = self_attention(inputs)
    print("Attended inputs shape: ", attended_inputs.shape)
    # Reshape the attended inputs
    reshaped_inputs = Reshape((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))(attended_inputs)
    # ConvLSTM layers
    convlstm1 = ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=True)(reshaped_inputs)
    convlstm2 = ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=True)(convlstm1)


    return Model(inputs=inputs, outputs=convlstm2)


# Decoder with LSTM
def decoder_lstm_model(vocab_size, max_length, embedding_dim):
    inputs = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = LSTM(512, return_sequences=True)(x)
    x = Dense(vocab_size, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)

# Model Creation
frame_shape = (num_frames, frame_height, frame_width, num_channels)

# Create Sequential Model that combines encoder and decoder
encoder = self_attention_convLSTM_model(frame_shape)
decoder = decoder_lstm_model(vocab_size, max_length=20, embedding_dim=256)

# Define the input layers
frame_input = Input(shape=frame_shape)
caption_input = Input(shape=(train_captions.shape[1],))

# Get the output from the encoder
encoded_frames = encoder(frame_input)

# Get the output from the decoder
decoded_captions = decoder(caption_input)

# Define the model
model = Model(inputs=[frame_input, caption_input], outputs=decoded_captions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([train_frames, train_captions], train_captions, batch_size=batch_size, epochs=5, validation_data=([val_frames, val_captions], val_captions))
