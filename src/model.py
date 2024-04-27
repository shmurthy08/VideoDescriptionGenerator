import tensorflow as tf
from tensorflow.keras.layers import Layer, ConvLSTM2D, LayerNormalization, Dense, Dropout, LSTM, Flatten, Embedding, Conv2D, Permute, Multiply, Reshape, Input, Concatenate, MultiHeadAttention
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

    # Output feature maps from intermediate layers
    intermediate_model = Model(inputs=inputs, outputs=[convlstm1, convlstm2])

    return intermediate_model

# Assuming your video frames have shape (batch_size, samples, height, width, channels)
input_shape = (10, 224, 224, 3)

# Instantiate the model
model = self_attention_convLSTM_model(input_shape)
model.compile(optimizer='adam', loss='mse')  # Using mean squared error as the loss function for feature extraction

# Fit the model
model.fit(train_frames, train_frames, batch_size=batch_size, epochs=1, validation_data=(val_frames, val_frames))

# Extract features from the intermediate model
_, features = model(train_frames)

print(features.shape)

# Transformer Decoder using Lightweight Pretrained GPT
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# # Load the GPT2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
# gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2-medium')

# # Freeze the weights of the GPT2 model
# gpt2.trainable = False

# # Transformer Decoder
# def transformer_decoder(vocab_size, max_length, embed_dim, num_heads, ff_dim, gpt2):
#     # Input layers
#     input_ids = Input(shape=(max_length,), dtype=tf.int32)
#     features = Input(shape=(224, 224, 64))  # Adjust the shape as needed

#     # GPT2 model
#     gpt2_output = gpt2(input_ids)[0]

#     # Concatenate the GPT2 output and the features
#     concat = Concatenate()([gpt2_output, features])

#     # LSTM layer
#     lstm = LSTM(512)(concat)

#     # Output layer
#     outputs = Dense(vocab_size, activation='softmax')(lstm)

#     # Model
#     model = Model(inputs=[input_ids, features], outputs=outputs)

#     return model


# # Instantiate the model
# # Pad the captions to the required length
# train_captions = pad_sequences(train_captions, maxlen=30, padding='post')
# val_captions = pad_sequences(val_captions, maxlen=30, padding='post')

# vocab_size = len(word_index) + 1
# max_length = 30
# embed_dim = 768
# num_heads = 12
# ff_dim = 3072


# decoder_model = transformer_decoder(vocab_size, max_length, embed_dim, num_heads, ff_dim, gpt2)
# decoder_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# # combine the models
# class VideoDescriptionGenerator(Model):
#     def __init__(self, intermediate_model, decoder_model):
#         super(VideoDescriptionGenerator, self).__init__()
#         self.intermediate_model = intermediate_model
#         self.decoder_model = decoder_model

#     def call(self, inputs):
#         frames = inputs[0]
#         captions = inputs[1]

#         # Extract features from video frames
#         _, features = self.intermediate_model(frames)

#         # Generate captions using the decoder model
#         outputs = self.decoder_model([captions, features])

#         return outputs

# # Instantiate the VideoDescriptionGenerator model
# video_description_generator = VideoDescriptionGenerator(model, decoder_model)
# video_description_generator.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# video_description_generator.fit([train_frames, train_captions], train_captions, batch_size=batch_size, epochs=10, validation_data=([val_frames, val_captions], val_captions))

    
    

# # Instantiate the VideoDescriptionGenerator model
# video_description_generator = VideoDescriptionGenerator(model, decoder_model)
# video_description_generator.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# video_description_generator.fit([train_frames, train_captions], train_captions, batch_size=batch_size, epochs=10, validation_data=([val_frames, val_captions], val_captions))