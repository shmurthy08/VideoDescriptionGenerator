from conv_transformer import TransformerEncoder, TransformerDecoder, VideoCaptioningModel, conv_lstm_extractor, LRSchedule
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
import matplotlib.pyplot as plt
# Parameters
num_frames = 5
frame_height = 224
frame_width = 224
num_channels = 3

epoch_num = 75
batch_size = 32

with open('word_index.pkl', 'rb') as f:
    word_index = pkl.load(f)

# Add <unk> token to word_index dict
word_index['<unk>'] = len(word_index) + 1

with open('index_word.pkl', 'rb') as f:
    index_word = pkl.load(f)

# Add <unk> token to index_word
index_word[len(word_index)+1] = '<unk>'

vocabulary_size = len(word_index) + 1

with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pkl.load(f)

# Preprocess Video Frames
def preprocess_frames(frames):
    '''
    For each video: normalize pixel values.
    '''
    # Resize frames to 224x224 and normalize pixel values
    resized_frames = np.array([frame / 255.0 for frame in frames])
    return resized_frames

test_frames = np.array([preprocess_frames(sample.frames) for sample in test_dataset])
# Define Model
# Create a learning rate schedule
num_test_steps = (len(test_dataset) // 75) * epoch_num
num_warmup_steps = num_test_steps // 75

# Create model
ex_model = conv_lstm_extractor()
encoder = TransformerEncoder(embed_dim=1408, num_heads=5, drop_rate=0.1)
decoder = TransformerDecoder(embed_dim=1408, ff_dim=3584, num_heads=5, vocab_size=vocabulary_size, drop_rate=0.1)
caption_model = VideoCaptioningModel(ex_model, encoder, decoder)
caption_model.build(input_shape=(None, num_frames, frame_height, frame_width, num_channels))
# Now load the weights
caption_model.load_weights('caption_model_weights.h5')

# Generate Captions
# Generate Captions for the first set of frames in the test set
first_test_frames = test_frames[0:32]
print(first_test_frames.shape)

predictions = caption_model(first_test_frames, training=False)
print(predictions.shape)
def decode_captions(predictions):
    '''
    Given a batch of predictions, decode them into text captions.
    '''
    return predictions




# # Plot the frames with the generated captions
# fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))

# for i, ax in enumerate(axes):
#     ax.imshow(first_test_frames[0, i])
#     ax.set_title(decoded_captions[i])
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig('generated_captions.png')
# plt.show()