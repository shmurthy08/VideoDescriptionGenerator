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

# Preprocess Captions
def preprocess_captions(captions, word_index, max_length=20):
    """
    Preprocess list of captions for training: Lowercase, tokenize using word_index, and pad sequences.

    Args:
    captions: List of list of captions (2D list where each sublist contains captions for one video).
    word_index: Dictionary mapping words to indices.
    max_length: Maximum length of tokenized caption sequences. Defaults to 20.

    Returns:
    tokenized_captions: 2D numpy array of padded, tokenized captions.
    """
    # Tokenize and pad each caption
    tokenized_captions = []
    for caption_list in captions:
        # Process each caption in the list of captions for one video
        tokenized_caption_list = []
        for caption in caption_list:
            # Lowercase, split, and convert words to indices, using 0 for unknown words
            tokens = [word_index.get(word.lower(), word_index['<unk>']) for word in caption.split()]
            tokenized_caption_list.append(tokens)
        tokenized_captions.append(tokenized_caption_list)

    # Flatten the list and pad sequences
    tokenized_captions_flat = [token for sublist in tokenized_captions for token in sublist]
    padded_captions = pad_sequences(tokenized_captions_flat, maxlen=max_length, padding='post', truncating='post')
    return np.array(padded_captions).reshape(len(captions), -1, max_length)

# Prepare training data
test_captions_data = [sample.captions for sample in test_dataset]
test_captions = preprocess_captions(test_captions_data, word_index)

# Define Model
# Create a learning rate schedule
num_test_steps = (len(test_dataset) // 75) * epoch_num
num_warmup_steps = num_test_steps // 75

# Create model
ex_model = conv_lstm_extractor()
encoder = TransformerEncoder(embed_dim=1408, num_heads=5, drop_rate=0.1)
decoder = TransformerDecoder(embed_dim=1408, ff_dim=3584, num_heads=5, vocab_size=vocabulary_size, drop_rate=0.1)
caption_model = VideoCaptioningModel(ex_model, encoder, decoder)
caption_model.build((test_frames.shape))
# Now load the weights
caption_model.load_weights('caption_model_weights.h5')

# Generate Captions
# Generate Captions for the first set of frames in the test set

predictions = caption_model((test_frames, test_captions), training=True, batch_size=batch_size)

print(predictions)