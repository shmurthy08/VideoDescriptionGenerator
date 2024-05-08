from conv_transformer import TransformerEncoder, TransformerDecoder, VideoCaptioningModel, def_conv_lstm_extractor
import numpy as np
import pickle as pkl
import os


with open('word_index.pkl', 'rb') as f:
    word_index = pkl.load(f)

# Add <unk> token to word_index dict
word_index['<unk>'] = len(word_index) + 1

with open('index_word.pkl', 'rb') as f:
    index_word = pkl.load(f)

# Add <unk> token to index_word
index_word[len(word_index)+1] = '<unk>'    


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


test_frames = np.array([preprocess_frames(sample.frames) for sample in train_dataset])
test_captions = [sample.captions for sample in test_dataset]
# Convert Captions to word indices
test_captions = [[word_index.get(word) for word in caption.split()] for caption in test_captions]


# Create model
ex_model = conv_lstm_extractor()

encoder = TransformerEncoder(embed_dim=1152, num_heads=3, drop_rate=0.1)

decoder = TransformerDecoder(embed_dim=1152, ff_dim=1536, num_heads=3, vocab_size=vocabulary_size, drop_rate=0.1)

caption_model = VideoCaptioningModel(ex_model, encoder, decoder)

# Load weights
caption_model.load_weights('caption_model_weights.h5')


# Predict captions using caption_model
# Note: start token is startseq and end token is endseq
def predict_captions(model, frames, seq_len=15):
    ...
    

