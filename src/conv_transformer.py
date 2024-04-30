import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Input, Conv3D, LayerNormalization, Conv2D, Flatten, Dense, Embedding, BatchNormalization, Lambda, MultiHeadAttention, GlobalAveragePooling2D, Reshape
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import pickle as pkl
import os
from preprocessing import ClipsCaptions
from keras_nlp.layers import TokenAndPositionEmbedding
import tensorflow_models as tfm
from tensorflow.keras.utils import plot_model



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
        dataset.append(pkl.load(f))


with open('all_captions.pkl', 'rb') as f:
    all_captions = pkl.load(f)

with open('word_index.pkl', 'rb') as f:
    word_index = pkl.load(f)

# Add <unk> token to word_index dict
word_index['<unk>'] = len(word_index) + 1

with open('index_word.pkl', 'rb') as f:
    index_word = pkl.load(f)

# Add <unk> token to index_word
index_word[0] = '<unk>'    

vocabulary_size = len(word_index) + 1

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
def preprocess_captions(captions, word_index, max_length=15):
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


max_caption_length = 20 
train_captions_data = [sample.captions for sample in train_dataset]  # Extract captions from each sample
train_captions = preprocess_captions(train_captions_data, word_index, max_caption_length)
train_frames = np.array([preprocess_frames(sample.frames) for sample in train_dataset])


# Prepare validation data
val_captions_data = [sample.captions for sample in val_dataset]
val_captions = preprocess_captions(val_captions_data, word_index, max_caption_length)
val_frames = np.array([preprocess_frames(sample.frames) for sample in val_dataset])







print(train_frames[0].shape)  
print(train_captions.shape)
print(type(train_frames))
print(type(train_captions))

# Parameters
batch_size = 7
num_frames = 5
frame_height = 224
frame_width = 224
num_channels = 3

# Build ConvLSTM Encoder
inputs = Input(shape=(num_frames-1, frame_height, frame_width, num_channels))
convlstm = ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same', return_sequences=True)(inputs)
convlstm = LayerNormalization()(convlstm)
convlstm = ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(convlstm)
convlstm = LayerNormalization()(convlstm)
conv = Conv3D(filters=32, kernel_size=(3,3,3), activation='sigmoid', padding='same')(convlstm)
conv = LayerNormalization()(conv)
output = Conv3D(filters=3, kernel_size=(3,3,3), activation='sigmoid', padding='same')(conv)

def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y
x_train, y_train = create_shifted_frames(train_frames)
x_val, y_val = create_shifted_frames(val_frames)

# Example slicing for training data
model = Model(inputs=inputs, outputs=output)

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
model.fit(x_train, y_train, batch_size=10, epochs=10, validation_data=(x_val, y_val))

plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)


# # Transformer Encoder to embed image features
# class TransformerEncoder(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attn1 = MultiHeadAttention(num_heads, embed_dim, dropout=0.3)
#         self.batchnorm1 = BatchNormalization()
#         self.layernorm1 = LayerNormalization()

#         self.dense = Dense(dense_dim, activation='relu')
#     def call(self, inputs, training):
#         inputs = self.dense(inputs)
#         inputs = self.batchnorm1(inputs)
#         attn_out = self.attn1(query=inputs, value=inputs, key=inputs, attention_mask=None, training=training)
#         out = self.layernorm1(attn_out + inputs)
#         return out
    
# class PositionalEmbedding(tf.keras.layers.Layer):
#     def __init__(self, vocab_size, embed_dim, seq_len, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.seq_len = seq_len
#         self.token_emb = Embedding(vocab_size, embed_dim)
#         self.pos_emb = Embedding(vocab_size, embed_dim)
#         self.scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))
        
#     def call(self, inputs):
#         length = tf.shape(inputs)[-1]
#         pos = tf.range(start=0, limit=length, delta=1)
#         emb_tokens = self.token_emb(inputs)
#         emb_tokens = emb_tokens * self.scale
#         emb_pos = self.pos_emb(pos)
#         return emb_tokens + emb_pos
#     def mask(self, inputs, mask=None):
#         return tf.math.not_equal(inputs, 0)
    
# class TransformerDecoder(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, dense_dim, ff_dim, num_heads, vocab_size, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.ff_dim = ff_dim
#         self.num_heads = num_heads
#         self.attn1 = MultiHeadAttention(num_heads, embed_dim, dropout=0.3)
#         self.attn2 = MultiHeadAttention(num_heads, embed_dim, dropout=0.3)
#         self.batchnorm1 = BatchNormalization()
#         self.batchnorm2 = BatchNormalization()
#         self.layernorm1 = LayerNormalization()
#         self.ff_nn1 = Dense(ff_dim, activation='relu')
#         self.ff_nn2 = Dense(embed_dim)
#         self.ff_nn3 = Dense(vocab_size, activation='softmax')
#         self.embed = PositionalEmbedding(embed_dim=embed_dim, vocab_size=vocabulary_size, seq_len=20)
#         self.masking = True
#     def call(self, inputs, encoder_out, training, mask=None):
#         inputs = self.embed(inputs)
#         causal_mask = self.causal_attn_mask(inputs)
#         combo_mask = None
#         pad_mask = None
#         if mask is not None:
#             pad_mask = tf.cast(mask[:, :, tf.newaxis], tf.int32) 
#             combo_mask = tf.cast(mask[:, :, tf.newaxis], tf.int32)   
#             combo_mask = tf.minimum(combo_mask, causal_mask)
            
#         attn_out1 = self.attn1(query=inputs, value=inputs, key=inputs, attention_mask=combo_mask, training=training)
#         out1 = self.batchnorm1(attn_out1 + inputs)
#         attn_out2 = self.attn2(query=out1, value=encoder_out, key=encoder_out, attention_mask=pad_mask, training=training)
#         out2 = self.batchnorm2(attn_out2 + out1)
        
#         ff_out1 = self.ff_nn1(out2)
#         ff_out2 = self.ff_nn2(ff_out1)
#         out_norm = self.layernorm1(ff_out2 + out2)
#         final_out = self.ff_nn3(out_norm)
#         return final_out
#     def causal_attn_mask(self, inputs):
#         shape = tf.shape(inputs)
#         batch_size, seq_len = shape[0], shape[1]
#         i = tf.range(seq_len)[:, tf.newaxis]
#         j = tf.range(seq_len)
#         mask = tf.cast(i >= j, dtype='int32')
#         mask = tf.reshape(mask, (1, seq_len, seq_len))
#         multiply = tf.concat([
#             tf.expand_dims(batch_size, -1),
#             tf.constant([1, 1], dtype=tf.int32)
#         ], axis=0)
        
        
#         return tf.tile(mask, multiply)        
                
    



# class VideoCaptioningModel(Model):
#     def __init__(
#         self,
#         conv_lstm_model,
#         transformer_encoder,
#         transformer_decoder,
#         num_captions_per_video=5,
#     ):
#         super().__init__()
#         self.conv_lstm_model = conv_lstm_model
#         self.encoder = transformer_encoder
#         self.decoder = transformer_decoder
#         self.loss_tracker = tf.metrics.Mean(name="loss")
#         self.acc_tracker = tf.metrics.Mean(name="accuracy")
#         self.num_captions_per_video = num_captions_per_video

#     def calculate_loss(self, y_true, y_pred, mask):
#         loss = self.compiled_loss(y_true, y_pred)
#         mask = tf.cast(mask, dtype=loss.dtype)
#         loss *= mask
#         return tf.reduce_sum(loss) / tf.reduce_sum(mask)

#     def calculate_accuracy(self, y_true, y_pred, mask):
#         # Ensure y_true is cast to int32 to match the output type of tf.argmax, which is int64 by default
#         y_true = tf.cast(y_true, dtype=tf.int64)
#         accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
#         accuracy = tf.math.logical_and(mask, accuracy)
#         accuracy = tf.cast(accuracy, dtype=tf.float32)
#         mask = tf.cast(mask, dtype=tf.float32)
#         return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

#     def _compute_caption_loss_and_acc(self, video_features, batch_seq, training=True):
#         encoder_out = self.encoder(video_features, training=training)
#         batch_seq_inp = batch_seq[:, :-1]
#         batch_seq_true = batch_seq[:, 1:]
#         mask = tf.math.not_equal(batch_seq_true, 0)
#         batch_seq_pred = self.decoder(
#             batch_seq_inp, encoder_out, training=training, mask=mask
#         )
#         loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
#         acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
#         return loss, acc

#     def train_step(self, batch_data):
#         batch_frames, batch_seq = batch_data
#         batch_loss = 0
#         batch_acc = 0

#         # 1. Get video embeddings from ConvLSTM
#         video_features = self.conv_lstm_model(batch_frames)

#         # 2. Pass each of the captions one by one to the decoder
#         for i in range(self.num_captions_per_video):
#             with tf.GradientTape() as tape:
#                 loss, acc = self._compute_caption_loss_and_acc(
#                     video_features, batch_seq[:, i, :], training=True
#                 )
#                 batch_loss += loss
#                 batch_acc += acc

#             # Get the list of all the trainable weights and apply gradients
#             train_vars = self.trainable_variables
#             grads = tape.gradient(loss, train_vars)
#             self.optimizer.apply_gradients(zip(grads, train_vars))

#         # Update the trackers
#         batch_acc /= float(self.num_captions_per_video)
#         self.loss_tracker.update_state(batch_loss)
#         self.acc_tracker.update_state(batch_acc)

#         return {
#             "loss": self.loss_tracker.result(),
#             "acc": self.acc_tracker.result(),
#         }

#     # Include test_step and metrics as in your initial structure
#     def test_step(self, batch_data):
#         batch_frames, batch_seq = batch_data
#         batch_loss = 0
#         batch_acc = 0

#         video_features = self.conv_lstm_model(batch_frames)

#         for i in range(self.num_captions_per_video):
#             loss, acc = self._compute_caption_loss_and_acc(
#                 video_features, batch_seq[:, i, :], training=False
#             )
#             batch_loss += loss
#             batch_acc += acc

#         batch_acc /= float(self.num_captions_per_video)
#         self.loss_tracker.update_state(batch_loss)
#         self.acc_tracker.update_state(batch_acc)

#         return {
#             "loss": self.loss_tracker.result(),
#             "acc": self.acc_tracker.result(),
#         }

#     @property
#     def metrics(self):
#         return [self.loss_tracker, self.acc_tracker]
#     def call(self, inputs):
#         video_features, captions = inputs
#         encoder_out = self.encoder(video_features, training=False)
#         decoder_out = self.decoder(captions, encoder_out, training=False)
#         return decoder_out



# conv_lstm = build_conv_lstm(num_frames, frame_height, frame_width, num_channels)
# encoder = TransformerEncoder(embed_dim=64, dense_dim=64, num_heads=2)
# decoder = TransformerDecoder(embed_dim=64, dense_dim=64, ff_dim=256, num_heads=3, vocab_size=vocabulary_size)

# caption_model = VideoCaptioningModel(conv_lstm, encoder, decoder, num_captions_per_video=5)
# caption_model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # caption_model.fit(train_frames, train_captions, batch_size=1, epochs=1, validation_data=(val_frames, val_captions))


