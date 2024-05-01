import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Input, Conv3D, LayerNormalization, Conv2D, Flatten, Dense, Embedding, BatchNormalization, Lambda, MultiHeadAttention, GlobalAveragePooling2D, Reshape, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
import numpy as np
import pickle as pkl
import os
from preprocessing import ClipsCaptions
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
import datetime



dataset_dir = '../data/preprocessed_5frames_5captions'
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

# Call ConvLSTM Feature Extraction
fe_model = load_model('fe_model.h5')
def conv_lstm_extractor():
    # Freeze all layers
    for layer in fe_model.layers:
        layer.trainable = False
    feature_output = fe_model.layers[-10].output
    # Flatten with TimeDistributed
    feature_output = TimeDistributed(Flatten())(feature_output)
    extract = Model(inputs=fe_model.input, outputs=feature_output)
    print(extract.summary())
    return extract


# Transformer Encoder to embed image features
class VideoPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.pos_encoding = self.add_weight("positional_encoding", 
                                            shape=(1, self.seq_len, self.embed_dim),
                                            initializer=self.positional_encoding_initializer,
                                            trainable=False)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return inputs + tf.broadcast_to(self.pos_encoding, [batch_size, self.seq_len, self.embed_dim])
    
    def positional_encoding_initializer(self, shape, dtype=None):
        angle_rads = self.get_angles(np.arange(self.seq_len)[:, np.newaxis],
                                     np.arange(self.embed_dim)[np.newaxis, :],
                                     self.embed_dim)
        # Apply sine to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cosine to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=dtype)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attn1 = MultiHeadAttention(num_heads, embed_dim, dropout=0.3)
        self.layernorm1 = LayerNormalization()
        self.vidpos = VideoPositionalEmbedding(seq_len=5, embed_dim=embed_dim)
        self.dense = Dense(embed_dim, activation='relu')
        self.add_layer = tf.keras.layers.Add()  # Add residual connection layer

    def call(self, inputs, training):
        inputs = self.dense(inputs)
        inputs = self.layernorm1(inputs)
        inputs = self.vidpos(inputs)
        attn_out = self.attn1(query=inputs, value=inputs, key=inputs, attention_mask=None, training=training)
        # Residual connection
        attn_out = self.add_layer([inputs, attn_out])
        return attn_out
    
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(vocab_size, embed_dim)
        self.scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))
        
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        pos = tf.range(start=0, limit=length, delta=1)
        emb_tokens = self.token_emb(inputs)
        emb_tokens = emb_tokens * self.scale
        emb_pos = self.pos_emb(pos)
        return emb_tokens + emb_pos
    def mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attn1 = MultiHeadAttention(num_heads, embed_dim, dropout=0.4)
        self.attn2 = MultiHeadAttention(num_heads, embed_dim, dropout=0.2)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layer_norm3 = LayerNormalization()

        self.ff_nn1 = Dense(ff_dim, activation='relu')
        self.ff_nn2 = Dense(embed_dim)
        self.ff_nn3 = Dense(vocab_size, activation='softmax')
        self.embed = PositionalEmbedding(embed_dim=embed_dim, vocab_size=vocabulary_size, seq_len=20)
        self.masking = True
        self.add_layer1 = tf.keras.layers.Add()  # Add residual connection layer
        self.add_layer2 = tf.keras.layers.Add()  # Add residual connection layer

    def call(self, inputs, encoder_out, training, mask=None):
        inputs = self.embed(inputs)
        causal_mask = self.causal_attn_mask(inputs)
        combo_mask = None
        pad_mask = None
        if mask is not None:
            pad_mask = tf.cast(mask[:, :, tf.newaxis], tf.int32) 
            combo_mask = tf.cast(mask[:, :, tf.newaxis], tf.int32)   
            combo_mask = tf.minimum(combo_mask, causal_mask)
            
        attn_out1 = self.attn1(query=inputs, value=inputs, key=inputs, attention_mask=combo_mask, training=training)
        out1 = self.layernorm1(attn_out1 + inputs)
        attn_out2 = self.attn2(query=out1, value=encoder_out, key=encoder_out, attention_mask=pad_mask, training=training)
        out2 = self.layernorm2(attn_out2 + out1)
        
        ff_out1 = self.ff_nn1(out2)
        ff_out2 = self.ff_nn2(ff_out1)
        ff_out2 = self.layer_norm3(ff_out2 + out2)
        final_out = self.ff_nn3(ff_out2)
        return final_out
    def causal_attn_mask(self, inputs):
        shape = tf.shape(inputs)
        batch_size, seq_len = shape[0], shape[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        multiply = tf.concat([
            tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32)
        ], axis=0)
        return tf.tile(mask, multiply)        


class VideoCaptioningModel(Model):
    def __init__(
        self,
        conv_lstm_extractor,
        transformer_encoder,
        transformer_decoder,
        num_captions_per_video=5,
    ):
        super().__init__()
        self.conv_lstm_extractor = conv_lstm_extractor
        self.encoder = transformer_encoder
        self.decoder = transformer_decoder
        self.loss_tracker = tf.metrics.Mean(name="loss")
        self.acc_tracker = tf.metrics.Mean(name="accuracy")
        self.num_captions_per_video = num_captions_per_video
        

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.compiled_loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        # Ensure y_true is cast to int32 to match the output type of tf.argmax, which is int64 by default
        y_true = tf.cast(y_true, dtype=tf.int64)
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def train_step(self, batch_data):
        batch_frames, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get video embeddings from ConvLSTM and flatten
        video_features = self.conv_lstm_extractor(batch_frames)  
        # 2. Process the captions
        for i in range(self.num_captions_per_video):
            with tf.GradientTape() as tape:
                encoder_out = self.encoder(video_features, training=True)
                batch_seq_inp = batch_seq[:, i, :-1]
                batch_seq_true = batch_seq[:, i, 1:]
                mask = tf.math.not_equal(batch_seq_true, 0)
                batch_seq_pred = self.decoder(
                    batch_seq_inp, encoder_out, training=True, mask=mask
                )
                loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
                acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
                
                # Accumulate loss and accuracy
                batch_loss += loss
                batch_acc += acc
            
            # Gradient update steps with gradient clipping
            train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
            grads = tape.gradient(loss, train_vars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.5)  # Gradient clipping
            grads = [tf.clip_by_value(g, -1.5, 1.5) for g in grads]  # Value clipping
            self.optimizer.apply_gradients(zip(grads, train_vars))
        
        loss = batch_loss
        batch_acc /= float(self.num_captions_per_video)
        
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(batch_acc)
        
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }
        
        
    # Include test_step and metrics as in your initial structure
    def test_step(self, batch_data):
        batch_frames, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        video_features = self.conv_lstm_extractor(batch_frames)
        
        for i in range(self.num_captions_per_video):
            encoder_out = self.encoder(video_features, training=False)
            batch_seq_inp = batch_seq[:, i, :-1]
            batch_seq_true = batch_seq[:, i, 1:]
            mask = tf.math.not_equal(batch_seq_true, 0)
            batch_seq_pred = self.decoder(
                batch_seq_inp, encoder_out, training=False, mask=mask
            )
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

            batch_loss += caption_loss
            batch_acc += caption_acc
        
        loss = batch_loss
        batch_acc /= float(self.num_captions_per_video)
        
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(batch_acc)
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }
    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

# Define Model
ex_model = conv_lstm_extractor()

encoder = TransformerEncoder(embed_dim=1024, dense_dim=2048, num_heads=2)

decoder = TransformerDecoder(embed_dim=1024, ff_dim=2048, num_heads=3, vocab_size=vocabulary_size)

caption_model = VideoCaptioningModel(ex_model, encoder, decoder)

# Learning Rate Warmup with Cosine Decay
initial_learning_rate = 0.001
warmup_epochs = 20
total_epochs = 45

# Calculate the number of warmup steps
warmup_steps = warmup_epochs * len(train_frames) // 24

# Define the learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, total_epochs - warmup_epochs)

# Define the optimizer with weight decay
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-6)

# Compile the model
caption_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Fit the model
history = caption_model.fit(
    x=train_frames,
    y=train_captions,
    validation_data=(val_frames, val_captions),
    batch_size=64,
    epochs=total_epochs,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ModelCheckpoint('caption_model.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1),
        TensorBoard(log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, write_graph=True, write_images=True)
    ]
)

# Keras Tuner
from keras_tuner import RandomSearch, HyperParameters

class MyHyperModel(HyperModel):
    def build(self, hp):
        ex_model = conv_lstm_extractor()
        encoder = TransformerEncoder(
            embed_dim=hp.Int('embed_dim', 256, 512, step=64),
            dense_dim=hp.Int('dense_dim', 1024, 4096, step=1024),
            num_heads=hp.Int('num_heads', 1, 4, step=1)
        )
        decoder = TransformerDecoder(
            embed_dim=hp.Int('embed_dim', 256, 512, step=64),
            ff_dim=hp.Int('ff_dim', 1024, 4096, step=1024),
            num_heads=hp.Int('num_heads', 1, 4, step=1),
            vocab_size=vocabulary_size
        )
        model = VideoCaptioningModel(ex_model, encoder, decoder)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'),
                weight_decay=hp.Float('weight_decay', 1e-5, 1e-3, sampling='log')
            ),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
                       tf.keras.callbacks.ModelCheckpoint('caption_model.h5', save_best_only=True, save_weights_only=True, monitor='val_loss'),
                       TensorBoard(log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, write_graph=True, write_images=True)]
        )

tuner = RandomSearch(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='hparam_tuning'
)

tuner.search(train_frames, train_captions, validation_data=(val_frames, val_captions), epochs=10, batch_size=32)
