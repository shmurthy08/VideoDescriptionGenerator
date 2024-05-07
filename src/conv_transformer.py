import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Input, Conv3D, LayerNormalization, Conv2D, Flatten, Dense, Embedding, BatchNormalization, Lambda, MultiHeadAttention, GlobalAveragePooling2D, Reshape, TimeDistributed, Add, Dropout
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
import keras_tuner as kt

        

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
    feature_output = fe_model.layers[-13].output # Extract features after Encoder is complete
    # Flatten with TimeDistributed
    feature_output = TimeDistributed(Flatten())(feature_output)
    extract = Model(inputs=fe_model.input, outputs=feature_output)
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
    def __init__(self, embed_dim, num_heads, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attn1 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn2 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn3 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn4 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn5 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn6 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.layernorm4 = LayerNormalization()
        self.layernorm5 = LayerNormalization()
        self.layernorm6 = LayerNormalization()
        self.layernorm7 = LayerNormalization()
        
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        self.drop3 = tf.keras.layers.Dropout(drop_rate)
        self.drop4 = tf.keras.layers.Dropout(drop_rate)
        self.drop5 = tf.keras.layers.Dropout(drop_rate)
        self.drop6 = tf.keras.layers.Dropout(drop_rate)
        self.drop7 = tf.keras.layers.Dropout(drop_rate)
        
        self.vidpos = VideoPositionalEmbedding(seq_len=5, embed_dim=embed_dim)
        self.dense = Dense(embed_dim, activation='relu')

    def call(self, inputs, training):
        inputs = self.dense(inputs)
        inputs = self.drop1(inputs)
        inputs = self.layernorm1(inputs)
        inputs = self.vidpos(inputs)
        
        attn_out = self.attn1(query=inputs, value=inputs, key=inputs, attention_mask=None, training=training)
        attn_out = self.drop2(attn_out)
        attn_out = self.layernorm2(attn_out + inputs)
        
        attn_out2 = self.attn2(query=attn_out, value=attn_out, key=attn_out, attention_mask=None, training=training)
        attn_out2 = self.drop3(attn_out2)
        attn_out2 = self.layernorm3(attn_out2 + attn_out)
        
        attn_out3 = self.attn3(query=attn_out2, value=attn_out2, key=attn_out2, attention_mask=None, training=training)
        attn_out3 = self.drop4(attn_out3)
        attn_out3 = self.layernorm4(attn_out3 + attn_out2)
        
        attn_out4 = self.attn4(query=attn_out3, value=attn_out3, key=attn_out3, attention_mask=None, training=training)
        attn_out4 = self.drop5(attn_out4)
        attn_out4 = self.layernorm5(attn_out4 + attn_out3)
        
        attn_out5 = self.attn5(query=attn_out4, value=attn_out4, key=attn_out4, attention_mask=None, training=training)
        attn_out5 = self.drop6(attn_out5)
        attn_out5 = self.layernorm6(attn_out5 + attn_out4)
        
        attn_out6 = self.attn6(query=attn_out5, value=attn_out5, key=attn_out5, attention_mask=None, training=training)
        attn_out6 = self.drop7(attn_out6)
        attn_out6 = self.layernorm7(attn_out6 + attn_out5)

        return attn_out6
    


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
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        
        self.attn1 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn2 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn3 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn4 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn5 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        self.attn6 = MultiHeadAttention(num_heads, embed_dim, dropout=drop_rate)
        
        self.layernorm0 = LayerNormalization()
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.layernorm4 = LayerNormalization()
        self.layernorm5 = LayerNormalization()
        self.layernorm6 = LayerNormalization()
        self.layernorm7 = LayerNormalization()
        
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        self.drop3 = tf.keras.layers.Dropout(drop_rate)
        self.drop4 = tf.keras.layers.Dropout(drop_rate)
        self.drop5 = tf.keras.layers.Dropout(drop_rate)
        self.drop6 = tf.keras.layers.Dropout(drop_rate)
        self.drop7 = tf.keras.layers.Dropout(drop_rate)
        
        self.ff_nn1 = Dense(ff_dim, activation='relu')
        self.ff_nn2 = Dense(embed_dim)
        self.ff_nn3 = Dense(vocab_size, activation='softmax')
        self.embed = PositionalEmbedding(embed_dim=embed_dim, vocab_size=vocabulary_size, seq_len=20)
        self.masking = True
    def call(self, inputs, encoder_out, training, mask=None):
        inputs = self.embed(inputs)
        inputs = self.drop1(inputs)
        inputs = self.layernorm0(inputs)
        causal_mask = self.causal_attn_mask(inputs)
        combo_mask = None
        pad_mask = None
        if mask is not None:
            pad_mask = tf.cast(mask[:, :, tf.newaxis], tf.int32) 
            combo_mask = tf.cast(mask[:, :, tf.newaxis], tf.int32)   
            combo_mask = tf.minimum(combo_mask, causal_mask)
            
        attn_out1 = self.attn1(query=inputs, value=inputs, key=inputs, attention_mask=combo_mask,  
                               training=training)
        attn_out1 = self.drop2(attn_out1)
        out1 = self.layernorm1(attn_out1 + inputs)
        
        attn_out2 = self.attn2(query=out1, value=encoder_out, key=encoder_out, attention_mask=pad_mask, 
                               training=training)
        attn_out2 = self.drop3(attn_out2)
        out2 = self.layernorm2(attn_out2 + out1)
        
        attn_out3 = self.attn3(query=out2, value=out2, key=out2, attention_mask=pad_mask,
                                training=training)
        attn_out3 = self.drop4(attn_out3)
        out3 = self.layernorm3(attn_out3 + out2)
        
        attn_out4 = self.attn4(query=out3, value=out3, key=out3, attention_mask=pad_mask,
                                training=training)
        attn_out4 = self.drop5(attn_out4)
        out4 = self.layernorm4(attn_out4 + out3)
        
        attn_out5 = self.attn5(query=out4, value=out4, key=out4, attention_mask=pad_mask,
                                training=training)
        attn_out5 = self.drop6(attn_out5)
        out5 = self.layernorm5(attn_out5 + out4)
        
        attn_out6 = self.attn6(query=out5, value=out5, key=out5, attention_mask=pad_mask,
                                training=training)
        attn_out6 = self.drop7(attn_out6)
        out6 = self.layernorm6(attn_out6 + out5)
        

        
        ff_out1 = self.ff_nn1(out6)
        ff_out2 = self.ff_nn2(ff_out1)
        ff_out2 = self.layernorm7(ff_out2 + out6)
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
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.5)  # Gradient norm clipping
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

# Learning Rate Scheduler for the optimizer
class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_progress = global_step / warmup_steps
        linear_warmup = self.post_warmup_learning_rate * warmup_progress
        
        cosine_decay = 0.5 * (1 + tf.cos((global_step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        post_warmup_lr = self.post_warmup_learning_rate * cosine_decay
        
        return tf.cond(
            global_step < warmup_steps,
            lambda: linear_warmup,
            lambda: post_warmup_lr
        )

# Create a learning rate schedule
num_train_steps = (len(train_dataset) // 32) * 75
num_warmup_steps = num_train_steps // 32

def train_generator():
    '''
    Generator function to yield training data
    '''
    for features, labels in zip(train_frames, train_captions):
        yield features, labels

def val_generator():
    '''
    Generator function to yield validation data
    '''
    for features, labels in zip(val_frames, val_captions):
        yield features, labels

# Define output types and shapes to match actual data
output_types = (tf.float32, tf.int32)
output_shapes = (tf.TensorShape([5, 224, 224, 3]), tf.TensorShape([5, 20]))

# Create the dataset from the generator
train_data = tf.data.Dataset.from_generator(
    train_generator,
    output_types=output_types,
    output_shapes=output_shapes
).batch(32)
val_data = tf.data.Dataset.from_generator(
    val_generator,
    output_types=output_types,
    output_shapes=output_shapes
).batch(32)


# Prefetch
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)


# Define Model
ex_model = conv_lstm_extractor()

encoder = TransformerEncoder(embed_dim=1152, num_heads=3, drop_rate=0.1)

decoder = TransformerDecoder(embed_dim=1152, ff_dim=1536, num_heads=3, vocab_size=vocabulary_size, drop_rate=0.1)

caption_model = VideoCaptioningModel(ex_model, encoder, decoder)
caption_model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        LRSchedule(
            post_warmup_learning_rate=1e-4,
            warmup_steps=num_warmup_steps,
            total_steps=num_train_steps
        ),
        weight_decay=1e-5
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = caption_model.fit(
    train_data,
    validation_data=val_data,
    epochs=75,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=9, restore_best_weights=False, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ModelCheckpoint('caption_model_weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1),
        TensorBoard(log_dir='logs/caption_final_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, write_graph=True, write_images=True)
    ],
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')  # Save the plot as a PNG image
plt.show()