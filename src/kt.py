from conv_transformer import TransformerEncoder, TransformerDecoder, VideoCaptioningModel, def_conv_lstm_extractor, LRSchedule
import pickle as pkl
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import TensorBoard
import kerastuner as kt


with open('word_index.pkl', 'rb') as f:
    word_index = pkl.load(f)

# Add <unk> token to word_index dict
word_index['<unk>'] = len(word_index) + 1

with open('index_word.pkl', 'rb') as f:
    index_word = pkl.load(f)

# Add <unk> token to index_word
index_word[len(word_index)+1] = '<unk>'    

vocabulary_size = len(word_index) + 1

with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pkl.load(f)

with open('val_dataset.pkl', 'rb') as f:
    val_dataset = pkl.load(f)

# Preprocess Video Frames
def preprocess_frames(frames):
    '''
    For each video: normalize pixel values.
    '''
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

# Prepare training data
train_captions_data = [sample.captions for sample in train_dataset]
train_captions = preprocess_captions(train_captions_data, word_index)
train_frames = np.array([preprocess_frames(sample.frames) for sample in train_dataset])

# Prepare validation data
val_captions_data = [sample.captions for sample in val_dataset]
val_captions = preprocess_captions(val_captions_data, word_index)
val_frames = np.array([preprocess_frames(sample.frames) for sample in val_dataset])


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
output_shapes = (tf.TensorShape([num_frames, frame_height, frame_width, num_channels]), tf.TensorShape([5, 20]))

# Create the dataset from the generator
train_data = tf.data.Dataset.from_generator(
    train_generator,
    output_types=output_types,
    output_shapes=output_shapes
).batch(batch_size)
val_data = tf.data.Dataset.from_generator(
    val_generator,
    output_types=output_types,
    output_shapes=output_shapes
).batch(batch_size)


# Prefetch
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)



class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Instantiate models and compile without callbacks
        ex_model = conv_lstm_extractor()
        encoder = TransformerEncoder(
            embed_dim=hp.Int('embed_dim', 256, 2048, step=128),
            num_heads=hp.Int('num_heads', 2, 5, step=1),
            drop_rate=hp.Float('drop_rate', 0.1, 0.5, step=0.1)
        )
        decoder = TransformerDecoder(
            embed_dim=hp.Int('embed_dim', 256, 2048, step=128),
            ff_dim=hp.Int('ff_dim', 512, 4096, step=1024),
            num_heads=hp.Int('num_heads', 2, 5, step=1),
            vocab_size=vocabulary_size,
            drop_rate=hp.Float('drop_rate', 0.1, 0.5, step=0.1)
        )
        model = VideoCaptioningModel(ex_model, encoder, decoder)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                LRSchedule(
                    post_warmup_learning_rate=hp.Choice('post_warmup_learning_rate', [1e-3, 1e-4, 1e-5, 1e-6]),
                    warmup_steps=num_warmup_steps,
                    total_steps=num_train_steps
                ),
                weight_decay=hp.Choice('weight_decay', [1e-3, 1e-4, 1e-5, 1e-6])
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )


tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective=kt.Objective('val_loss', "min"),
    executions_per_trial=1,
    directory='bayesian_dir',
    project_name='fixed_multilayer_hparam_tuning_acc'
)


tuner.search(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss', verbose=1),
        TensorBoard(log_dir='logs/hparam_tuning/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, write_graph=True, write_images=True)
    ],
  
)