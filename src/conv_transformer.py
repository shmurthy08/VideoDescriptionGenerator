import numpy as np
import pickle as pkl
import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model, Sequential
from preprocessing import ClipsCaptions
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    ConvLSTM2D, Input, Conv3D, LayerNormalization, Conv2D, Flatten, Dense,
    Embedding, BatchNormalization, Lambda, MultiHeadAttention, GlobalAveragePooling2D,
    Reshape, TimeDistributed, Add, Dropout
)

        

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
def preprocess_captions(captions, word_index, max_length=10):
    """
    Preprocess list of captions for training: Lowercase, tokenize using word_index, and pad sequences.

    Args:
    captions: List of list of captions (2D list where each sublist contains captions for one video).
    word_index: Dictionary mapping words to indices.
    max_length: Maximum length of tokenized caption sequences. Defaults to 10.

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


# Parameters
num_frames = 5
frame_height = 224
frame_width = 224
num_channels = 3

epoch_num = 75
batch_size = 32


# Call ConvLSTM Feature Extraction Model
def conv_lstm_extractor():
    fe_model = load_model('Feature_extract.h5')

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
    '''
    Positional Encoding for Video Frames to preserve spatio-temporal information.
    '''
    def __init__(self, seq_len, embed_dim, **kwargs):
        '''
        Initialize the layer with sequence length and embedding dimensions.
        
        Args:
        seq_len: Number of frames in the video sequence.
        embed_dim: Dimension of the embedding vector.
        '''
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        '''
        Build the positional encoding layer.
        
        Args:
        input_shape: Shape of the input tensor.
        '''
        self.pos_encoding = self.add_weight("positional_encoding", 
                                            shape=(1, self.seq_len, self.embed_dim),
                                            initializer=self.positional_encoding_initializer,
                                            trainable=False)
        
    def call(self, inputs):
        '''
        Call function to apply positional encoding to the input tensor.
        
        Args:
        inputs: Input tensor of shape (batch_size, seq_len, embed_dim).
        '''
        batch_size = tf.shape(inputs)[0]
        return inputs + tf.broadcast_to(self.pos_encoding, [batch_size, self.seq_len, self.embed_dim])
    
    def positional_encoding_initializer(self, shape, dtype=None):
        '''
        Custom initializer for positional encoding.
        
        Args:
        shape: Shape of the tensor to initialize.
        dtype: Data type of the tensor.
        '''
        
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
        '''
        Get the angles for the positional encoding.
        Ensures uniformity in even and odd indices.
        
        Args:
        pos: Positional indices in the sequence.
        i: Embedding indices.
        d_model: Embedding dimension.
        '''
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


class TransformerEncoder(tf.keras.layers.Layer):
    '''
    Class to Create/Initialize Transformer Encoder.
    Skeleton structure based off of Keras Guide: https://github.com/keras-team/keras-io/blob/master/examples/vision/image_captioning.py
    '''
    def __init__(self, embed_dim, num_heads, drop_rate, **kwargs):
        '''
        Initializes the Transformer Encoder Layer.

        Args:
            embed_dim (int): The dimensionality of the embedding.
            num_heads (int): The number of attention heads.
            drop_rate (float): The dropout rate.
            **kwargs: Additional keyword arguments.

        '''
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
        '''
        Executes the forward pass of the Transformer Encoder Layer.

        Args:
            inputs: The input tensor.
            training: A boolean indicating whether the model is in training mode or not.

        Returns:
            The output tensor after the forward pass.
        '''
        inputs = self.dense(inputs)
        inputs = self.drop1(inputs, training=training)
        inputs = self.layernorm1(inputs)
        inputs = self.vidpos(inputs)
        
        attn_out = self.attn1(query=inputs, value=inputs, key=inputs, attention_mask=None, training=training)
        attn_out = self.drop2(attn_out, training=training)
        attn_out = self.layernorm2(attn_out + inputs)
        
        attn_out2 = self.attn2(query=attn_out, value=attn_out, key=attn_out, attention_mask=None, training=training)
        attn_out2 = self.drop3(attn_out2, training=training)
        attn_out2 = self.layernorm3(attn_out2 + attn_out)
        
        attn_out3 = self.attn3(query=attn_out2, value=attn_out2, key=attn_out2, attention_mask=None, training=training)
        attn_out3 = self.drop4(attn_out3, training=training)
        attn_out3 = self.layernorm4(attn_out3 + attn_out2)
        
        attn_out4 = self.attn4(query=attn_out3, value=attn_out3, key=attn_out3, attention_mask=None, training=training)
        attn_out4 = self.drop5(attn_out4, training=training)
        attn_out4 = self.layernorm5(attn_out4 + attn_out3)
        
        attn_out5 = self.attn5(query=attn_out4, value=attn_out4, key=attn_out4, attention_mask=None, training=training)
        attn_out5 = self.drop6(attn_out5, training=training)
        attn_out5 = self.layernorm6(attn_out5 + attn_out4)
        
        attn_out6 = self.attn6(query=attn_out5, value=attn_out5, key=attn_out5, attention_mask=None, training=training)
        attn_out6 = self.drop7(attn_out6, training=training)
        attn_out6 = self.layernorm7(attn_out6 + attn_out5)

        return attn_out6
    

class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Positional embedding layer for Transformer models.
    
    This layer combines token embeddings and positional embeddings to create
    the input embeddings for a Transformer model. It takes token indices as input
    and returns the corresponding embeddings with positional information.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimensionality of the embedding vectors.
        seq_len (int): The maximum sequence length.
    
    Attributes:
        token_emb (tf.keras.layers.Embedding): The token embedding layer.
        pos_emb (tf.keras.layers.Embedding): The positional embedding layer.
        scale (tf.Tensor): The scaling factor for the embeddings.
    """
    
    def __init__(self, vocab_size, embed_dim, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(vocab_size, embed_dim)
        self.scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))
        
    def call(self, inputs):
        """
        Compute the input embeddings with positional information.
        
        Args:
            inputs (tf.Tensor): The input tensor of token indices.
        
        Returns:
            tf.Tensor: The input embeddings with positional information.
        """
        length = tf.shape(inputs)[-1]
        pos = tf.range(start=0, limit=length, delta=1)
        emb_tokens = self.token_emb(inputs)
        emb_tokens = emb_tokens * self.scale
        emb_pos = self.pos_emb(pos)
        return emb_tokens + emb_pos
    
    def mask(self, inputs, mask=None):
        """
        Create a mask tensor based on the input tensor.
        
        Args:
            inputs (tf.Tensor): The input tensor.
            mask (tf.Tensor, optional): An optional mask tensor.
        
        Returns:
            tf.Tensor: The mask tensor.
        """
        return tf.math.not_equal(inputs, 0)
    
    
class TransformerDecoder(tf.keras.layers.Layer):
    """
    Transformer Decoder layer for the Video Description Generator model.
    
    Args:
        embed_dim (int): The dimensionality of the embedding.
        ff_dim (int): The dimensionality of the feed-forward layer.
        num_heads (int): The number of attention heads.
        vocab_size (int): The size of the vocabulary.
        drop_rate (float): The dropout rate.
    
    Attributes:
        attn1 (MultiHeadAttention): The first multi-head attention layer.
        attn2 (MultiHeadAttention): The second multi-head attention layer.
        attn3 (MultiHeadAttention): The third multi-head attention layer.
        attn4 (MultiHeadAttention): The fourth multi-head attention layer.
        attn5 (MultiHeadAttention): The fifth multi-head attention layer.
        attn6 (MultiHeadAttention): The sixth multi-head attention layer.
        layernorm0 (LayerNormalization): The first layer normalization layer.
        layernorm1 (LayerNormalization): The second layer normalization layer.
        layernorm2 (LayerNormalization): The third layer normalization layer.
        layernorm3 (LayerNormalization): The fourth layer normalization layer.
        layernorm4 (LayerNormalization): The fifth layer normalization layer.
        layernorm5 (LayerNormalization): The sixth layer normalization layer.
        layernorm6 (LayerNormalization): The seventh layer normalization layer.
        layernorm7 (LayerNormalization): The eighth layer normalization layer.
        drop1 (Dropout): The first dropout layer.
        drop2 (Dropout): The second dropout layer.
        drop3 (Dropout): The third dropout layer.
        drop4 (Dropout): The fourth dropout layer.
        drop5 (Dropout): The fifth dropout layer.
        drop6 (Dropout): The sixth dropout layer.
        drop7 (Dropout): The seventh dropout layer.
        ff_nn1 (Dense): The first feed-forward neural network layer.
        ff_nn2 (Dense): The second feed-forward neural network layer.
        ff_nn3 (Dense): The third feed-forward neural network layer.
        embed (PositionalEmbedding): The positional embedding layer.
        masking (bool): Whether to apply masking during training.
    """
    
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
        self.embed = PositionalEmbedding(embed_dim=embed_dim, vocab_size=vocabulary_size, seq_len=10)
        self.masking = True
    
    def call(self, inputs, encoder_out, training, mask=None):
        """
        Perform a forward pass through the Transformer Decoder layer.
        
        Args:
            inputs (tf.Tensor): The input tensor.
            encoder_out (tf.Tensor): The output tensor from the encoder.
            training (bool): Whether the model is in training mode or not.
            mask (tf.Tensor, optional): The mask tensor. Defaults to None.
        
        Returns:
            tf.Tensor: The final output tensor.
        """
        inputs = self.embed(inputs)
        inputs = self.drop1(inputs, training=training)
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
        attn_out1 = self.drop2(attn_out1, training=training)
        out1 = self.layernorm1(attn_out1 + inputs)
        
        attn_out2 = self.attn2(query=out1, value=encoder_out, key=encoder_out, attention_mask=pad_mask, 
                               training=training)
        attn_out2 = self.drop3(attn_out2, training=training)
        out2 = self.layernorm2(attn_out2 + out1)
        
        attn_out3 = self.attn3(query=out2, value=out2, key=out2, attention_mask=pad_mask,
                                training=training)
        attn_out3 = self.drop4(attn_out3, training=training)
        out3 = self.layernorm3(attn_out3 + out2)
        
        attn_out4 = self.attn4(query=out3, value=out3, key=out3, attention_mask=pad_mask,
                                training=training)
        attn_out4 = self.drop5(attn_out4, training=training)
        out4 = self.layernorm4(attn_out4 + out3)
        
        attn_out5 = self.attn5(query=out4, value=out4, key=out4, attention_mask=pad_mask,
                                training=training)
        attn_out5 = self.drop6(attn_out5, training=training)
        out5 = self.layernorm5(attn_out5 + out4)
        
        attn_out6 = self.attn6(query=out5, value=out5, key=out5, attention_mask=pad_mask,
                                training=training)
        attn_out6 = self.drop7(attn_out6, training=training)
        out6 = self.layernorm6(attn_out6 + out5)
        
        ff_out1 = self.ff_nn1(out6)
        ff_out2 = self.ff_nn2(ff_out1)
        ff_out2 = self.layernorm7(ff_out2 + out6)
        final_out = self.ff_nn3(ff_out2)
        
        return final_out
    
    def causal_attn_mask(self, inputs):
        """
        Generate a causal attention mask.
        
        Args:
            inputs (tf.Tensor): The input tensor.
        
        Returns:
            tf.Tensor: The causal attention mask.
        """
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



class VideoCaptioningModel(tf.keras.Model):
    '''
    Class to Create/Initialize Video Captioning Model.
    
    Args:
        conv_lstm_extractor (tf.keras.Model): The ConvLSTM feature extractor model.
        transformer_encoder (tf.keras.Model): The Transformer encoder model.
        transformer_decoder (tf.keras.Model): The Transformer decoder model.
        num_captions_per_video (int): The number of captions per video.
        
    Attributes:
        conv_lstm_extractor (tf.keras.Model): The ConvLSTM feature extractor model.
        encoder (tf.keras.Model): The Transformer encoder model.
        decoder (tf.keras.Model): The Transformer decoder model.
        loss_tracker (tf.keras.metrics.Mean): The loss tracker.
        acc_tracker (tf.keras.metrics.Mean): The accuracy tracker.
        correct_order_tracker (tf.keras.metrics.Mean): The correct order tracker.
        total_words_tracker (tf.keras.metrics.Mean): The total words tracker.
        two_gram_overlap_tracker (tf.keras.metrics.Mean): The two-gram overlap tracker.
        num_captions_per_video (int): The number of captions per video.
    '''
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
        self.correct_order_tracker = tf.metrics.Mean(name="correct_order_words")
        self.total_words_tracker = tf.metrics.Mean(name="total_words")
        self.two_gram_overlap_tracker = tf.metrics.Mean(name="two_gram_overlap")
        self.num_captions_per_video = num_captions_per_video

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.compiled_loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_pred_indices = tf.argmax(y_pred, axis=2)
        accuracy = tf.equal(y_true, y_pred_indices)
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def calculate_correct_order_words(self, y_true, y_pred, mask):
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_pred_indices = tf.argmax(y_pred, axis=2)
        correct_order = tf.equal(y_true, y_pred_indices)
        correct_order = tf.math.logical_and(mask, correct_order)
        correct_order = tf.cast(correct_order, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        correct_order_sum = tf.reduce_sum(correct_order)
        total_words = tf.reduce_sum(mask)
        return correct_order_sum, total_words

    def calculate_2gram_overlap(self, y_true, y_pred, mask):
        y_pred_indices = tf.argmax(y_pred, axis=-1)
        true_ngrams = []
        pred_ngrams = []

        for i in range(self.num_captions_per_video):
            valid_true = tf.boolean_mask(y_true[:, i], mask[:, i])
            valid_pred = tf.boolean_mask(y_pred_indices[:, i], mask[:, i])

            true_strings = tf.strings.as_string(valid_true)
            pred_strings = tf.strings.as_string(valid_pred)

            true_2grams = tf.strings.ngrams(true_strings, ngram_width=2, separator=' ')
            pred_2grams = tf.strings.ngrams(pred_strings, ngram_width=2, separator=' ')

            true_ngrams.append(true_2grams)
            pred_ngrams.append(pred_2grams)

        true_ngrams_flat = tf.concat(true_ngrams, axis=0)
        pred_ngrams_flat = tf.concat(pred_ngrams, axis=0)

        intersection = tf.sets.intersection(
            tf.expand_dims(true_ngrams_flat, 0),
            tf.expand_dims(pred_ngrams_flat, 0)
        )
        union = tf.sets.union(
            tf.expand_dims(true_ngrams_flat, 0),
            tf.expand_dims(pred_ngrams_flat, 0)
        )

        overlap = tf.size(intersection.values) / tf.size(union.values)
        return overlap

    def train_step(self, batch_data):
        batch_frames, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

    
        # Get video features from conv_lstm_extractor
        video_features = self.conv_lstm_extractor(batch_frames)  
        # Process the captions
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
        batch_two_gram_overlap = 0
        batch_correct_order = 0
        batch_total_words = 0

        video_features = self.conv_lstm_extractor(batch_frames, training=False)

        for i in range(self.num_captions_per_video):
            encoder_out = self.encoder(video_features, training=False)
            batch_seq_inp = batch_seq[:, i, :-1]
            batch_seq_true = batch_seq[:, i, 1:]
            mask = tf.math.not_equal(batch_seq_true, 0)
            batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=False, mask=mask)
            
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
            caption_overlap = self.calculate_2gram_overlap(batch_seq_true, batch_seq_pred, mask)
            correct_order, total_words = self.calculate_correct_order_words(batch_seq_true, batch_seq_pred, mask)
            batch_correct_order += correct_order
            batch_total_words += total_words
            batch_two_gram_overlap += caption_overlap
            batch_loss += caption_loss
            batch_acc += caption_acc

            
        loss = batch_loss
        batch_acc /= float(self.num_captions_per_video)
        batch_correct_order /= float(self.num_captions_per_video)
        batch_total_words /= float(self.num_captions_per_video)
        batch_two_gram_overlap /= float(self.num_captions_per_video)
        

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(batch_acc)
        self.correct_order_tracker.update_state(batch_correct_order)
        self.total_words_tracker.update_state(batch_total_words)
        self.two_gram_overlap_tracker.update_state(batch_two_gram_overlap)

        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
            "correct_order_words": self.correct_order_tracker.result(),  
            "total_words": self.total_words_tracker.result(),
            "two_gram_overlap": self.two_gram_overlap_tracker.result()

        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker, self.correct_order_tracker, self.total_words_tracker, self.two_gram_overlap_tracker]
    
   
    def get_config(self):
        return {
            "conv_lstm_extractor": tf.keras.utils.serialize_keras_object(self.conv_lstm_extractor),
            "transformer_encoder": tf.keras.utils.serialize_keras_object(self.encoder),
            "transformer_decoder": tf.keras.utils.serialize_keras_object(self.decoder),
            "num_captions_per_video": self.num_captions_per_video
        }

    @classmethod
    def from_config(cls, config):
        # Here, you need to ensure that the objects are correctly instantiated from their configs
        conv_lstm_extractor = tf.keras.utils.deserialize_keras_object(config['conv_lstm_extractor'])
        transformer_encoder = tf.keras.utils.deserialize_keras_object(config['transformer_encoder'])
        transformer_decoder = tf.keras.utils.deserialize_keras_object(config['transformer_decoder'])
        num_captions_per_video = config['num_captions_per_video']
        
        return cls(conv_lstm_extractor, transformer_encoder, transformer_decoder, num_captions_per_video)



# Learning Rate Scheduler for the optimizer
class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule for a custom optimizer.

    Args:
        post_warmup_learning_rate (float): The learning rate after the warm-up phase.
        warmup_steps (int): The number of steps for the warm-up phase.
        total_steps (int): The total number of steps.

    Returns:
        float: The learning rate at the given step.

    """
    def __init__(self, post_warmup_learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        '''
        Call function to calculate the learning rate at a given step.
        
        Args:
        step: The current step in the training process.
        '''
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_progress = global_step / warmup_steps
        linear_warmup = self.post_warmup_learning_rate * warmup_progress
        
        cosine_decay = 0.5 * (1 + tf.cos((global_step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        post_warmup_lr = self.post_warmup_learning_rate * cosine_decay
        
        # Use a tf.cond to handle the different LR schedules during warm-up and after
        return tf.cond(
            global_step < warmup_steps,
            lambda: linear_warmup,
            lambda: post_warmup_lr
        )



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
output_shapes = (tf.TensorShape([num_frames, frame_height, frame_width, num_channels]), tf.TensorShape([5, 10]))

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



# Define Model
# Create a learning rate schedule
num_train_steps = (len(train_dataset) // batch_size) * epoch_num
num_warmup_steps = num_train_steps // batch_size

ex_model = conv_lstm_extractor()

encoder = TransformerEncoder(embed_dim=1408, num_heads=5, drop_rate=0.1)

decoder = TransformerDecoder(embed_dim=1408, ff_dim=3584, num_heads=5, vocab_size=vocabulary_size, drop_rate=0.1)

caption_model = VideoCaptioningModel(ex_model, encoder, decoder)
caption_model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        LRSchedule(
            post_warmup_learning_rate=1e-4,
            warmup_steps=num_warmup_steps,
            total_steps=num_train_steps
        ),
        weight_decay=1e-4
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = caption_model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
               tf.keras.callbacks.ModelCheckpoint('caption_model_weights.h5', save_weights_only=True),
               TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)],
    
)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('caption_model_accuracy_plot.png')  

