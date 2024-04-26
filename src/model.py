import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Dropout, Input
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Define ConvLSTM Model
def build_convlstm_model(input_shape=(10, 224, 224, 3), lstm_units=256, dropout_rate=0.5):
    model = Sequential([
        ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
        Dropout(dropout_rate),
        Flatten(),
        Dense(lstm_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(input_shape[0]*lstm_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1024, activation='relu')  # Embedding size for GPT-2
    ])
    return model

# Load pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
gpt2_model = TFGPT2LMHeadModel.from_pretrained("gpt2-medium")

# Freeze GPT-2 layers
for layer in gpt2_model.layers:
    layer.trainable = False

# Define combined model
input_layer = Input(shape=(10, 224, 224, 3))
convlstm_output = build_convlstm_model()(input_layer)
gpt2_output = gpt2_model(convlstm_output)[0]
caption_model = Model(inputs=input_layer, outputs=gpt2_output)

caption_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

caption_model.summary()
