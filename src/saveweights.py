from tensorflow.keras.models import load_model

# Save weights
model = load_model('Feature_extract.h5')
model.save_weights('Feature_extract_weights.h5')