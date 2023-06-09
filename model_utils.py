import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_model(input_shape=(224, 224, 3)):
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add custom classifier on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)
