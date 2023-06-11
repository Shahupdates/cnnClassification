import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

def create_model(input_shape=(224, 224, 3), num_classes=10, num_nodes=256, dropout_rate=0.5, unfreeze_layers=5):
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Unfreeze the top layers of ResNet
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    # Add custom classifier on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_nodes, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)
