# image_classifier.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageClassifier:
    def __init__(self):
        self.model = None

    def train(self, train_dir, test_dir, batch_size=32, epochs=10):
        input_shape = (224, 224, 3)

        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Load pre-trained ResNet50 model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        # Add custom classifier on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)

        # Create the complete model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model with fine-tuning
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=test_generator.n // batch_size
        )

    def save_model(self, model_path):
        if self.model:
            self.model.save(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
