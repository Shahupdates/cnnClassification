# image_classifier.py

from preprocessing import get_image_data_generator
from evaluation import evaluate_model, plot_confusion_matrix, plot_learning_curve
from model_utils import create_model, save_model, load_model

class ImageClassifier:
    def __init__(self):
        self.model = None

    def train(self, train_dir, test_dir, batch_size=32, epochs=10):
        input_shape = (224, 224, 3)

        # Create data generators with augmentation
        train_datagen = get_image_data_generator()
        test_datagen = get_image_data_generator(rotation_range=0, width_shift_range=0, height_shift_range=0, shear_range=0, zoom_range=0, horizontal_flip=False, vertical_flip=False)

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

        # Create the model
        self.model = create_model(input_shape)

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

        # Plot learning curve
        plot_learning_curve(history)

    def evaluate(self, test_dir, batch_size=32):
        input_shape = (224, 224, 3)
        test_datagen = get_image_data_generator(rotation_range=0, width_shift_range=0, height_shift_range=0, shear_range=0, zoom_range=0, horizontal_flip=False, vertical_flip=False)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        evaluate_model(self.model, test_generator)

    def save_model(self, model_path):
        save_model(self.model, model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)
