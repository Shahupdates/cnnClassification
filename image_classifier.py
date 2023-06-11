# image_classifier.py

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing import get_image_data_generator
from evaluation import evaluate_model, plot_confusion_matrix, plot_learning_curve
from model_utils import create_model, save_model, load_model

class ImageClassifier:
    def __init__(self):
        self.model = None

    def generate_data(self, directory, batch_size, input_shape, augmentation_params={}, class_mode='categorical'):
        datagen = get_image_data_generator(**augmentation_params)
        generator = datagen.flow_from_directory(
            directory,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode=class_mode
        )
        return generator

    def train(self, train_dir, test_dir, batch_size=32, epochs=10):
        input_shape = (224, 224, 3)

        # Create data generators
        train_generator = self.generate_data(train_dir, batch_size, input_shape, 
                                             augmentation_params={'horizontal_flip': True, 'vertical_flip': True})
        test_generator = self.generate_data(test_dir, batch_size, input_shape)

        # Create the model
        self.model = create_model(input_shape)

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Define early stopping and model checkpoint for optimization
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Train the model with fine-tuning
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=test_generator.n // batch_size,
            callbacks=[early_stopping, model_checkpoint]
        )

        # Load the best weights
        self.model.load_weights('best_model.h5')

        # Plot learning curve
        plot_learning_curve(history)

    def evaluate(self, test_dir, batch_size=32):
        input_shape = (224, 224, 3)
        test_generator = self.generate_data(test_dir, batch_size, input_shape)
        evaluate_model(self.model, test_generator)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, model_path):
        save_model(self.model, model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)
