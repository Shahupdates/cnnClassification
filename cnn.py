import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set hyperparameters
batch_size = 32
epochs = 10
input_shape = (224, 224, 3)

# Create data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    shear_range=0.2,  # Apply random shear augmentation
    zoom_range=0.2,  # Apply random zoom augmentation
    horizontal_flip=True  # Flip images horizontally
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale pixel values for testing

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size
)

# Evaluate the model
score = model.evaluate(test_generator, verbose=0)

# Print the test loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])
