# cnnClassification

This code builds a convolutional neural network (CNN) using the TensorFlow library. The CNN has three convolutional layers followed by max pooling layers, and a fully connected layer with 256 neurons. It uses the softmax activation function to output a probability distribution over 10 classes. The model is compiled with the Adam optimizer and the categorical cross-entropy loss function.

The code then creates data generators using the ImageDataGenerator class to perform data augmentation, rescaling the pixel values of the images to be between 0 and 1. It trains the model using the fit method on the training data generator, and evaluates the model on the test data generator using the evaluate method.

This code can be used to classify images into one of ten categories, such as different types of animals or objects. The performance of the model can be improved by tuning hyperparameters such as the number of layers, the number of neurons per layer, and the learning rate of the optimizer.
