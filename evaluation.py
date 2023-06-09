from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def evaluate_model(model, test_generator):
    # Evaluate the model on the test data
    score = model.evaluate(test_generator, verbose=0)

    # Print the test loss and accuracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def plot_confusion_matrix(model, test_generator):
    # Generate predictions
    y_pred = model.predict(test_generator)
    y_true = test_generator.classes

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_learning_curve(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
