# gui.py

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
from image_classifier import ImageClassifier

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Classifier')

        # Initialize classifier
        self.classifier = ImageClassifier()

        # Initialize hyperparameters with default values
        self.batch_size = tk.StringVar(value='32')
        self.epochs = tk.StringVar(value='10')
        self.train_dir = tk.StringVar(value='train')
        self.test_dir = tk.StringVar(value='test')

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text='Batch Size').grid(row=0, column=0)
        tk.Entry(self.root, textvariable=self.batch_size).grid(row=0, column=1)

        tk.Label(self.root, text='Epochs').grid(row=1, column=0)
        tk.Entry(self.root, textvariable=self.epochs).grid(row=1, column=1)

        tk.Label(self.root, text='Training Directory').grid(row=2, column=0)
        tk.Entry(self.root, textvariable=self.train_dir).grid(row=2, column=1)
        tk.Button(self.root, text='Browse', command=self.browse_train_dir).grid(row=2, column=2)

        tk.Label(self.root, text='Testing Directory').grid(row=3, column=0)
        tk.Entry(self.root, textvariable=self.test_dir).grid(row=3, column=1)
        tk.Button(self.root, text='Browse', command=self.browse_test_dir).grid(row=3, column=2)

        tk.Button(self.root, text='Train', command=self.start_training).grid(row=4, column=0)
        tk.Button(self.root, text='Load Model', command=self.load_model).grid(row=4, column=1)
        tk.Button(self.root, text='Save Model', command=self.save_model).grid(row=4, column=2)

        self.progress = tk.Label(self.root, text='Ready')
        self.progress.grid(row=5, column=0, columnspan=3)

    def browse_train_dir(self):
        self.train_dir.set(filedialog.askdirectory())

    def browse_test_dir(self):
        self.test_dir.set(filedialog.askdirectory())

    def start_training(self):
        try:
            batch_size = int(self.batch_size.get())
            epochs = int(self.epochs.get())
            if batch_size <= 0 or epochs <= 0:
                raise ValueError('Batch size and epochs must be positive.')
            self.progress['text'] = 'Training...'
            threading.Thread(target=self.train).start()
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def train(self):
        try:
            # Retrieve hyperparameters
            batch_size = int(self.batch_size.get())
            epochs = int(self.epochs.get())
            train_dir = self.train_dir.get()
            test_dir = self.test_dir.get()

            # Train the model
            self.classifier.train(train_dir, test_dir, batch_size, epochs)

            self.progress['text'] = 'Training complete'
        except Exception as e:
            self.progress['text'] = 'Training failed'
            messagebox.showerror("Error", str(e))

    def load_model(self):
        model_path = filedialog.askopenfilename()
        self.classifier.load_model(model_path)
        self.progress['text'] = 'Model loaded'

    def save_model(self):
        model_path = filedialog.asksaveasfilename(defaultextension=".h5")
        self.classifier.save_model(model_path)
        self.progress['text'] = 'Model saved'

if __name__ == '__main__':
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
