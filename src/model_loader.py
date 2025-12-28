import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, ReLU

class ModelManager:
    def __init__(self, models_dir="models"):
        self.digit_model = None
        self.solver_model = None
        self.models_dir = models_dir

    def build_svhn_model(self):
        """Reconstructs the architecture from your code lines 212-220"""
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10) 
        ])
        return model

    def build_sudoku_solver(self):
        """Reconstructs the architecture from your code lines 246-257"""
        model = Sequential()
        for _ in range(10):
            model.add(Conv2D(filters=512, kernel_size=3, padding='same', input_shape=(9, 9, 1) if model.layers == [] else None))
            model.add(BatchNormalization())
            model.add(ReLU())
        model.add(Conv2D(filters=9, kernel_size=1, padding='same'))
        # Add Softmax if missing (per your code line 263)
        model.add(tf.keras.layers.Softmax())
        return model

    def load_models(self):
        # 1. Load Digit Model
        digit_path = os.path.join(self.models_dir, "digit_svhn-196-0.14.hdf5")
        if os.path.exists(digit_path):
            self.digit_model = self.build_svhn_model()
            self.digit_model.load_weights(digit_path)
            print(f"Loaded {digit_path}")
        else:
            print(f"CRITICAL: {digit_path} not found.")

        # 2. Load Solver Model
        solver_path = os.path.join(self.models_dir, "sudoku_conv-20-0.10.hdf5")
        if os.path.exists(solver_path):
            self.solver_model = self.build_sudoku_solver()
            self.solver_model.load_weights(solver_path)
            print(f"Loaded {solver_path}")
        else:
            print(f"CRITICAL: {solver_path} not found.")

    def predict_digits(self, batch_images):
        if not self.digit_model: return None
        return self.digit_model.predict(batch_images)

    def predict_solver(self, input_board):
        if not self.solver_model: return None
        return self.solver_model.predict(input_board)