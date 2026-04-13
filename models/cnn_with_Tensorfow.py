import tensorflow as tf


def build_model(num_classes: int = 6, input_shape: tuple = (150, 150, 3)):
    """
    TensorFlow/Keras CNN for Intel Image Classification.
    Input : (B, 150, 150, 3)  – RGB images, pixel values in [0, 1]
    Output: (B, 6)            – softmax probabilities for 6 scene classes

    Architecture (mirrors the PyTorch CNN1)
    ────────────────────────────────────────
    4 convolutional blocks (Conv2D → BN → ReLU → MaxPool):
        32 → 64 → 128 → 256 filters
    GlobalAveragePooling2D
    Classifier: Dropout(0.5) → Dense(512, relu) → Dropout(0.3) → Dense(6, softmax)
    """
    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Block 4
        tf.keras.layers.Conv2D(256, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Global pooling instead of Flatten to keep param count reasonable
        tf.keras.layers.GlobalAveragePooling2D(),

        # Classifier head
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name='intel_cnn_tf')

    return model
