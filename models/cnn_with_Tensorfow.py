import tensorflow as tf


def build_model(num_classes: int = 6, input_shape: tuple = (150, 150, 3)):
    """
    TensorFlow/Keras CNN for Intel Image Classification.
    Input : (B, 150, 150, 3)  – RGB images, pixel values in [0, 1]
    Output: (B, 6)            – softmax probabilities for 6 scene classes

    Architecture
    ────────────────────────────────────────
    4 convolutional blocks (Conv2D → BN → ReLU → MaxPool):
        32 → 64 → 128 → 256 filters  + L2 regularization on each Conv
    GlobalAveragePooling2D
    Classifier: Dropout(0.5) → Dense(512, relu) → Dropout(0.5) → Dense(6, softmax)
    """
    reg = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same',
                               kernel_regularizer=reg, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding='same', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding='same', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Block 4
        tf.keras.layers.Conv2D(256, 3, padding='same', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        # Global pooling
        tf.keras.layers.GlobalAveragePooling2D(),

        # Classifier head — dropout augmenté pour réduire l'overfitting
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name='intel_cnn_tf')

    return model
