import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # apply sin to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerModel:
    def __init__(self, sequence_length: int, n_features: int):
        """
        Initialize Transformer model for time series prediction.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features in the input data
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        
    def build_model(self, 
                   embed_dim: int = 32,
                   num_heads: int = 4,
                   ff_dim: int = 32,
                   num_transformer_blocks: int = 2,
                   mlp_units: list = [64, 32],
                   dropout: float = 0.2,
                   mlp_dropout: float = 0.3):
        """
        Build Transformer model architecture.
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Project input to embedding dimension
        x = layers.Dense(embed_dim)(inputs)
        
        # Positional encoding
        x = PositionalEncoding(self.sequence_length, embed_dim)(x)
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # MLP layers
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        self.model = Model(inputs, outputs)
        
        # Custom loss function
        def custom_loss(y_true, y_pred):
            # Binary cross-entropy for direction
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            # Mean squared error for magnitude
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            # Add L2 regularization loss
            reg_loss = tf.reduce_sum(self.model.losses)
            return bce + 0.2 * mse + reg_loss
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_loss,
            metrics=['accuracy']
        )
        
        logger.info("Transformer model built successfully")
        
    def train(self, 
              X: np.ndarray,
              y: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 50,
              batch_size: int = 32):
        """
        Train the model.
        """
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            min_delta=0.001
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("Model training completed")
        
        # Save model
        self.model.save('models/transformer_model')
        logger.info("Model saved to models/transformer_model")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        return self.model.predict(X)
