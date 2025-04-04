"""
Module containing the gaze tracking model definition and training utilities.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input # type: ignore
from tensorflow.keras.applications import ResNet50, MobileNetV2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from typing import Tuple, List, Dict, Any, Optional, Union

class GazeTrackingModel:
    """Class for building and training gaze tracking models."""
    
    def __init__(self, model_type: str = 'resnet50', input_shape: Tuple[int, int, int] = (64, 64, 3),
                use_mixed_precision: bool = False):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model to create ('resnet50', 'mobilenet', 'custom')
            input_shape: Input shape for the model (height, width, channels)
            use_mixed_precision: Whether to use mixed precision training
        """
        # Initializes the model with specified architecture type and input shape
        self.model_type = model_type
        self.input_shape = input_shape
        self.use_mixed_precision = use_mixed_precision
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """
        Build the model according to the specified type.
        
        Returns:
            Keras Model instance
        """
        # Set mixed precision policy if enabled
        if self.use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision enabled with {policy.name}")
            except Exception as e:
                print(f"Failed to enable mixed precision: {e}")
                self.use_mixed_precision = False
                
        # Build the selected model architecture
        if self.model_type == 'resnet50':
            return self._build_resnet50_model()
        elif self.model_type == 'mobilenet':
            return self._build_mobilenet_model()
        elif self.model_type == 'custom':
            return self._build_custom_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_resnet50_model(self) -> Model:
        """
        Build a model based on ResNet50.
        """
        # Creates ResNet50 base model with pretrained ImageNet weights
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freezes the pretrained layers initially
        for layer in base_model.layers:
            layer.trainable = False
            
        # Adds custom top layers for gaze regression
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer for x,y coordinates (linear activation)
        output = Dense(2, activation='linear', dtype='float32')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        # Compiles the model with MSE loss and MAE metric
        lr = 0.002 if self.use_mixed_precision else 0.001
        model.compile(
            optimizer=Adam(learning_rate=lr), 
            loss='mse', 
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_mobilenet_model(self) -> Model:
        """
        Build a model based on MobileNetV2 (lighter than ResNet50).
        
        Returns:
            MobileNetV2-based model
        """
        # Create the base model with pretrained weights
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freeze base model layers initially
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer (always float32 for stability)
        output = Dense(2, activation='linear', dtype='float32')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        # Use appropriate learning rate
        lr = 0.002 if self.use_mixed_precision else 0.001
        model.compile(
            optimizer=Adam(learning_rate=lr), 
            loss='mse', 
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_custom_model(self) -> Model:
        """
        Build a custom CNN model (lighter and doesn't need pretrained weights).
        
        Returns:
            Custom CNN model
        """
        inputs = Input(shape=self.input_shape)
        
        # First convolutional block
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Second convolutional block
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Third convolutional block
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Fully connected layers
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(2, activation='linear', dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def unfreeze_top_layers(self, num_layers: int = 30) -> None:
        """
        Unfreeze the top layers of the base model for fine-tuning.
        
        Args:
            num_layers: Number of top layers to unfreeze
        """
        if self.model_type not in ['resnet50', 'mobilenet']:
            print("Layer unfreezing only applicable to pretrained models")
            return
            
        # Get all layers
        all_layers = self.model.layers
        
        # Find the base model
        for layer in all_layers:
            if hasattr(layer, 'layers'):  # This is the base model
                base_model_layers = layer.layers
                # Unfreeze the last n layers
                for i, layer in enumerate(base_model_layers):
                    if i >= len(base_model_layers) - num_layers:
                        layer.trainable = True
                        print(f"Unfreezing layer: {layer.name}")
                break
                
        # Recompile with a lower learning rate for fine-tuning
        lr = 0.0005 if self.use_mixed_precision else 0.0001
        self.model.compile(
            optimizer=Adam(learning_rate=lr), 
            loss='mse', 
            metrics=['mae', 'mse']
        )
    
    def get_callbacks(self, checkpoint_dir: str) -> List:
        """
        Get training callbacks.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            List of callbacks
        """
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        return [
            EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=5, 
                min_lr=0.00001, 
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'), 
                save_best_only=True, 
                monitor='val_loss',
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(checkpoint_dir, 'logs'),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
