"""
Model that combines face, eye images, head pose data, and metadata for gaze prediction
"""

import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv2D, MaxPooling2D, Dense, Flatten, 
    Dropout, BatchNormalization, Concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50, MobileNetV2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from typing import Tuple, Dict, Any, Optional

def create_head_pose_gaze_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    use_mixed_precision: bool = False,
    metadata_shape: int = 6  # Dimension of metadata features
) -> Model:
    """
    Create a model that uses face, eyes, head pose and metadata for gaze prediction
    
    Args:
        input_shape: Input shape for image inputs
        use_mixed_precision: Whether to use mixed precision
        metadata_shape: Dimension of metadata input vector
        
    Returns:
        Keras Model with multiple inputs including head pose and metadata
    """
    # Set precision policy
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled with {policy.name}")
    
    # Create input layers
    face_input = Input(shape=input_shape, name='face_input')
    left_eye_input = Input(shape=input_shape, name='left_eye_input')
    right_eye_input = Input(shape=input_shape, name='right_eye_input')
    head_pose_input = Input(shape=(3,), name='head_pose_input')  # pitch, yaw, roll
    metadata_input = Input(shape=(metadata_shape,), name='metadata_input')  # New metadata input
    
    # Face branch using ResNet50
    face_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_tensor=face_input,
        input_shape=input_shape
    )
    
    # Freeze early layers of ResNet50
    for layer in face_model.layers[:-20]:
        layer.trainable = False
    
    face_features = face_model.output
    face_features = GlobalAveragePooling2D()(face_features)
    face_features = Dense(512, activation='relu')(face_features)
    face_features = Dropout(0.5)(face_features)
    
    # Eye feature extraction branches (shared weights)
    def create_eye_branch(input_tensor):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x
    
    left_eye_features = create_eye_branch(left_eye_input)
    right_eye_features = create_eye_branch(right_eye_input)
    
    # Process head pose data
    head_pose_features = Dense(64, activation='relu')(head_pose_input)
    head_pose_features = Dense(128, activation='relu')(head_pose_features)
    head_pose_features = Dense(256, activation='relu')(head_pose_features)
    
    # Process metadata
    metadata_features = Dense(64, activation='relu')(metadata_input)
    metadata_features = BatchNormalization()(metadata_features)
    metadata_features = Dense(128, activation='relu')(metadata_features)
    metadata_features = BatchNormalization()(metadata_features)
    
    # Combine all features
    combined_features = Concatenate()(
        [face_features, left_eye_features, right_eye_features, head_pose_features, metadata_features]
    )
    
    # Fully connected layers
    x = Dense(512, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer (always float32)
    outputs = Dense(2, activation='linear', dtype='float32')(x)
    
    # Create the model
    model = Model(
        inputs=[face_input, left_eye_input, right_eye_input, head_pose_input, metadata_input],
        outputs=outputs
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def create_face_and_head_pose_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    use_mixed_precision: bool = False,
    metadata_shape: int = 6  # Dimension of metadata features
) -> Model:
    """
    Create a simpler model that uses face, head pose and metadata for gaze prediction
    
    Args:
        input_shape: Input shape for image inputs
        use_mixed_precision: Whether to use mixed precision
        metadata_shape: Dimension of metadata input vector
        
    Returns:
        Keras Model with face, head pose and metadata inputs
    """
    # Set precision policy
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled with {policy.name}")
    
    # Create input layers
    face_input = Input(shape=input_shape, name='face_input')
    head_pose_input = Input(shape=(3,), name='head_pose_input')  # pitch, yaw, roll
    metadata_input = Input(shape=(metadata_shape,), name='metadata_input')  # New metadata input
    
    # Face branch using ResNet50
    face_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_tensor=face_input,
        input_shape=input_shape
    )
    
    # Freeze early layers
    for layer in face_model.layers[:-20]:
        layer.trainable = False
    
    face_features = face_model.output
    face_features = GlobalAveragePooling2D()(face_features)
    face_features = Dense(512, activation='relu')(face_features)
    face_features = Dropout(0.5)(face_features)
    
    # Process head pose data
    head_pose_features = Dense(64, activation='relu')(head_pose_input)
    head_pose_features = Dense(128, activation='relu')(head_pose_features)
    head_pose_features = BatchNormalization()(head_pose_features)
    
    # Process metadata
    metadata_features = Dense(32, activation='relu')(metadata_input)
    metadata_features = BatchNormalization()(metadata_features)
    metadata_features = Dense(64, activation='relu')(metadata_features)
    metadata_features = BatchNormalization()(metadata_features)
    
    # Combine features
    combined_features = Concatenate()([face_features, head_pose_features, metadata_features])
    
    # Enhanced fully connected layers with BatchNormalization
    x = Dense(512, activation='relu')(combined_features)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(2, activation='linear', dtype='float32')(x)
    
    # Create model
    model = Model(
        inputs=[face_input, head_pose_input, metadata_input],
        outputs=outputs
    )
    
    # Compile model with improved settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def get_training_callbacks(checkpoint_path: str) -> list:
    """
    Get default callbacks for model training
    
    Args:
        checkpoint_path: Path to save model checkpoints
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

