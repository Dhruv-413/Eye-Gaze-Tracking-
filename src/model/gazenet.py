import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Dense, Flatten, Dropout, BatchNormalization, 
    Concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.applications import EfficientNetB3 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from typing import Tuple, Dict, Any, Optional

def create_gazenet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    use_mixed_precision: bool = False,
    metadata_shape: int = 10,  # Dimension of metadata features
    learning_rate: float = 0.0005,  # GazeNet typically uses smaller learning rates
    dropout_rate: float = 0.4,
    dense_units: Tuple[int, int] = (512, 256),
    freeze_layers_pct: float = 0.7,
    include_pose_estimation: bool = True  # GazeNet incorporates face pose estimation
) -> Model:
    """
    Create a GazeNet-inspired model for calibration-free eye gaze prediction
    based on EfficientNetB3 architecture.
    
    Args:
        input_shape: Input shape for image inputs
        use_mixed_precision: Whether to use mixed precision
        metadata_shape: Dimension of metadata input vector
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate for regularization
        dense_units: Tuple of dense layer units
        freeze_layers_pct: Percentage of early layers to freeze
        include_pose_estimation: Whether to include face pose estimation branch
        
    Returns:
        Keras Model with multiple inputs and outputs
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
    metadata_input = Input(shape=(metadata_shape,), name='metadata_input')
    
    # Use EfficientNetB3 as base
    efficient_net = EfficientNetB3(
        weights='imagenet', 
        include_top=False, 
        pooling='avg',
        input_shape=input_shape
    )
    
    # Freeze early layers of EfficientNet
    num_layers = len(efficient_net.layers)
    num_layers_to_freeze = int(num_layers * freeze_layers_pct)
    for layer in efficient_net.layers[:num_layers_to_freeze]:
        layer.trainable = False
    
    # Get features from each input using the same backbone
    face_features = efficient_net(face_input)
    left_eye_features = efficient_net(left_eye_input)
    right_eye_features = efficient_net(right_eye_input)
    
    # Process metadata (could include camera parameters, screen info, etc.)
    metadata_features = Dense(64, activation='relu')(metadata_input)
    metadata_features = BatchNormalization()(metadata_features)
    metadata_features = Dense(128, activation='relu')(metadata_features)
    metadata_features = BatchNormalization()(metadata_features)
    
    # Combine all visual features
    visual_features = Concatenate()([face_features, left_eye_features, right_eye_features])
    visual_features = Dense(512, activation='relu')(visual_features)
    visual_features = BatchNormalization()(visual_features)
    visual_features = Dropout(dropout_rate)(visual_features)
    
    # Combine with metadata
    combined_features = Concatenate()([visual_features, metadata_features])
    
    # GazeNet typically has a shared representation before splitting into task-specific branches
    shared_representation = Dense(dense_units[0], activation='relu')(combined_features)
    shared_representation = BatchNormalization()(shared_representation)
    shared_representation = Dropout(dropout_rate)(shared_representation)
    
    # Branch for gaze estimation (primary task)
    gaze_representation = Dense(dense_units[1], activation='relu')(shared_representation)
    gaze_representation = BatchNormalization()(gaze_representation)
    gaze_representation = Dropout(dropout_rate * 0.6)(gaze_representation)
    
    # Output for gaze direction (x, y coordinates)
    gaze_output = Dense(2, activation='linear', name='gaze_output', dtype='float32')(gaze_representation)
    
    if include_pose_estimation:
        # Branch for face pose estimation (auxiliary task in GazeNet)
        # Typically estimates head rotation angles (pitch, yaw, roll)
        pose_representation = Dense(dense_units[1], activation='relu')(shared_representation)
        pose_representation = BatchNormalization()(pose_representation)
        pose_representation = Dropout(dropout_rate * 0.6)(pose_representation)
        
        # Output for face pose (3D angles: pitch, yaw, roll)
        pose_output = Dense(3, activation='linear', name='pose_output', dtype='float32')(pose_representation)
        
        # Multi-output model with both gaze and pose estimation
        model = Model(
            inputs=[face_input, left_eye_input, right_eye_input, metadata_input],
            outputs=[gaze_output, pose_output]
        )
        
        # Compile model with appropriate loss weights
        # GazeNet usually puts more weight on gaze estimation than pose
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'gaze_output': 'mse',
                'pose_output': 'mse'
            },
            loss_weights={
                'gaze_output': 1.0,  # Primary task
                'pose_output': 0.5   # Auxiliary task
            },
            metrics={
                'gaze_output': ['mae', 'mse'],
                'pose_output': ['mae', 'mse']
            }
        )
    else:
        # Single-output model with only gaze estimation
        model = Model(
            inputs=[face_input, left_eye_input, right_eye_input, metadata_input],
            outputs=gaze_output
        )
        
        # Compile model for single output
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
    
    return model

def get_gazenet_callbacks(checkpoint_path: str) -> list:
    """
    Get default callbacks for GazeNet model training
    
    Args:
        checkpoint_path: Path to save model checkpoints
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
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

def create_gazenet_with_optimal_params(
    optimal_params: Dict[str, Any],
    input_shape=(224, 224, 3),
    metadata_shape=10,
    use_mixed_precision=False
) -> Model:
    """
    Create a GazeNet model with optimal hyperparameters
    
    Args:
        optimal_params: Dictionary of optimal parameters from Optuna
        input_shape: Shape of image inputs
        metadata_shape: Shape of metadata vector
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Keras Model with optimal hyperparameters
    """
    # Extract hyperparameters
    learning_rate = optimal_params.get('learning_rate', 0.0005)
    dropout_rate = optimal_params.get('dropout_rate', 0.4)
    dense1 = optimal_params.get('dense1', 512)
    dense2 = optimal_params.get('dense2', 256)
    freeze_layers_pct = optimal_params.get('freeze_layers_pct', 0.7)
    include_pose = optimal_params.get('include_pose_estimation', True)
    
    model = create_gazenet_model(
        input_shape=input_shape,
        metadata_shape=metadata_shape,
        use_mixed_precision=use_mixed_precision,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dense_units=(dense1, dense2),
        freeze_layers_pct=freeze_layers_pct,
        include_pose_estimation=include_pose
    )
    
    return model
