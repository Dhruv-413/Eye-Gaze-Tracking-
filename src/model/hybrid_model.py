import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Dense, Flatten, Dropout, BatchNormalization, 
    Concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50, EfficientNetB3 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from typing import Tuple, Dict, Any, Optional

def create_hybrid_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    use_mixed_precision: bool = False,
    metadata_shape: int = 10,
    learning_rate: float = 0.0005,
    dropout_rate: float = 0.4,
    dense_units: Tuple[int, int] = (512, 256),
    include_pose_estimation: bool = True,
    fusion_method: str = 'feature'  # 'feature' or 'decision'
) -> Model:
    """
    Create a hybrid model that combines ResNet50 for facial feature extraction
    and EfficientNetB3 (GazeNet) for eye-specific features and pose estimation.
    
    Args:
        input_shape: Input shape for image inputs
        use_mixed_precision: Whether to use mixed precision
        metadata_shape: Dimension of metadata input vector
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate for regularization
        dense_units: Tuple of dense layer units
        include_pose_estimation: Whether to include face pose estimation branch
        fusion_method: Method for fusing features ('feature' or 'decision')
        
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
    
    # ResNet50 branch for facial feature extraction
    resnet_model = ResNet50(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=input_shape
    )
    
    # Freeze early layers of ResNet50 (first 75%)
    for layer in resnet_model.layers[:int(len(resnet_model.layers) * 0.75)]:
        layer.trainable = False
    
    # Extract facial features using ResNet50
    face_features_resnet = resnet_model(face_input)
    
    # EfficientNetB3 branch for eye-specific features (GazeNet approach)
    efficient_net = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=input_shape
    )
    
    # Freeze early layers of EfficientNet (first 70%)
    for layer in efficient_net.layers[:int(len(efficient_net.layers) * 0.7)]:
        layer.trainable = False
    
    # Extract eye features using EfficientNet
    left_eye_features = efficient_net(left_eye_input)
    right_eye_features = efficient_net(right_eye_input)
    
    # Process metadata features
    metadata_features = Dense(64, activation='relu')(metadata_input)
    metadata_features = BatchNormalization()(metadata_features)
    metadata_features = Dense(128, activation='relu')(metadata_features)
    metadata_features = BatchNormalization()(metadata_features)
    
    if fusion_method == 'feature':
        # Feature-level fusion
        # Combine facial features (ResNet) with eye features (EfficientNet)
        all_features = Concatenate()([face_features_resnet, left_eye_features, right_eye_features, metadata_features])
        
        # Shared representation
        shared = Dense(dense_units[0], activation='relu')(all_features)
        shared = BatchNormalization()(shared)
        shared = Dropout(dropout_rate)(shared)
        
        # Gaze estimation branch
        gaze_representation = Dense(dense_units[1], activation='relu')(shared)
        gaze_representation = BatchNormalization()(gaze_representation)
        gaze_representation = Dropout(dropout_rate * 0.6)(gaze_representation)
        gaze_output = Dense(2, activation='linear', name='gaze_output', dtype='float32')(gaze_representation)
        
        if include_pose_estimation:
            # Pose estimation branch
            pose_representation = Dense(dense_units[1], activation='relu')(shared)
            pose_representation = BatchNormalization()(pose_representation)
            pose_representation = Dropout(dropout_rate * 0.6)(pose_representation)
            pose_output = Dense(3, activation='linear', name='pose_output', dtype='float32')(pose_representation)
    
    else:  # Decision-level fusion
        # Process ResNet facial features
        face_representation = Dense(dense_units[0], activation='relu')(face_features_resnet)
        face_representation = BatchNormalization()(face_representation)
        face_representation = Dropout(dropout_rate)(face_representation)
        face_gaze = Dense(2, activation='linear')(face_representation)
        
        # Process EfficientNet eye features
        eye_features = Concatenate()([left_eye_features, right_eye_features, metadata_features])
        eye_representation = Dense(dense_units[0], activation='relu')(eye_features)
        eye_representation = BatchNormalization()(eye_representation)
        eye_representation = Dropout(dropout_rate)(eye_representation)
        eye_gaze = Dense(2, activation='linear')(eye_representation)
        
        # Fusion layer for final decision
        gaze_fusion = Concatenate()([face_gaze, eye_gaze])
        gaze_output = Dense(2, activation='linear', name='gaze_output', dtype='float32')(gaze_fusion)
        
        if include_pose_estimation:
            # Pose estimation from EfficientNet features
            pose_representation = Dense(dense_units[1], activation='relu')(eye_features)
            pose_representation = BatchNormalization()(pose_representation)
            pose_representation = Dropout(dropout_rate * 0.6)(pose_representation)
            pose_output = Dense(3, activation='linear', name='pose_output', dtype='float32')(pose_representation)
    
    # Create the model with the appropriate outputs
    if include_pose_estimation:
        model = Model(
            inputs=[face_input, left_eye_input, right_eye_input, metadata_input],
            outputs=[gaze_output, pose_output]
        )
        
        # Compile model with multiple outputs
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
        model = Model(
            inputs=[face_input, left_eye_input, right_eye_input, metadata_input],
            outputs=gaze_output
        )
        
        # Compile model with single output
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
    
    return model

def get_hybrid_model_callbacks(checkpoint_path: str) -> list:
    """
    Get default callbacks for hybrid model training
    
    Args:
        checkpoint_path: Path to save model checkpoints
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # More patience for complex model
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
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
