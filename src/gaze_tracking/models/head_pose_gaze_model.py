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
import optuna  # Added Optuna for hyperparameter tuning

def create_head_pose_gaze_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    use_mixed_precision: bool = False,
    metadata_shape: int = 6,  # Dimension of metadata features
    learning_rate: float = 0.001,
    dropout_rate: float = 0.5,
    dense_units: Tuple[int, int, int] = (512, 512, 256)  # Tunable parameters
) -> Model:
    """
    Create a model that uses face, eyes, head pose and metadata for gaze prediction
    
    Args:
        input_shape: Input shape for image inputs
        use_mixed_precision: Whether to use mixed precision
        metadata_shape: Dimension of metadata input vector
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate for regularization
        dense_units: Tuple of dense layer units (face_dense, combined_dense1, combined_dense2)
        
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
    face_features = Dense(dense_units[0], activation='relu')(face_features)
    face_features = Dropout(dropout_rate)(face_features)
    
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
        x = Dropout(dropout_rate)(x)
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
    x = Dense(dense_units[1], activation='relu')(combined_features)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units[2], activation='relu')(x)
    x = Dropout(dropout_rate * 0.6)(x)
    
    # Output layer (always float32)
    outputs = Dense(2, activation='linear', dtype='float32')(x)
    
    # Create the model
    model = Model(
        inputs=[face_input, left_eye_input, right_eye_input, head_pose_input, metadata_input],
        outputs=outputs
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def create_face_and_head_pose_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    use_mixed_precision: bool = False,
    metadata_shape: int = 6,  # Dimension of metadata features
    learning_rate: float = 0.001,
    dropout_rate: float = 0.5,
    dense_units: Tuple[int, int, int] = (512, 512, 256)  # Tunable parameters
) -> Model:
    """
    Create a simpler model that uses face, head pose and metadata for gaze prediction
    
    Args:
        input_shape: Input shape for image inputs
        use_mixed_precision: Whether to use mixed precision
        metadata_shape: Dimension of metadata input vector
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate for regularization
        dense_units: Tuple of dense layer units (face_dense, combined_dense1, combined_dense2)
        
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
    face_features = Dense(dense_units[0], activation='relu')(face_features)
    face_features = Dropout(dropout_rate)(face_features)
    
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
    x = Dense(dense_units[1], activation='relu')(combined_features)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units[2], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.6)(x)
    
    # Output layer
    outputs = Dense(2, activation='linear', dtype='float32')(x)
    
    # Create model
    model = Model(
        inputs=[face_input, head_pose_input, metadata_input],
        outputs=outputs
    )
    
    # Compile model with improved settings
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
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

def optuna_objective(trial, model_type: str, X_train, y_train, X_val, y_val, input_shape, metadata_shape, batch_size=32, epochs=50):
    """
    Objective function for Optuna hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        model_type: Type of model to tune ('full' or 'simple')
        X_train: Training inputs dictionary with keys matching model inputs
        y_train: Training targets
        X_val: Validation inputs dictionary
        y_val: Validation targets
        input_shape: Shape of image inputs
        metadata_shape: Shape of metadata vector
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        
    Returns:
        Validation mean squared error
    """
    # Define hyperparameters to search
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    
    # Different network architectures based on the model type
    if model_type == 'full':
        dense1 = trial.suggest_categorical('dense1', [256, 512, 768, 1024])
        dense2 = trial.suggest_categorical('dense2', [256, 384, 512, 768])
        dense3 = trial.suggest_categorical('dense3', [128, 192, 256, 384])
        
        model = create_head_pose_gaze_model(
            input_shape=input_shape,
            metadata_shape=metadata_shape,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dense_units=(dense1, dense2, dense3)
        )
    else:  # simple model
        dense1 = trial.suggest_categorical('dense1', [256, 384, 512, 768])
        dense2 = trial.suggest_categorical('dense2', [256, 384, 512, 640])
        dense3 = trial.suggest_categorical('dense3', [128, 192, 256, 320])
        
        model = create_face_and_head_pose_model(
            input_shape=input_shape,
            metadata_shape=metadata_shape,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dense_units=(dense1, dense2, dense3)
        )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Return best validation loss
    return min(history.history['val_mse'])

def optimize_hyperparameters(
    model_type: str, 
    X_train, 
    y_train, 
    X_val, 
    y_val,
    input_shape=(224, 224, 3),
    metadata_shape=6,
    n_trials=100, 
    study_name='gaze_model_optimization',
    batch_size=32,
    epochs=50
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization for gaze models
    
    Args:
        model_type: Type of model to optimize ('full' or 'simple')
        X_train: Training inputs dictionary
        y_train: Training target values
        X_val: Validation inputs dictionary
        y_val: Validation target values
        input_shape: Shape of image inputs
        metadata_shape: Shape of metadata vector 
        n_trials: Number of optimization trials
        study_name: Name for the Optuna study
        batch_size: Training batch size
        epochs: Maximum training epochs
        
    Returns:
        Dictionary with best parameters and best value
    """
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Create the study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Run optimization
    objective = lambda trial: optuna_objective(
        trial, model_type, X_train, y_train, X_val, y_val, 
        input_shape, metadata_shape, batch_size, epochs
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best MSE: {study.best_trial.value:.6f}")
    print("Best hyperparameters:", study.best_params)
    
    # Return best parameters
    return {
        'best_params': study.best_params,
        'best_value': study.best_trial.value,
        'study': study
    }

def create_model_with_optimal_params(
    model_type: str,
    optimal_params: Dict[str, Any],
    input_shape=(224, 224, 3),
    metadata_shape=6,
    use_mixed_precision=False
) -> Model:
    """
    Create a model with the optimal hyperparameters
    
    Args:
        model_type: Type of model to create ('full' or 'simple')
        optimal_params: Dictionary of optimal parameters from Optuna
        input_shape: Shape of image inputs
        metadata_shape: Shape of metadata vector
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Keras Model with optimal hyperparameters
    """
    # Extract hyperparameters
    learning_rate = optimal_params.get('learning_rate', 0.001)
    dropout_rate = optimal_params.get('dropout_rate', 0.5)
    dense1 = optimal_params.get('dense1', 512)
    dense2 = optimal_params.get('dense2', 512)
    dense3 = optimal_params.get('dense3', 256)
    
    if model_type == 'full':
        model = create_head_pose_gaze_model(
            input_shape=input_shape,
            metadata_shape=metadata_shape,
            use_mixed_precision=use_mixed_precision,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dense_units=(dense1, dense2, dense3)
        )
    else:  # simple model
        model = create_face_and_head_pose_model(
            input_shape=input_shape,
            metadata_shape=metadata_shape,
            use_mixed_precision=use_mixed_precision,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dense_units=(dense1, dense2, dense3)
        )
        
    return model

