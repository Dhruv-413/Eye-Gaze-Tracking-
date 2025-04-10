import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv2D, MaxPooling2D, Dense, Flatten, 
    Dropout, BatchNormalization, Concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from typing import Tuple, Dict, Any, Optional
import optuna  # Added Optuna for hyperparameter tuning

def create_eye_gaze_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    use_mixed_precision: bool = False,
    metadata_shape: int = 10,  # Dimension of metadata features
    learning_rate: float = 0.001,
    dropout_rate: float = 0.5,
    dense_units: Tuple[int, int] = (512, 256)  # Simplified parameters
) -> Model:
    """
    Create a straightforward model that uses face, eyes, and metadata for gaze prediction
    
    Args:
        input_shape: Input shape for image inputs
        use_mixed_precision: Whether to use mixed precision
        metadata_shape: Dimension of metadata input vector
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate for regularization
        dense_units: Tuple of dense layer units (combined_dense1, combined_dense2)
        
    Returns:
        Keras Model with multiple inputs
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
    
    # Face branch using ResNet50
    face_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        pooling='avg'
    )
    
    # Freeze early layers of ResNet50
    for layer in face_model.layers[:-20]:
        layer.trainable = False
    
    face_features = face_model(face_input)
    
    # Eye feature extraction using the same ResNet50
    left_eye_features = face_model(left_eye_input)
    right_eye_features = face_model(right_eye_input)
    
    # Process metadata with simple dense layers
    metadata_features = Dense(64, activation='relu')(metadata_input)
    metadata_features = BatchNormalization()(metadata_features)
    metadata_features = Dense(128, activation='relu')(metadata_features)
    metadata_features = BatchNormalization()(metadata_features)
    
    # Combine all features
    combined_features = Concatenate()([face_features, left_eye_features, right_eye_features, metadata_features])
    
    # Fully connected layers
    x = Dense(dense_units[0], activation='relu')(combined_features)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units[1], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.6)(x)
    
    # Output layer (always float32)
    outputs = Dense(2, activation='linear', dtype='float32')(x)
    
    # Create the model
    model = Model(
        inputs=[face_input, left_eye_input, right_eye_input, metadata_input],
        outputs=outputs
    )
    
    # Compile model
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

def optimize_model_hyperparameters(
    X_train, 
    y_train, 
    X_val, 
    y_val,
    input_shape=(224, 224, 3),
    metadata_shape=10,
    n_trials=50, 
    study_name='gaze_model_optimization',
    batch_size=32,
    epochs=30
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization for eye gaze model
    
    Args:
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
    
    def objective(trial):
        # Define hyperparameters to search
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
        dense1 = trial.suggest_categorical('dense1', [256, 512, 768])
        dense2 = trial.suggest_categorical('dense2', [128, 256, 384])
        
        # Create and train model
        model = create_eye_gaze_model(
            input_shape=input_shape,
            metadata_shape=metadata_shape,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dense_units=(dense1, dense2)
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return min(history.history['val_mse'])
    
    # Create the study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Run optimization
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
    optimal_params: Dict[str, Any],
    input_shape=(224, 224, 3),
    metadata_shape=10,
    use_mixed_precision=False
) -> Model:
    """
    Create a model with the optimal hyperparameters
    
    Args:
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
    dense2 = optimal_params.get('dense2', 256)
    
    model = create_eye_gaze_model(
        input_shape=input_shape,
        metadata_shape=metadata_shape,
        use_mixed_precision=use_mixed_precision,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dense_units=(dense1, dense2)
    )
    
    return model