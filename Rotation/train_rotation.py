#!/usr/bin/env python3
# train_dnn_geodesic_hybrid.py

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from utils import (
    quaternion_to_euler,
    load_vertex_data,
    compute_hybrid_features_all_vertices,
    load_quaternion_from_json,
)

def configure_gpu():
    print(f"TensorFlow version: {tf.__version__}")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            logical = tf.config.list_logical_devices('GPU')
            print(f"GPUs detected: {[d.name for d in logical]}")
        else:
            print("No GPU detected by TensorFlow. Using CPU.")
    except Exception as e:
        print(f"GPU setup error: {e}")

#############################
# CONFIGURATION PARAMETERS #
#############################
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saves", "4SP.h5")
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
TRAIN_VAL_SPLIT_RATIO = 0.8
LEARNING_RATE = 1e-4  # Initial learning rate for CosineDecayRestarts
WEIGHT_DECAY = 1e-4 # Weight decay for AdamW optimizer
BATCH_SIZE = 1#512#128
EPOCHS = 5#100
DROP_OUT_RATE = 0.1
USE_ADAMW = False  # Toggle between AdamW and Adam+L2-on-kernels
L2_REGULARIZATION = 1e-4  # Used when USE_ADAMW is False

#############################
# Helper Functions
#############################
def _load_and_process_sample(json_path_tensor, edges_tensor, faces_tensor, image_width_tensor, image_height_tensor):
    """
    Loads and processes a single sample. To be wrapped by tf.py_function.
    """
    json_path = json_path_tensor.numpy().decode('utf-8')
    edges = edges_tensor.numpy().tolist()
    faces = faces_tensor.numpy().tolist()
    image_width = image_width_tensor.numpy()
    image_height = image_height_tensor.numpy()

    try:
        raw_vertex = load_vertex_data(json_path)
        hybrid_features = compute_hybrid_features_all_vertices(raw_vertex, edges, faces, image_width, image_height)
        quaternion = load_quaternion_from_json(json_path)
        return tf.convert_to_tensor(hybrid_features, dtype=tf.float32), tf.convert_to_tensor(quaternion, dtype=tf.float32)
    except Exception as e:
        print(f"Error processing {json_path}, skipping: {e}")
        # Return zero tensors to be filtered out later
        return tf.zeros([1], dtype=tf.float32), tf.zeros([4], dtype=tf.float32)

def create_tf_dataset(json_paths, edges, faces, batch_size, is_training, image_width, image_height):
    """Creates a tf.data.Dataset for on-the-fly data loading."""
    
    edges_tensor = tf.constant(edges, dtype=tf.int32)
    faces_tensor = tf.constant(faces, dtype=tf.int32)
    iw_tensor = tf.constant(image_width, dtype=tf.int32)
    ih_tensor = tf.constant(image_height, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices(json_paths)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(json_paths))
        # Repeat the dataset for multiple epochs.
        # This is crucial to prevent the 'OUT_OF_RANGE' error during training with model.fit
        dataset = dataset.repeat()

    dataset = dataset.map(lambda path: tf.py_function(
        _load_and_process_sample,
        [path, edges_tensor, faces_tensor, iw_tensor, ih_tensor],
        [tf.float32, tf.float32]
    ), num_parallel_calls=tf.data.AUTOTUNE)

    # Let feature dimensionality be inferred from preprocessing; avoid hardcoding sizes

    # Filter out samples that had processing errors
    dataset = dataset.filter(lambda features, quaternion: tf.reduce_sum(tf.abs(features)) > 0)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print(f"Created {'training' if is_training else 'validation'} tf.data.Dataset.")
    return dataset

def load_dataset(vertices_folder, edges, faces, image_width=1920, image_height=1080):
    """
    Loads the dataset from the folder of JSON files.
    For each file, computes a hybrid feature vector (length depends on mesh layout)
    and extracts the target quaternion (4-dim).
    Returns:
       X: (N, F) numpy array where F is the feature dimension.
       Y: (N, 4) numpy array.
    """
    json_files = sorted([os.path.join(vertices_folder, f) for f in os.listdir(vertices_folder)
                         if f.endswith('.json')])
    X_list = []
    Y_list = []
    num_files = len(json_files)
    print(f"Found {num_files} JSON files to process.")

    for i, jf in enumerate(json_files):
        try:
            raw_vertex = load_vertex_data(jf)
            hybrid = compute_hybrid_features_all_vertices(raw_vertex, edges, faces, image_width, image_height)
            q = load_quaternion_from_json(jf)
            X_list.append(hybrid)
            Y_list.append(q)
            if (i + 1) % 500 == 0 or (i + 1) == num_files:
                print(f"  Processed {i + 1}/{num_files} files...")
        except Exception as e:
            print(f"Skipping {jf} due to error: {e}")
            continue
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y

#############################
# Loss Function
#############################

def geodesic_loss(q_true, q_pred):
    """
    Computes a numerically more stable geodesic loss between batches of unit quaternions.
    
    Loss = mean(2 * arccos(|dot(q_true, q_pred)|))
    
    Here we clip the dot product to a maximum of 0.999999 to avoid numerical singularity.
    """
    # Ensure both true and predicted quaternions are normalized
    q_true = tf.math.l2_normalize(q_true, axis=1)
    q_pred = tf.math.l2_normalize(q_pred, axis=1)
    dot = tf.abs(tf.reduce_sum(q_true * q_pred, axis=1))
    dot = tf.clip_by_value(dot, -1.0, 0.999999)
    theta = 2.0 * tf.acos(dot)
    return tf.reduce_mean(theta)

def geodesic_deg(y_true, y_pred):
    """
    Mean geodesic angle in degrees between unit quaternions.
    """
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    dot = tf.abs(tf.reduce_sum(y_true * y_pred, axis=1))
    dot = tf.clip_by_value(dot, -1.0, 0.999999)
    theta = 2.0 * tf.acos(dot)
    return tf.reduce_mean(theta * (180.0/np.pi))

class DegCheck(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        vl = logs.get("val_loss")
        vd = logs.get("val_geodesic_deg")
        if vl is not None and vd is not None:
            print(f"[deg-check] val_loss(deg)={vl*180.0/np.pi:.3f} vs val_geodesic_deg={vd:.3f}")

#############################
# Model Definition with Residual Connections
#############################

class UnitNormalize(tf.keras.layers.Layer):
    """Outputs z / ||z|| with numerical stability; preserves input shape."""
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, inputs):
        norm = tf.norm(inputs, axis=1, keepdims=True)
        return inputs / (norm + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def build_dnn_model_v2(input_dim, normalization_layer=None):
    """
    Builds a deeper model with no explicit kernel regularization (using AdamW instead).
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    if normalization_layer is not None:
        x = normalization_layer(x)

    # Wider initial processing block
    x = tf.keras.layers.Dense(512, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)

    # --- Residual Block 1 (maintains 512 dims) ---
    shortcut1 = x
    x = tf.keras.layers.Dense(512, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = tf.keras.layers.Dense(512, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut1])
    x = tf.keras.layers.Activation('swish')(x)

    # --- Residual Block 2 (downsamples to 256 dims) ---
    shortcut2 = tf.keras.layers.Dense(256, use_bias=False)(x)
    shortcut2 = tf.keras.layers.BatchNormalization()(shortcut2)
    x = tf.keras.layers.Dense(256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = tf.keras.layers.Dense(256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut2])
    x = tf.keras.layers.Activation('swish')(x)
    
    # --- Residual Block 3 (downsamples to 128 dims) ---
    shortcut3 = tf.keras.layers.Dense(128, use_bias=False)(x)
    shortcut3 = tf.keras.layers.BatchNormalization()(shortcut3)
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut3])
    x = tf.keras.layers.Activation('swish')(x)

    # --- NEW Residual Block 4 (downsamples to 64 dims) ---
    shortcut4 = tf.keras.layers.Dense(64, use_bias=False)(x)
    shortcut4 = tf.keras.layers.BatchNormalization()(shortcut4)
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut4])
    x = tf.keras.layers.Activation('swish')(x)

    # --- Final Bottleneck Layers ---
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    x = tf.keras.layers.Dense(32, activation='swish')(x)
    #x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    
    # --- Output Layer ---
    outputs = tf.keras.layers.Dense(4, activation='linear')(x)
    outputs = UnitNormalize(name="unit_normalize_output")(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def DenseL2(units):
    return tf.keras.layers.Dense(
        units,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)
    )

def build_dnn_model_v2_l2(input_dim, normalization_layer=None):
    """
    Same as v2 but applies L2 only to Dense kernels (no AdamW weight decay).
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    if normalization_layer is not None:
        x = normalization_layer(x)

    # Wider initial processing block
    x = DenseL2(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)

    # --- Residual Block 1 (maintains 512 dims) ---
    shortcut1 = x
    x = DenseL2(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = DenseL2(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut1])
    x = tf.keras.layers.Activation('swish')(x)

    # --- Residual Block 2 (downsamples to 256 dims) ---
    shortcut2 = DenseL2(256)(x)
    shortcut2 = tf.keras.layers.BatchNormalization()(shortcut2)
    x = DenseL2(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = DenseL2(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut2])
    x = tf.keras.layers.Activation('swish')(x)

    # --- Residual Block 3 (downsamples to 128 dims) ---
    shortcut3 = DenseL2(128)(x)
    shortcut3 = tf.keras.layers.BatchNormalization()(shortcut3)
    x = DenseL2(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = DenseL2(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut3])
    x = tf.keras.layers.Activation('swish')(x)

    # --- NEW Residual Block 4 (downsamples to 64 dims) ---
    shortcut4 = DenseL2(64)(x)
    shortcut4 = tf.keras.layers.BatchNormalization()(shortcut4)
    x = DenseL2(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = DenseL2(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut4])
    x = tf.keras.layers.Activation('swish')(x)

    # --- Final Bottleneck Layers ---
    x = DenseL2(64)(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = DenseL2(32)(x)
    x = tf.keras.layers.Activation('swish')(x)

    # --- Output Layer ---
    outputs = tf.keras.layers.Dense(4, activation='linear')(x)
    outputs = UnitNormalize(name="unit_normalize_output")(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_dnn_model_v3_small(input_dim, normalization_layer=None):
    """
    Builds a smaller model with 3 residual blocks.
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    if normalization_layer is not None:
        x = normalization_layer(x)

    # Wider initial processing block
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)

    # --- Residual Block 1 (maintains 512 dims) ---
    shortcut1 = x
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut1])
    x = tf.keras.layers.Activation('swish')(x)

    # --- Residual Block 2 (downsamples to 256 dims) ---
    shortcut2 = tf.keras.layers.Dense(64, use_bias=False)(x)
    shortcut2 = tf.keras.layers.BatchNormalization()(shortcut2)
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut2])
    x = tf.keras.layers.Activation('swish')(x)
    

    # --- Final Bottleneck Layers ---
    x = tf.keras.layers.Dense(32, activation='swish')(x)
    x = tf.keras.layers.Dropout(DROP_OUT_RATE)(x)
    
    # --- Output Layer ---
    outputs = tf.keras.layers.Dense(4, activation='linear')(x)
    outputs = UnitNormalize(name="unit_normalize_output")(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

    
    

#############################
# Prediction Analysis Function
#############################

def analyze_predictions(model, X_val, Y_val, num_samples=10):
    """
    Takes a subset of the validation set, predicts quaternions,
    converts both predictions and ground truth to Euler angles,
    and prints the angular error (in degrees) for each sample.
    """
    num_to_show = min(num_samples, X_val.shape[0])
    preds = model.predict(X_val[:num_to_show])
    for i in range(num_to_show):
        q_true = Y_val[i]
        q_pred = preds[i]
        euler_true = quaternion_to_euler(q_true)
        euler_pred = quaternion_to_euler(q_pred)
        # Compute angular error in radians via dot product
        dot = np.clip(np.abs(np.dot(q_true, q_pred)), -1.0, 1.0)
        angular_error_rad = 2 * np.arccos(dot)
        angular_error_deg = np.degrees(angular_error_rad)
        print(f"Sample {i}:")
        print(f"  Ground truth Euler angles (deg): {np.degrees(euler_true)}")
        print(f"  Predicted Euler angles (deg):    {np.degrees(euler_pred)}")
        print(f"  Angular error: {angular_error_deg:.2f}°")
        print("-"*50)

#############################
# Main Training Script
#############################

def main():
    configure_gpu()
    # --- Path Setup ---
    # Assumes input data is in the same directory, and saves models to a 'models' sub-folder.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input paths for .npy files
    input_x_path = os.path.join(script_dir,'X_data.npy')
    input_y_path = os.path.join(script_dir,'Y_data.npy')

    # Output paths for models and checkpoints
    model_dir = os.path.join(script_dir, 'models')
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    model_path = os.path.join(model_dir, 'final_model.keras')
    best_model_path = os.path.join(model_dir, 'best_model.keras')

    # --- Create output directories if they don't exist ---
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --- Load pre-processed data ---
    print("Loading pre-processed data...")
    try:
        X = np.load(input_x_path)
        Y = np.load(input_y_path)
    except FileNotFoundError:
        # Try fallback location where combine script writes outputs
        fallback_dir = os.path.join(script_dir, 'saves_vastai_noise_not_visible_1M')
        fx = os.path.join(fallback_dir, 'X_data.npy')
        fy = os.path.join(fallback_dir, 'Y_data.npy')
        try:
            X = np.load(fx)
            Y = np.load(fy)
            print(f"Loaded data from fallback directory: {fallback_dir}")
        except FileNotFoundError:
            print(f"Error: Could not find data files at {input_x_path} or {input_y_path}, nor at {fx} or {fy}.")
            print("Please run preprocessing to generate the .npy files.")
            return
        
    print(f"Data loaded. X shape: {X.shape}, Y shape: {Y.shape}")

    # Normalize and canonicalize label quaternions once
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    Y = np.where(Y[:, :1] < 0, -Y, Y)

    # --- Shuffle and Split Data ---
    p = np.random.permutation(len(X))
    X, Y = X[p], Y[p]
    
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    if len(X_train) < BATCH_SIZE or len(X_val) < BATCH_SIZE:
        print("Not enough data to form a single batch. Please check dataset and BATCH_SIZE.")
        return

    # Calculate steps per epoch to correctly configure checkpoint saving frequency
    steps_per_epoch = len(X_train) // BATCH_SIZE

    # --- Build and Compile Model ---
    input_dim = X_train.shape[1]
    # Remove global Normalization: categorical flags (0/1) and wrapped angles should not be standardized
    if USE_ADAMW:
        model = build_dnn_model_v2(input_dim, normalization_layer=None)
    else:
        model = build_dnn_model_v2_l2(input_dim, normalization_layer=None)
    
    # --- Learning Rate Schedule ---
    # Replace restarts with plain cosine decay to avoid LR spikes
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=steps_per_epoch * 100,  # span roughly the whole run
        alpha=0.05
    )

    # Optimizer: AdamW if enabled, else Adam (L2 only on Dense kernels)
    if USE_ADAMW:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0
        )
    else:
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )

    model.compile(
        optimizer=opt,
        loss=geodesic_loss,
        metrics=[geodesic_deg]
    )
    model.summary()

    # --- Callbacks ---
    callbacks_list = []
    callbacks_list.append(DegCheck())
    callbacks_list.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
    ))
    callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ))
    # Checkpoint to save every 5 epochs
    if steps_per_epoch > 0:
        callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
            save_freq=5 * steps_per_epoch,  # Save every 5 epochs (calculated in batches)
            save_weights_only=False,
            verbose=1
        ))

    # --- Train the Model ---
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_val, Y_val),
                        callbacks=callbacks_list,
                        shuffle=True)
    
    # --- Plot training history ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (rad)')
    plt.plot(history.history['val_loss'], label='Val Loss (rad)')
    plt.title('Geodesic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Radians')
    plt.legend()

    # Plot geodesic degrees metric if present
    metric_key = 'geodesic_deg'
    val_metric_key = 'val_geodesic_deg'
    if metric_key in history.history and val_metric_key in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[metric_key], label='Train Geodesic (deg)')
        plt.plot(history.history[val_metric_key], label='Val Geodesic (deg)')
        plt.title('Geodesic Angle')
        plt.xlabel('Epoch')
        plt.ylabel('Degrees')
        plt.legend()
    else:
        plt.subplot(1, 2, 2)
        plt.plot(history.history.get('mae', []), label='Train MAE')
        plt.plot(history.history.get('val_mae', []), label='Val MAE')
        plt.title('MAE (fallback)')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()
    
    # --- Analyze and Save ---
    print("\nAnalyzing predictions on a subset of the validation set:")
    analyze_predictions(model, X_val, Y_val, num_samples=10)
    
    model.save(model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model was saved to {best_model_path}")

if __name__ == "__main__":
    main()