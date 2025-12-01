#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess_invis import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)


#############################
# CONFIGURATIONS #
#############################
DEBUG = True

##### TRAINING HYPERPARAMETERS #####
TRAIN_VAL_SPLIT_RATIO = 0.8
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 40

##### REDUCELRONPATEAU CALLBACK PARAMETERS #####
RLROP_MONITOR = 'val_loss'
RLROP_FACTOR = 0.75
RLROP_PATIENCE = 10
RLROP_MIN_LR = 1e-6

HIDDEN_DIM = 256
DROPOUT_RATE = 0.3
L2_REG = 1e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_INPUT_PATH = os.path.join(BASE_DIR, 'X_data.npy')
Y_INPUT_PATH = os.path.join(BASE_DIR, 'Y_data.npy')

##### MODEL SAVE PATHS #####
BEST_MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'BestModel', 'best_model_invisible.h5')
FINAL_MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'final_model_invisible.h5')
PERIODIC_MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'PeriodicTrainingModels', 'periodic_model_invisible.h5')
LOG_DIR = os.path.join(BASE_DIR, 'training_logs_invisible')


##### LOSSES AND METRICS #####
def masked_mse_loss(y_true, y_pred):
    mask = y_true[:, :, 0]
    true_coords = y_true[:, :, 1:3]
    sq_err = tf.square(y_pred - true_coords)
    mask_expanded = tf.expand_dims(mask, axis=-1)
    masked_sq_err = sq_err * mask_expanded
    total_loss = tf.reduce_sum(masked_sq_err)
    count = tf.reduce_sum(mask_expanded) + 1e-8
    return total_loss / count


def masked_mae_metric(y_true, y_pred):
    mask = y_true[:, :, 0]
    true_coords = y_true[:, :, 1:3]
    abs_err = tf.abs(y_pred - true_coords)
    mask_expanded = tf.expand_dims(mask, axis=-1)
    masked_abs_err = abs_err * mask_expanded
    total_err = tf.reduce_sum(masked_abs_err)
    count = tf.reduce_sum(mask_expanded) + 1e-8
    return total_err / count


def masked_euclidean_loss(y_true, y_pred, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    mask = y_true[:, :, 0]
    true_coords_norm = y_true[:, :, 1:3]
    pred_coords_norm = y_pred
    img_dims = tf.constant([float(image_width), float(image_height)], dtype=tf.float32)
    true_coords = true_coords_norm * img_dims
    pred_coords = pred_coords_norm * img_dims
    error_per_vertex = tf.norm(pred_coords - true_coords, axis=-1)
    masked_error_per_vertex = error_per_vertex * mask
    total_masked_error = tf.reduce_sum(masked_error_per_vertex)
    num_masked_vertices = tf.reduce_sum(mask) + 1e-8
    return total_masked_error / num_masked_vertices


def weighted_masked_euclidean_loss(y_true, y_pred,
                                   image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                   non_visible_weight=3.0, flipped_weight=1.0):
    mask = y_true[:, :, 0]
    true_coords_norm = y_true[:, :, 1:3]
    pred_coords_norm = y_pred
    original_visibility_flag = y_true[:, :, 3]
    weights = tf.where(original_visibility_flag == 0.0, non_visible_weight, flipped_weight)
    img_dims = tf.constant([float(image_width), float(image_height)], dtype=tf.float32)
    true_coords = true_coords_norm * img_dims
    pred_coords = pred_coords_norm * img_dims
    error_per_vertex = tf.norm(pred_coords - true_coords, axis=-1)
    masked_error = error_per_vertex * mask
    weighted_masked_error = masked_error * weights
    total_weighted_error = tf.reduce_sum(weighted_masked_error)
    total_weight_sum = tf.reduce_sum(mask * weights) + 1e-8
    return total_weighted_error / total_weight_sum


# =============================
# Model Definition (build_model_v5 analogue)
# =============================
def build_model_v5(input_dim: int, num_vertices: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    l2_reg = tf.keras.regularizers.l2(L2_REG)

    # Input normalization to stabilize training on mixed-scale engineered features
    norm = tf.keras.layers.Normalization(axis=-1)
    # Note: we'll adapt this layer on X_train before fitting
    x = norm(inputs)
    x = tf.keras.layers.Dense(HIDDEN_DIM, use_bias=False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    shortcut1 = x
    x = tf.keras.layers.Dense(HIDDEN_DIM, use_bias=False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(HIDDEN_DIM, use_bias=False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut1])
    x = tf.keras.layers.Activation('swish')(x)

    shortcut2 = tf.keras.layers.Dense(HIDDEN_DIM//2, use_bias=False, kernel_regularizer=l2_reg)(x)
    shortcut2 = tf.keras.layers.BatchNormalization()(shortcut2)
    x = tf.keras.layers.Dense(HIDDEN_DIM//2, use_bias=False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(HIDDEN_DIM//2, use_bias=False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut2])
    x = tf.keras.layers.Activation('swish')(x)

    shortcut3 = tf.keras.layers.Dense(HIDDEN_DIM//4, use_bias=False, kernel_regularizer=l2_reg)(x)
    shortcut3 = tf.keras.layers.BatchNormalization()(shortcut3)
    x = tf.keras.layers.Dense(HIDDEN_DIM//4, use_bias=False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(HIDDEN_DIM//4, use_bias=False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut3])
    x = tf.keras.layers.Activation('swish')(x)

    x = tf.keras.layers.Dense(HIDDEN_DIM//4, activation='swish')(x)
    x = tf.keras.layers.Dense(2 * num_vertices, activation='sigmoid')(x)
    outputs = tf.keras.layers.Reshape((num_vertices, 2))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Attach the normalization layer reference for adaptation later
    model._input_norm_layer = norm
    return model


class PeriodicModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, period):
        super(PeriodicModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.model.save(self.filepath)
            print(f'\nEpoch {epoch+1}: saving model to {self.filepath}')


# =============================
# Utilities
# =============================
def ensure_dirs_exist(paths):
    for path in paths:
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)


# =============================
# Main
# =============================
def main():
    # Ensure output directories exist
    ensure_dirs_exist([BEST_MODEL_SAVE_PATH, PERIODIC_MODEL_SAVE_PATH])
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

    # Load preprocessed arrays (offline augmentation already applied during preprocessing)
    if not (os.path.exists(X_INPUT_PATH) and os.path.exists(Y_INPUT_PATH)):
        print(f"Missing preprocessed arrays. Expected at {X_INPUT_PATH} and {Y_INPUT_PATH}.")
        return
    X = np.load(X_INPUT_PATH)
    Y = np.load(Y_INPUT_PATH)
    if DEBUG:
        print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")

    num_samples = X.shape[0]
    input_dim = X.shape[1]
    num_vertices = Y.shape[1]

    # Train/val split
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * num_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    # Build and compile model
    model = build_model_v5(input_dim, num_vertices)
    # Adapt normalization on training data
    if hasattr(model, '_input_norm_layer'):
        model._input_norm_layer.adapt(X_train)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss=weighted_masked_euclidean_loss,
                  metrics=[masked_mae_metric])
    model.summary()

    # Callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True
    )
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=RLROP_MONITOR, factor=RLROP_FACTOR, patience=RLROP_PATIENCE, min_lr=RLROP_MIN_LR
    )
    periodic_checkpoint_callback = PeriodicModelCheckpoint(
        filepath=PERIODIC_MODEL_SAVE_PATH, period=10
    )
    best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_SAVE_PATH,
        save_weights_only=False,
        monitor='val_loss', mode='min', save_best_only=True, verbose=1
    )

    # Train
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[reduce_lr_callback, periodic_checkpoint_callback, best_model_checkpoint_callback, early_stopping_callback],
        verbose=1,
    )

    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['masked_mae_metric'], label='Train Masked MAE')
    plt.plot(history.history['val_masked_mae_metric'], label='Val Masked MAE')
    plt.title('Masked MAE (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'training_curves.png'))
    plt.show()

    # Save final model
    print(f"Saving final model to {FINAL_MODEL_SAVE_PATH}...")
    model.save(FINAL_MODEL_SAVE_PATH)
    print("Final model saved.")


if __name__ == '__main__':
    main()


