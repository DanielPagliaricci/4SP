#!/usr/bin/env python3
# Train from preprocessed NPZ: mirrors train_heatmap_gemini.py

import os, json, sys, importlib.util
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D,
    BatchNormalization, Layer, Concatenate
)
from tensorflow.keras.applications import ResNet50

# ─────────────────────────  GPU setup helper  ─────────────────────────
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

# ─────────────────────────  paths / params  ─────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))  # contains X_data.npy, Y_data.npy
LOG_DIR   = os.path.join(BASE_DIR, "logs")
MODEL_OUT = os.path.join(BASE_DIR, "visible_keypoints_model.keras")

DATA_DIR = BASE_DIR

HEATMAP_VIZ_DIR = os.path.join(BASE_DIR, "plots")
SAMPLE_IMAGE_PATH = os.path.join(BASE_DIR, "image_sample.png")

BATCH     = 128
EPOCHS    = 20#500
VAL_SPL   = 0.15
LAMBDA_L1 = 5.0

FINE_TUNE_EPOCHS = 500

LR        = 5e-4   # initial learning rate with backbone frozen
FT_LR     = 2e-6   # fine-tuning learning rate with backbone unfrozen
REDUCE_LR_FACTOR = 0.8
REDUCE_LR_PATIENCE = 10

USE_EARLYSTOP = False
EARLYSTOP_PATIENCE = 40
USE_FINE_TUNE = False

# ═══════════════════════════════  losses  ══════════════════════════════
def focal_bce(y_true, y_pred, gamma_focal=2.0, alpha_focal=0.25):
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    epsilon = 1e-7
    pt_clipped = tf.clip_by_value(pt, epsilon, 1.0 - epsilon)
    alpha_factor_t = y_true * alpha_focal + (1 - y_true) * (1 - alpha_focal)
    modulating_factor = tf.pow(1.0 - pt, gamma_focal)
    cross_entropy_component = -tf.math.log(pt_clipped)
    pixel_focal_loss = alpha_factor_t * modulating_factor * cross_entropy_component
    keypoint_present_mask = tf.cast(tf.reduce_max(y_true, axis=[1, 2], keepdims=True) > 0, tf.float32)
    masked_loss = pixel_focal_loss * keypoint_present_mask
    num_present_keypoints_in_batch = tf.maximum(tf.reduce_sum(keypoint_present_mask), 1e-6)
    return tf.reduce_sum(masked_loss) / num_present_keypoints_in_batch

def masked_l1(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1.), tf.float32)
    return tf.reduce_sum(tf.abs(y_true - y_pred) * mask) / tf.maximum(tf.reduce_sum(mask), 1e-6)

# ═══════════════════  temperature-scaled SoftArgMax  ════════════════════
class SoftArgMax(Layer):
    def __init__(self, beta=15., **kw):
        super().__init__(**kw)
        self.beta = beta

    def call(self, x_in):
        b = tf.shape(x_in)[0]
        h = tf.shape(x_in)[1]; w = tf.shape(x_in)[2]
        n = x_in.shape[-1]
        yv = tf.cast(tf.range(h), tf.float32)
        xv = tf.cast(tf.range(w), tf.float32)
        y_grid, x_grid = tf.meshgrid(yv, xv, indexing="ij")
        flat_x = tf.reshape(x_in, (b, h * w, n))
        p = tf.nn.softmax(self.beta * flat_x, axis=1)
        x_grid_flat = tf.reshape(x_grid, (1, h * w, 1))
        y_grid_flat = tf.reshape(y_grid, (1, h * w, 1))
        ex = tf.reduce_sum(p * x_grid_flat, axis=1)
        ey = tf.reduce_sum(p * y_grid_flat, axis=1)
        hf, wf = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
        ex_norm = ex / tf.maximum(wf - 1.0, 1e-6)
        ey_norm = ey / tf.maximum(hf - 1.0, 1e-6)
        coords = tf.stack([ex_norm, ey_norm], axis=-1)
        return tf.reshape(coords, (b, n * 2))

# ═════════════════════════════  U-Net style model  ═══════════════════════════════
def build_model(img_size, num_v, heat_size):
    inp = Input(shape=(img_size, img_size, 3), name="image_input")
    base_resnet_model = ResNet50(include_top=False, weights="imagenet", input_tensor=inp)

    s1_features = base_resnet_model.get_layer("conv1_relu").output
    s2_features = base_resnet_model.get_layer("conv2_block3_out").output
    s3_features = base_resnet_model.get_layer("conv3_block4_out").output
    s4_features = base_resnet_model.get_layer("conv4_block6_out").output

    bottleneck = base_resnet_model.output
    f = [1024, 512, 256, 128, 64]

    d1 = UpSampling2D(2, name="d1_up")(bottleneck)
    skip4 = Conv2D(f[0]//2, 1, padding="same", activation="relu", name="skip4_1x1")(s4_features)
    d1 = Concatenate(name="d1_concat")([d1, skip4])
    d1 = Conv2D(f[0], 3, padding="same", activation="relu", name="d1_conv1")(d1)
    d1 = BatchNormalization(name="d1_bn1")(d1)
    d1 = Conv2D(f[0], 3, padding="same", activation="relu", name="d1_conv2")(d1)
    d1 = BatchNormalization(name="d1_bn2")(d1)

    d2 = UpSampling2D(2, name="d2_up")(d1)
    skip3 = Conv2D(f[1]//2, 1, padding="same", activation="relu", name="skip3_1x1")(s3_features)
    d2 = Concatenate(name="d2_concat")([d2, skip3])
    d2 = Conv2D(f[1], 3, padding="same", activation="relu", name="d2_conv1")(d2)
    d2 = BatchNormalization(name="d2_bn1")(d2)
    d2 = Conv2D(f[1], 3, padding="same", activation="relu", name="d2_conv2")(d2)
    d2 = BatchNormalization(name="d2_bn2")(d2)

    d3 = UpSampling2D(2, name="d3_up")(d2)
    skip2 = Conv2D(f[2]//2, 1, padding="same", activation="relu", name="skip2_1x1")(s2_features)
    d3 = Concatenate(name="d3_concat")([d3, skip2])
    d3 = Conv2D(f[2], 3, padding="same", activation="relu", name="d3_conv1")(d3)
    d3 = BatchNormalization(name="d3_bn1")(d3)
    d3 = Conv2D(f[2], 3, padding="same", activation="relu", name="d3_conv2")(d3)
    d3 = BatchNormalization(name="d3_bn2")(d3)

    d4 = UpSampling2D(2, name="d4_up")(d3)
    skip1 = Conv2D(f[3]//2, 1, padding="same", activation="relu", name="skip1_1x1")(s1_features)
    d4 = Concatenate(name="d4_concat")([d4, skip1])
    d4 = Conv2D(f[3], 3, padding="same", activation="relu", name="d4_conv1")(d4)
    d4 = BatchNormalization(name="d4_bn1")(d4)
    d4 = Conv2D(f[4], 3, padding="same", activation="relu", name="d4_conv2")(d4)
    d4 = BatchNormalization(name="d4_bn2")(d4)

    heat = Conv2D(num_v, 1, activation="sigmoid", name="heatmaps", padding="same")(d4)
    coords = SoftArgMax(beta=15., name="coords")(heat)

    final_model = tf.keras.Model(inputs=inp, outputs=[heat, coords], name="keypoint_detector_unet")
    return final_model, base_resnet_model

# ──────────────────────────────── data I/O ─────────────────────────────────
def load_np_datasets():
    x_path = os.path.join(DATA_DIR, "X_data.npy")
    y_npz_path = os.path.join(DATA_DIR, "Y_data.npz")
    y_npy_legacy_path = os.path.join(DATA_DIR, "Y_data.npy")

    X = np.load(x_path)

    heatmaps = None
    coords = None
    if os.path.exists(y_npz_path):
        Y = np.load(y_npz_path, allow_pickle=False)
        # Avoid printing Y.shape (Y is NpzFile). Inspect keys instead.
        print(f"Loaded X: {X.shape}, Y keys: {list(Y.keys())}")
        heatmaps = Y["heatmaps"]
        coords = Y["coords"]
    elif os.path.exists(y_npy_legacy_path):
        # Backward-compat: older runs saved a dict into a .npy (object). Safe here since it's local.
        print("Warning: Falling back to legacy Y_data.npy format (uses pickle). Consider regenerating NPZ.")
        legacy = np.load(y_npy_legacy_path, allow_pickle=True).item()
        heatmaps = legacy["heatmaps"]
        coords = legacy["coords"]
    else:
        raise FileNotFoundError(f"Could not find Y labels file at {y_npz_path} or {y_npy_legacy_path}")
    assert X.shape[0] == heatmaps.shape[0] == coords.shape[0], "Mismatched dataset sizes"
    return X.astype(np.float32), heatmaps.astype(np.float32), coords.astype(np.float32)

def make_tf_datasets(X, heatmaps, coords):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    cut = int(num_samples * (1 - VAL_SPL))
    tr_idx, va_idx = indices[:cut], indices[cut:]

    def build_ds(idx):
        x = X[idx]
        h = heatmaps[idx]
        c = coords[idx]
        ds = tf.data.Dataset.from_tensor_slices((x, {"heatmaps": h, "coords": c}))
        ds = ds.shuffle(len(idx)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
        return ds

    return build_ds(tr_idx), build_ds(va_idx), tr_idx, va_idx

# ─────────────────────────────── visualization ─────────────────────────────
def reverse_preprocess_input(img_batch):
    # img_batch is preprocessed by resnet50 preprocess_input (BGR + mean subtraction)
    x = img_batch.copy()
    # Add means back
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # Convert BGR back to RGB
    x = x[..., ::-1]
    x = np.clip(x / 255.0, 0, 1)
    return x

def visualize_before_training(X, heatmaps, out_dir, num_samples=4, img_sz=None, heat_sz=None):
    os.makedirs(out_dir, exist_ok=True)
    n = min(num_samples, X.shape[0])
    imgs = reverse_preprocess_input(X[:n])
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i in range(n):
        heat = heatmaps[i]
        heat_vis = np.max(heat, axis=-1)
        heat_vis_resized = tf.image.resize(heat_vis[..., None], (img_sz, img_sz), method="bilinear").numpy().squeeze()
        axes[0, i].imshow(imgs[i]); axes[0, i].set_axis_off(); axes[0, i].set_title(f"sample {i}")
        axes[1, i].imshow(heat_vis_resized, cmap="hot"); axes[1, i].set_axis_off()
    plt.tight_layout()
    path = os.path.join(out_dir, "pretrain_sanity_grid.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Pre-training sanity grid saved → {path}")

# ─────────────────────────────── epoch-end visualization ─────────────────────────
def _load_preproc_from_preprocess_script():
    preproc_path = os.path.join(BASE_DIR, "TANGO_preprocess_heatmap.py")
    try:
        spec = importlib.util.spec_from_file_location("tango_preproc", preproc_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["tango_preproc"] = module
        spec.loader.exec_module(module)  # type: ignore
        return module.preprocess_image_for_model
    except Exception as e:
        print(f"Warning: could not import preprocess_image_for_model from {preproc_path}: {e}")
        return None

class EpochHeatmapViz(tf.keras.callbacks.Callback):
    def __init__(self, sample_path, out_dir, img_sz, heat_sz, num_v, period=5):
        super().__init__()
        self.sample_path = sample_path
        self.out_dir = out_dir
        self.img_sz = img_sz
        self.heat_sz = heat_sz
        self.num_v = num_v
        self.period = period
        self.preproc_func = _load_preproc_from_preprocess_script()
        os.makedirs(self.out_dir, exist_ok=True)

    def _fallback_preprocess(self, image_path):
        img_bytes = tf.io.read_file(image_path)
        img = tf.io.decode_image(img_bytes, channels=3, dtype=tf.uint8)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img_resized = tf.image.resize(img, (self.img_sz, self.img_sz))
        img_scaled = img_resized * 255.0
        from tensorflow.keras.applications.resnet50 import preprocess_input as _pp
        img_ready = _pp(img_scaled)
        return img_ready.numpy(), img_resized.numpy()

    def on_epoch_end(self, epoch, logs=None):
        # Only run every `period` epochs to reduce disk usage
        if (epoch + 1) % max(int(self.period), 1) != 0:
            return
        if not os.path.exists(self.sample_path):
            if epoch == 0:
                print(f"Sample image not found at {self.sample_path}; skipping epoch viz.")
            return
        try:
            if self.preproc_func is not None:
                x_ready, img_disp = self.preproc_func(self.sample_path)
            else:
                x_ready, img_disp = self._fallback_preprocess(self.sample_path)
        except Exception as e:
            print(f"Failed to preprocess sample image: {e}")
            return

        x_in = np.expand_dims(x_ready.astype(np.float32), axis=0)
        try:
            pred_heat, pred_coords = self.model.predict(x_in, verbose=0)
        except Exception as e:
            print(f"Prediction failed during epoch-end viz: {e}")
            return

        heat = pred_heat[0]  # H x W x N
        n_v = heat.shape[-1]
        n_to_show = min(11, n_v)

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        # Flatten axes for easy indexing
        grid = axes.reshape(-1)
        # Slot 0: preprocessed image
        grid[0].imshow(np.clip(img_disp, 0, 1)); grid[0].set_axis_off(); grid[0].set_title("input")
        # Remaining: first 11 heatmaps
        for i in range(n_to_show):
            hm = heat[..., i]
            hm_up = tf.image.resize(hm[..., None], (self.img_sz, self.img_sz), method="bilinear").numpy().squeeze()
            grid[i + 1].imshow(hm_up, cmap="hot")
            grid[i + 1].set_axis_off(); grid[i + 1].set_title(f"v{i}")
        # Any leftover cells blank
        for j in range(n_to_show + 1, 12):
            grid[j].set_axis_off()
        plt.tight_layout()
        out_path = os.path.join(self.out_dir, f"epoch_{epoch+1:03d}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        # Optional console note
        print(f"Saved epoch-end heatmap grid → {out_path}")

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    configure_gpu()
    print(f"Loading preprocessed data from {DATA_DIR} ...")
    X, H, C = load_np_datasets()
    IMG_SZ = X.shape[1]
    HEAT_SZ = H.shape[1]
    NUM_V = H.shape[-1]
    print(f"Loaded: X {X.shape}, heatmaps {H.shape}, coords {C.shape}, NUM_V={NUM_V}")

    visualize_before_training(X, H, LOG_DIR, num_samples=4, img_sz=IMG_SZ, heat_sz=HEAT_SZ)

    train_ds, val_ds, tr_idx, va_idx = make_tf_datasets(X, H, C)

    model, resnet_backbone = build_model(IMG_SZ, NUM_V, HEAT_SZ)
    if resnet_backbone:
        resnet_backbone.trainable = False
        print(f"ResNet50 backbone '{resnet_backbone.name}' frozen for initial training.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss        = {"heatmaps": focal_bce, "coords": masked_l1},
        loss_weights= {"heatmaps": 1.0,       "coords": LAMBDA_L1}
    )
    model.summary(line_length=150)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_images=False),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(LOG_DIR, "best_model.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, "training_log.csv"), append=True),
        EpochHeatmapViz(
            sample_path=SAMPLE_IMAGE_PATH,
            out_dir=HEATMAP_VIZ_DIR,
            img_sz=IMG_SZ,
            heat_sz=HEAT_SZ,
            num_v=NUM_V,
            period=5,
        )
    ]

    # Conditionally enable EarlyStopping
    if USE_EARLYSTOP:
        callbacks.insert(2, tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLYSTOP_PATIENCE, restore_best_weights=True, verbose=1
        ))

    print(f"Starting training with ResNet50 backbone frozen (lr={LR})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    if USE_FINE_TUNE:
        print("\nUnfreezing ResNet50 backbone for fine-tuning...")
        if resnet_backbone:
            resnet_backbone.trainable = True
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR), 
                loss        = {"heatmaps": focal_bce, "coords": masked_l1},
                loss_weights= {"heatmaps": 1.0,       "coords": LAMBDA_L1}
            )
            print(f"ResNet50 backbone '{resnet_backbone.name}' unfrozen. Recompiled model with lower learning rate.")
            model.summary(line_length=150)

            fine_tune_epochs = FINE_TUNE_EPOCHS
            print(f"Starting fine-tuning for {fine_tune_epochs} epochs (lr={FT_LR})...")
            history_fine = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS + fine_tune_epochs,
                initial_epoch=history.epoch[-1] + 1,
                callbacks=callbacks
            )
    else:
        print("\nSkipping fine-tuning stage (USE_FINE_TUNE=False).")

    model.save(MODEL_OUT)
    print(f"✅ Model saved → {MODEL_OUT}")


