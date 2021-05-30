"""Train an EfficientNetB4 model to predict GBM vs PCNSL.

This requires TensorFlow >= 2.3.0.
"""

import argparse
import math
from pathlib import Path
import pickle
from typing import Tuple, Union

import h5py
import numpy as np
import tensorflow as tf

PathType = Union[str, Path]


def augment_base(x, y):
    x = tf.image.random_brightness(x, max_delta=2)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_hue(x, max_delta=0.25)
    return x, y


def augment_base_and_noise(x, y):
    x, y = augment_base(x, y)
    # Apply gaussian noise to fraction of samples.
    x = tf.cond(
        pred=tf.random.uniform([]) < 0.1,
        true_fn=lambda: x
        + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.05, dtype=x.dtype),
        false_fn=lambda: x,
    )
    return x, y


def load_data_into_train_val(
    data_path: PathType, augmentation: str
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    print("Loading data from HDF5...", flush=True)
    with h5py.File(str(data_path)) as f:
        x_gbm = f["/gbm/380_380/features"][:]
        y_gbm = f["/gbm/380_380/labels"][:]
        x_pcnsl = f["/pcnsl/380_380/features"][:]
        y_pcnsl = f["/pcnsl/380_380/labels"][:]
    print("gbm features shape", x_gbm.shape)
    print("gbm labels shape", y_gbm.shape)
    print("pcnsl features shape", x_pcnsl.shape)
    print("pcnsl labels shape", y_pcnsl.shape, flush=True)
    x = np.concatenate((x_gbm, x_pcnsl)).astype(np.float32)
    y = np.concatenate((y_gbm, y_pcnsl)).astype(np.float32)

    # Shuffle the samples. The shuffling is the same for features and labels.
    print("Shuffling samples ...", flush=True)
    shuffle_inds = np.arange(y.shape[0])
    np.random.seed(42)
    np.random.shuffle(shuffle_inds)
    x = x[shuffle_inds]
    y = y[shuffle_inds]
    inds = np.random.choice([0, 1], size=y.size, p=[0.85, 0.15])
    x_train, y_train = x[inds == 0], y[inds == 0]
    x_val, y_val = x[inds == 1], y[inds == 1]

    # Create tf.data.Dataset
    print("Creating tf.data.Dataset ...", flush=True)
    batch_size = 8
    dset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if augmentation == "none":
        print("Not applying augmentation.")
    elif augmentation == "base":
        print("Applying 'base' augmentation.")
        dset_train = dset_train.map(augment_base)
    elif augmentation == "base_and_noise":
        print("Applying 'base_and_noise' augmentation.")
        dset_train = dset_train.map(augment_base)
    else:
        raise ValueError(f"unknown augmentation type: {augmentation}")
    dset_train = dset_train.shuffle(1000, reshuffle_each_iteration=True)
    dset_train = dset_train.batch(batch_size)

    dset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    dset_val = dset_val.batch(batch_size)

    return dset_train, dset_val


def get_model() -> tf.keras.Model:
    print("Creating model ...", flush=True)
    tfkl = tf.keras.layers

    # This is from the tf.keras.applications.efficientnet implementation in version
    # 2.5.0 of tensorflow.
    DENSE_KERNEL_INITIALIZER = {
        "class_name": "VarianceScaling",
        "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
    }

    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        input_shape=(380, 380, 3),
        weights="imagenet",
    )
    base_model.activity_regularizer = tf.keras.regularizers.l2(l=0.01)

    _x = tfkl.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    _x = tfkl.Dropout(0.5)(_x)
    _x = tfkl.Dense(
        1,
        activation="sigmoid",
        name="predictions",
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
    )(_x)
    model = tf.keras.Model(inputs=base_model.input, outputs=_x)
    return model


def main(
    data_path: PathType,
    checkpoint_prefix: PathType,
    augmentation: str = "none",
    epochs: int = 300,
):

    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-04),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )

    def schedule_lr(epoch):
        if epoch < 50:
            return 1e-04
        else:
            return 1e-04 * math.exp(0.015 * (50 - epoch))

    checkpoint_prefix = Path(checkpoint_prefix)
    checkpoint_prefix.mkdir(parents=True, exist_ok=False)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(schedule_lr, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_prefix / "ckpt_{epoch:03d}_{val_loss:0.4f}.hdf5"),
            save_best_only=True,
            verbose=1,
        ),
    ]

    dset_train, dset_val = load_data_into_train_val(
        data_path=data_path, augmentation=augmentation
    )
    print("Beginning training...", flush=True)
    history = model.fit(
        dset_train,
        epochs=epochs,
        validation_data=dset_val,
        callbacks=callbacks,
        verbose=2,
    )

    # We save as pickle and not as json because the numpy arrays in this dictionary
    # do not play nicely with json. Pickle is fine with it, though.
    print("Saving training/validation history to pickle file ...")
    with (checkpoint_prefix / "history.pkl").open("wb") as f:
        pickle.dump(history.history, f)


def get_parsed_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_path", help="Path to HDF5 with data.")
    p.add_argument("ckpt_prefix", help="Directory in which to save checkpoints.")
    p.add_argument(
        "--augmentation",
        choices=["none", "base", "base_and_noise"],
        default="none",
        help="Type of augmentation to apply to training data.",
    )
    p.add_argument("--epochs", type=int, default=300, help="Number of epochs to train.")
    args = p.parse_args()
    args.data_path = Path(args.data_path)
    args.ckpt_prefix = Path(args.ckpt_prefix)
    return args


if __name__ == "__main__":

    args = get_parsed_args()

    print("-" * 40)
    print("Arguments passed to this script:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print("-" * 40, flush=True)

    main(
        data_path=args.data_path,
        checkpoint_prefix=args.ckpt_prefix,
        augmentation=args.augmentation,
        epochs=args.epochs,
    )
    print("Reached end of python script.")
