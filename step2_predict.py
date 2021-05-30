"""
Give the prediction probability that an magnetic resonance image contains
glioblastoma (GBM) or primary central nervous system lymphoma (PCNSL).

Images must be axial, T1-weighted, contrast-enhanced MR images.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, NamedTuple, Union

import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

PathType = Union[str, Path]

REQUIRED_IMAGE_SIZE = (380, 380)


class Prediction(NamedTuple):
    """Container for one prediction."""

    path: Path
    prob_gbm: float
    prob_pcnsl: float

    def get_class(self) -> str:
        """Return probability and class of top prediction."""
        s = "{:.02f} % {}"
        if self.prob_pcnsl > self.prob_gbm:
            return s.format(self.prob_pcnsl * 100, "PCNSL")
        else:
            return s.format(self.prob_gbm * 100, "GBM")


def get_model() -> tf.keras.Model:
    """Construct EfficientNetB4 for binary classification."""

    tfkl = tf.keras.layers

    # This is from the tf.keras.applications.efficientnet implementation in version
    # 2.5.0 of tensorflow.
    DENSE_KERNEL_INITIALIZER = {
        "class_name": "VarianceScaling",
        "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
    }

    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        input_shape=(REQUIRED_IMAGE_SIZE[0], REQUIRED_IMAGE_SIZE[1], 3),
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


def get_image(path: PathType) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize(size=REQUIRED_IMAGE_SIZE, resample=Image.LANCZOS)
    img = np.asarray(img)
    # Sanity check that our image is uint8 (ie, values are in range 0-255).
    assert img.dtype == np.uint8
    return img


def predict_on_images(
    model_weights: PathType,
    image_paths: Iterable[PathType],
) -> List[Prediction]:

    model = get_model()
    model.load_weights(model_weights)

    predictions: List[Prediction] = []
    for image_path in image_paths:
        image = get_image(image_path)
        prob_pcnsl = model.predict(image[np.newaxis], verbose=0)
        # This code is for the sigmoid-output model (with one output unit). This would
        # have to be changed if using a model trained with two output units and softmax.
        prob_pcnsl = float(prob_pcnsl[0])
        prob_gbm = 1.0 - prob_pcnsl
        p = Prediction(path=Path(image_path), prob_gbm=prob_gbm, prob_pcnsl=prob_pcnsl)
        predictions.append(p)
    return predictions


def write_csv(predictions: Iterable[Prediction], path: PathType):
    with open(path, "w", newline="") as csvfile:
        fieldnames = ["path", "prob_gbm", "prob_pcnsl"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows((p._asdict() for p in predictions))


def get_parsed_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-weights", required=True, help="Path to trained model weights."
    )
    # nargs="+" means variable number of arguments but at least one.
    p.add_argument("image", nargs="+", help="Path(s) to image(s).")
    p.add_argument("--csv", help="Path to CSV output file.")
    args = p.parse_args()
    args.model_weights = Path(args.model_weights)
    args.image = [Path(p) for p in args.image]
    if args.csv is not None:
        args.csv = Path(args.csv)

    if not args.model_weights.exists():
        raise FileNotFoundError(f"model weights file not found: {args.model_weights}")
    for image_path in args.image:
        if not image_path.exists():
            raise FileNotFoundError(f"image file not found: {image_path}")

    return args


if __name__ == "__main__":

    args = get_parsed_args()
    predictions = predict_on_images(
        model_weights=args.model_weights,
        image_paths=args.image,
    )

    for prediction in predictions:
        print(f"{str(prediction.path):<40} {prediction.get_class()}")

    if args.csv is not None:
        print(f"Saving to CSV: {args.csv}")
        write_csv(predictions=predictions, path=args.csv)
