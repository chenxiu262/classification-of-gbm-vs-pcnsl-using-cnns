"""Run Grad-CAM on the test set and save the overlays as images.

Much of this script was made with the help of a Keras blog post
https://keras.io/examples/vision/grad_cam/.
"""

import argparse
from pathlib import Path
from typing import Union

import matplotlib.cm
import numpy as np
from PIL import Image
import tensorflow as tf

PathType = Union[str, Path]


def get_model(weights: PathType = None) -> tf.keras.Model:
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
        weights=None,
    )
    base_model.activity_regularizer = tf.keras.regularizers.l2(l=0.01)

    _x = tfkl.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    _x = tfkl.Dropout(0.5)(_x)
    _x = tfkl.Dense(
        1,
        # No activation here.
        activation=None,
        name="predictions",
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
    )(_x)
    model = tf.keras.Model(inputs=base_model.input, outputs=_x)
    if weights is not None:
        model.load_weights(weights)
    return model


def load_image(path: PathType) -> Image.Image:
    """Load and process an image in the same way that was done for training."""
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize(size=(380, 380), resample=Image.LANCZOS)
    return img


def get_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "top_activation",
) -> np.ndarray:
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis])
        # We invert the predictions if GBM is the predicted class.
        if preds[0][0] < 0.5:
            print("Inverting prediction because model thinks this is GBM")
            preds = 1.0 - preds

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(preds, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_gradcam_overlay(
    img_path: PathType,
    model: tf.keras.Model,
    alpha: float = 0.4,
) -> Image.Image:

    # Load the original image
    original_shape = Image.open(img_path).size
    img = load_image(img_path)
    img_array = np.asarray(img)
    # Sanity check...
    assert img_array.dtype == np.uint8
    img_array = img_array.astype(np.float32)

    # Get heatmap.
    heatmap = get_gradcam_heatmap(img_array=img_array, model=model)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = matplotlib.cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(img.size)
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img = superimposed_img.resize(original_shape, resample=Image.LANCZOS)
    return superimposed_img


def get_parsed_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-weights", required=True, help="Path to trained model weights."
    )
    p.add_argument(
        "--output-dir", required=True, help="Path in which to save heatmap overlays."
    )
    # nargs="+" means variable number of arguments but at least one.
    p.add_argument("image", nargs="+", help="Path(s) to image(s).")
    args = p.parse_args()
    args.model_weights = Path(args.model_weights)
    args.output_dir = Path(args.output_dir)
    args.image = [Path(p) for p in args.image]
    if not args.model_weights.exists():
        raise FileNotFoundError(f"model weights file not found: {args.model_weights}")
    for image_path in args.image:
        if not image_path.exists():
            raise FileNotFoundError(f"image file not found: {image_path}")

    return args


if __name__ == "__main__":

    args = get_parsed_args()

    print("Loading model ...", flush=True)
    model = get_model(args.model_weights)
    for i, image_path in enumerate(args.image):
        print(f"Image {i+1} of {len(args.image)}")
        print(f"Processing {image_path}")
        output_path = args.output_dir / image_path.name
        if output_path.exists():
            raise FileExistsError(output_path)
        overlay = get_gradcam_overlay(img_path=image_path, model=model)
        print(f"Saving to {output_path}")
        overlay.save(output_path)
