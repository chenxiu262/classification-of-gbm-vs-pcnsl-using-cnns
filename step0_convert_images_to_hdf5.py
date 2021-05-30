"""
Convert image data to HDF5. Expects data to be in `./data/gbm` and `./data/pcnsl`.

Images are resized to a common size, and values are scaled to range [0-255]. Then,
images are stacked into HDF5 datasets. The HDF5 output file contains the datasets
"/TYPE/SIZE/features" and "/TYPE/SIZE/labels", where TYPE is gbm or pcnsl and SIZE
indicates the resized shape of the images (eg 380_380).

GBM samples are given the label 0, and PCNSL samples are given the label 1.
"""

import argparse
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np
from PIL import Image

PathType = Union[str, Path]
IMAGE_DTYPE = np.uint8


def _process_one_image(filename: PathType, size: Tuple[int, int]) -> np.ndarray:
    """Process one image.

    Processing includes resizing the image to a certain size.
    """
    img = Image.open(filename)
    img = img.convert("RGB")
    img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.asarray(img)
    # Sanity check that our image is uint8 (ie, values are in range 0-255).
    assert img.dtype == IMAGE_DTYPE
    return img


def _process_directory_of_images(
    path: PathType, size: Tuple[int, int], label: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a directory of images.

    The processing consists of resizing the images to a common size.

    Return two numpy arrays. The first is a stacked array of images with shape
    (num_images, size[0], size[1], 3), and the second array is an array of labels,
    with shape (num_images). The array of labels is filled with a constant value
    `label`.
    """
    path = Path(path)
    files = list(path.glob("*"))
    nfiles = len(files)
    x = np.zeros((nfiles, *size, 3), dtype=IMAGE_DTYPE)
    print(f"++ Processing images in {path}")
    print(f"0 / {nfiles}", end="\r")
    for j, f in enumerate(files):
        x[j] = _process_one_image(f, size=size)
        print(f"{j + 1} / {nfiles} images", end="\r")
    print()
    return x, np.zeros(nfiles, dtype=np.uint8) + label


def save_one_size_to_hdf5(
    input_path: PathType, output_path: PathType, size: Tuple[int, int]
) -> None:
    """Process images and save them to one HDF5 file."""
    print(f"++ Working on size {size}")
    s = "_".join(map(str, size))
    input_path = Path(input_path)
    x0, y0 = _process_directory_of_images(input_path / "gbm", size=size, label=0)
    x1, y1 = _process_directory_of_images(input_path / "pcnsl", size=size, label=1)
    compression = "gzip"
    print(f"++ Saving data to {output_path}", flush=True)
    with h5py.File(output_path, mode="a") as f:
        f.create_dataset(f"/gbm/{s}/features", data=x0, compression=compression)
        f.create_dataset(f"/gbm/{s}/labels", data=y0, compression=compression)
        f.create_dataset(f"/pcnsl/{s}/features", data=x1, compression=compression)
        f.create_dataset(f"/pcnsl/{s}/labels", data=y1, compression=compression)


def get_parsed_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir", help="Directory containing images.")
    p.add_argument("output_hdf5", help="Path to output HDF5 file.")
    p.add_argument(
        "--size", type=int, nargs=2, default=(380, 380), help="Size of resized images."
    )
    args = p.parse_args()
    args.input_dir = Path(args.input_dir)
    args.output_hdf5 = Path(args.output_hdf5)
    if not (args.input_dir / "gbm").is_dir() or not (args.input_dir / "pcnsl").is_dir():
        raise ValueError(
            "input directory is expected to contain gbm and pcnsl directories."
        )
    return args


if __name__ == "__main__":
    args = get_parsed_args()

    print("-" * 40)
    print("Arguments passed to this script:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print("-" * 40, flush=True)

    save_one_size_to_hdf5(
        input_path=args.input_dir,
        output_path=args.output_hdf5,
        size=args.size,
    )
    print("Reached end of python script.")
