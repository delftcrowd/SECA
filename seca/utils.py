import os
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path_image: str):
    """Loads an image from the specified path

    Parameters
    ----------
    image_path : str
        path to specific image in the dataset
    """

    # Verify image exists
    assert os.path.exists(path_image), "Cannot find image"

    # Load image
    return io.imread(path_image)


def save_image(image, filepath: str, image_name: str, extension: str = ".jpg"):
    """Save image to file

    Parameters
    ----------
    image : [type]
        [description]
    filepath : str
        path for saving the image
    extension : str
        extension of the image. Defaults to jpeg.

    """

    # Verify save directory exists
    assert os.path.exists(filepath), "Output folder does not exist"

    image_name = Path(image_name)
    path_image = os.path.join(filepath, image_name.with_suffix(extension))
    plt.imsave(path_image, image)

    # Verify successful saving
    assert os.path.isfile(
        path_image
    ), "Error saving: cannot find {image_name}{extension}"


def setup_https_requests():
    """Ensure proper handling of https requests for Google API

    Issue and solution are described here: https://github.com/tensorflow/tensorflow/issues/33285

    """
    import requests

    requests.packages.urllib3.disable_warnings()
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context
