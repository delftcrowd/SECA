# General imports
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from pathlib import Path
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Machine-learning imports
from keras.applications.inception_v3 import preprocess_input as inception_preproc
from tensorflow.keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import saliency
import tensorflow.compat.v1 as tf
import tensorflow.python.keras.backend as K

# Local imports
from .utils import load_image, save_image


def saliency_map_extraction(
    path_images: str, path_output: str, num_of_images: int = None
) -> None:
    """_summary_

    Parameters
    ----------
    path_images : str
        path to the image dataset
    path_output : str
        path to the
    num_of_images : int
        number of images to select from the dataset, defaults to None
    """

    # Verify input parameters
    assert os.path.exists(path_images), f"Cannot find directory with images: {path_images}"
    assert os.path.exists(path_output), f"Cannot find directory for output: {path_output}"

    # Initialize session
    sess = K.get_session()
    graph = sess.graph

    # Check dataset
    # verify_images(path_images)
    selected_images = upload_data_to_model(path_images, num_of_images)

    with graph.as_default():  # registers graph as default graph. Operations will be added to the graph

        model, y, neuron_selector, prediction, images = setup_model(
            graph, weights="imagenet"
        )

        # Construct the saliency object.
        gradient_saliency = saliency.tf1.GradientSaliency(graph, sess, y, images)

        for img in tqdm(selected_images):

            path_image = Path(img)

            # Create the folder
            output_folder = os.path.join(path_output, path_image.parents[0])
            create_output_folder(output_folder)

            # Skip if heatmap is already extracted
            path_export_heatmap = os.path.join(
                output_folder, f"{path_image.stem}_heatmap.jpg"
            )
            if not path_export_heatmap:
                continue

            # Load image
            image = load_image(os.path.join(path_images, path_image))
            im = inception_preproc(image)

            # Predict label
            y_pred = sess.run(prediction, feed_dict={images: [im]})[0]

            # Compute the vanilla mask and the smoothed mask.
            smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(
                im, stdev_spread=0.05, nsamples=10, feed_dict={neuron_selector: y_pred}
            )

            # Call the visualization methods to convert the 3D tensors to 2D grayscale.
            smoothgrad_mask_grayscale = saliency.tf1.VisualizeImageGrayscale(
                smoothgrad_mask_3d
            )

            # Save images and heatmaps
            save_image(image, output_folder, path_image.name)
            export_heatmap(image, smoothgrad_mask_grayscale, path_export_heatmap)


def setup_model(graph, weights: str = "imagenet"):
    """Set up the ML model

    Parameters
    ----------
    graph : tensorflow.python.framework.ops.Graph
        Keras data graph
    weights : str, optional
        One of None (random initialization), imagenet (pre-training on ImageNet),
        or the path to the weights file to be loaded. Default to imagenet.

    Returns
    -------
    [type]
        [description]
    """

    model = InceptionV3(weights=weights)
    logits = graph.get_tensor_by_name("predictions/Softmax:0")
    neuron_selector = tf.placeholder(
        tf.int32
    )  # Used to select the logit of the prediction
    y = logits[0][neuron_selector]  # logit of prediction
    prediction = tf.argmax(logits, 1)
    images = graph.get_tensor_by_name("input_1:0")

    return model, y, neuron_selector, prediction, images


def verify_images(path_images: str):
    """Verify images in dataset

    Parameters
    ----------
    path_images : str
        directory with images dataset
    """
    # TODO: Check file extensions
    # TODO: Check file size > 0

    # Print image names
    print(([name for name in os.listdir(path_images)]))


def upload_data_to_model(
    path_images: str,
    sample_size: int = None,
    target_size: tuple = (299, 299),
    batch_size: int = 1,
):
    """[summary]

    Parameters
    ----------
    path_images : [type]
        [description]
    sample_size : [type], optional
        [description], by default None
    target_size : [type], optional
        [description], by default (299, 299)
    batch_size : [type], optional
        [description], by default 1

    Returns
    -------
    selected_images : list[str]
        List of selected image paths

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    """

    test_datagen = ImageDataGenerator(preprocessing_function=inception_preproc)
    test_generator = test_datagen.flow_from_directory(
        path_images,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # Select random sample if num_of_images is smaller than dataset
    num_images = len(test_generator.filenames) if not sample_size else sample_size

    if num_images > len(test_generator.filenames):
        print(len(test_generator.filenames))
        raise ValueError(
            "The number of annotations cannot be higher than the number of available images."
        )
    elif num_images == 0:
        raise ValueError("The number of annotations needs to be greater than zero.")

    image_selection = np.random.choice(len(test_generator.filenames), num_images)
    selected_images = np.array(test_generator.filenames)[image_selection]

    return selected_images


def create_output_folder(path: str) -> None:
    """Create the output directory if it doesn't exist

    Parameters
    ----------
    path_output : str
        path to the directory to be created

    """

    if not os.path.exists(path):
        os.makedirs(path)


def export_heatmap(
    image, smoothgrad_mask_grayscale, path_save: str, colormap: str = "inferno"
) -> None:
    """Export the heatmap as overlay on the original image

    Parameters
    ----------
    image : [type]
        original image
    smoothgrad_mask_grayscale : [type]
        heatmap
    colormap : [type], optional
        [description], by default "inferno"
    """

    cm = plt.get_cmap(colormap)

    # RGBA (A contains colormap) -> convert o RGB via rgba2rgb
    colored_heatmap = cm(smoothgrad_mask_grayscale)

    # img1*alpha + img2*(1-alpha)
    image_overlay = 0.5 * (image / 255) + 0.5 * rgba2rgb(colored_heatmap)
    plt.imsave(path_save, image_overlay)
