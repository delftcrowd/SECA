from __future__ import absolute_import, division, print_function, unicode_literals
from keras.applications.inception_v3 import preprocess_input as inception_preproc
from tensorflow.keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import saliency
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from ..utils import load_image


def saliency_map_extraction(path_images, path_heatmaps):
    
    sess = K.get_session()
    graph = sess.graph
    # images_dir = "/content/drive/MyDrive/example_data/test"
    # heatmap_dir = "content/example_heatmaps/"
    num_of_images = 3
    with graph.as_default():  # registers graph as default graph. Operations will be added to the graph
        
        model, y, neuron_selector, prediction, images = setup_model(graph, weigths='imagenet')
        verify_images()
        selected_images = upload_data_to_model(path_images, num_of_images)
        # Construct the saliency object.
        gradient_saliency = saliency.tf1.GradientSaliency(graph, sess, y, images)

        for i, img in enumerate(tqdm(selected_images)):
            
            # Create the folder if it does not exist
            create_output_folder(path_heatmaps, selected_images)
                
            # Skip if heatmap is already extracted
            if os.path.exists(os.path.join(path_heatmaps, selected_images[i])):
                continue

            # Run model
            image = load_image(os.path.join(path_images, img))
            
            smoothgrad_mask_grayscale = run_model(sess, gradient_saliency, image, prediction, stdev_spread=.05, nsamples=10)

            # Plot and save images
            path_image = os.path.normpath(os.path.join(path_heatmaps + selected_images[i]))
            save_image(image, path_image)

            path_heatmap = os.path.normpath(os.path.join(path_heatmaps + selected_images[i][:-5] + '_heatmap' + selected_images[i][-5:]))
            export_heatmap(smoothgrad_mask_grayscale, path_heatmap)



def setup_model(graph, weights: str = "imagenet"):
    """[summary]

    Parameters
    ----------
    weights : [type], optional
        [description], by default "imagenet"

    Returns
    -------
    [type]
        [description]
    """


    model = InceptionV3(weights='imagenet')
    logits = graph.get_tensor_by_name('predictions/Softmax:0')
    neuron_selector = tf.placeholder(tf.int32)  # Used to select the logit of the prediction
    y = logits[0][neuron_selector]  # logit of prediction
    prediction = tf.argmax(logits, 1)
    images = graph.get_tensor_by_name('input_1:0') 

    return model, y, neuron_selector, prediction, images

def verify_images(images_dir):
    """Verify images to be analysed

    - Check file extensions
    - Check size (not zero)
    - Print names
    """
    print(([name for name in os.listdir(images_dir)]))

def upload_data_to_model(path_images: str, sample_size: int = None, target_size: tuple = (299, 299), batch_size: int = 1):
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
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    """
    
    test_datagen = ImageDataGenerator(preprocessing_function=inception_preproc)
    test_generator = test_datagen.flow_from_directory(path_images, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
    
    num_images = len(test_generator.filenames) if not sample_size else sample_size

    if num_images > len(test_generator.filenames):
        print(len(test_generator.filenames))
        raise ValueError('The number of annotations cannot be higher than the number of available images.')
    elif num_images == 0:
        raise ValueError('The number of annotations needs to be greater than zero.')
    
    image_selection = np.random.choice(len(test_generator.filenames), num_images)
    selected_images = np.array(test_generator.filenames)[image_selection]
    
    return selected_images

def create_output_folder(path_heatmap, selected_images):
    """[summary]

    Parameters
    ----------
    path_heatmap : [type]
        [description]
    selected_images : [type]
        [description]
    """
    if not os.path.exists(os.path.join(path_heatmap, selected_images[i].split('/')[0])):
        os.makedirs(os.path.join(path_heatmap, selected_images[i].split('/')[0]))
    

def run_model(sess, gradient_saliency, image, prediction, stdev_spread=.05, nsamples=10):
    """[summary]

    Parameters
    ----------
    image : [type]
        [description]
    prediction : [type]
        [description]
    stdev_spread : [type], optional
        [description], by default .05
    nsamples : [type], optional
        [description], by default 10

    Returns
    -------
    [type]
        [description]
    """
    
    im = inception_preproc(image)
    # Predict label
    y_pred = sess.run(prediction, feed_dict={images: [im]})[0]
    # Compute the vanilla mask and the smoothed mask.
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, stdev_spread=stdev_spread, nsamples=nsamples, feed_dict={neuron_selector: y_pred})
    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    smoothgrad_mask_grayscale = saliency.tf1.VisualizeImageGrayscale(smoothgrad_mask_3d)

    return smoothgrad_mask_grayscale

def save_image(image, path_image):
    """[summary]

    Parameters
    ----------
    image : [type]
        [description]
    path_image : [type]
        [description]
    """
    # plt.imsave(os.path.normpath(os.path.join(heatmap_dir + selected_images[i])), image)
    plt.imsave(path_image, image)

def export_heatmap(image, path_image, colormap="inferno"):
    """[summary]

    Parameters
    ----------
    image : [type]
        [description]
    path_image : [type]
        [description]
    colormap : [type], optional
        [description], by default "inferno"
    """

    cm = plt.get_cmap(colormap)
    colored_heatmap = cm(image)  # RGBA (A contains colormap) -> convert o RGB via rgba2rgb
    image_overlay = 0.5 * (image/255) + 0.5 * rgba2rgb(colored_heatmap)  # img1*alpha + img2*(1-alpha)
    # plt.imsave(os.path.normpath(os.path.join(heatmap_dir + selected_images[i][:-5] + '_heatmap' + selected_images[i][-5:])), image_overlay)
    plt.imsave(path_image, image_overlay)
