import numpy as np
from scipy import ndimage as ndi
from skimage import color
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel, sobel
from tqdm import tqdm

import cv2
import random

def augment_image(img):
    """
    This function takes a numpy array representing an image, 
    flips it horizontally, vertically with equal probability,
    then rotates it left, right, or not at all with equal probability.

    Parameters:
    img (numpy.ndarray): A numpy array of shape (n, m, 3) representing an image.

    Returns:
    numpy.ndarray: The processed image.
    """
    # Flip the image horizontally, vertically with equal probability
    flip_code = random.choice([-1, 1])
    img = cv2.flip(img, flip_code)

    # Rotate the image left, right, or not at all with equal probability
    rotate_code = random.choice([-1, 0, 1])
    if rotate_code == -1:
        # Rotate left (counter-clockwise)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate_code == 1:
        # Rotate right (clockwise)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img

def crop_and_fill(image, N, background='black', replace_scaled=0):
    """
    This function takes an image and a crop size as input, and returns a cropped image.
    The cropping behavior depends on the size of the input image relative to the crop size.
    The background of the output image can be either black or a scaled version of the input image.
    If replace_scaled is a probability between 0 and 1, a proportion of the scaled background will be replaced with pixels from the original image.

    Parameters:
    image (numpy.ndarray): The input image, a numpy array of shape (n, m, 3).
    N (int): The desired crop size.
    background (str): The type of background for the output image. Can be either 'black' or 'scaled'.
    replace_scaled (float): The probability of replacing each pixel in the scaled background with a pixel from the original image.

    Returns:
    numpy.ndarray: The cropped image, a numpy array of shape (N, N, 3).
    """
    n, m, _ = image.shape

    if n==N and m==N:
        return image
        
    if background == 'scaled':
        output = cv2.resize(image, (N, N))
        if replace_scaled > 0:
            indices = np.random.choice(np.arange(N*N), replace=True, size=int(replace_scaled*N*N)) 
            flat_indices = np.random.choice(np.arange(n*m), replace=True, size=len(indices))
            row_indices = flat_indices // m  # convert to row indices
            col_indices = flat_indices % m  # convert to column indices
            replacements = image[row_indices, col_indices].reshape(-1, 3)  # use 2D indices
            output.reshape(-1, 3)[indices] = replacements
    else:
        output = np.zeros((N, N, 3), dtype=image.dtype)

    if n > N and m > N:  # both bigger
        x = np.random.randint(0, n - N)
        y = np.random.randint(0, m - N)
        output = image[x:x+N, y:y+N]

    elif n < N and m < N: # both smalle
        x = (N - n) // 2
        y = (N - m) // 2
        output[x:x+n, y:y+m] = image

    else: # one is smaller and the other is same or bigger
        if n < N and m == N: # grr, edge case
            x = (N - n) // 2
            y = 0
            output[x:x+n, :] = image[:, y:y+N]
        elif n < N and m > N:
            x = (N - n) // 2
            y = np.random.randint(0, m - N)
            output[x:x+n, :] = image[:, y:y+N]
            
        elif n == N and m < N: # grr, edge case
            x = 0
            y = (N - m) // 2 
            output[:, y:y+m] = image[x:x+N, :]
        elif n > N and m < N:
            x = np.random.randint(0, n - N) 
            y = (N - m) // 2 
            output[:, y:y+m] = image[x:x+N, :]

    black_pixels = np.where(np.all(output == [0, 0, 0], axis=-1))
    non_black_pixels = np.where(np.any(image != [0, 0, 0], axis=-1))
    replacements = image[non_black_pixels]
    np.random.shuffle(replacements)
    replacements = np.repeat(replacements, np.ceil(len(black_pixels[0]) / len(replacements)), axis=0)
    output[black_pixels] = replacements[:len(black_pixels[0])]

    return output

def preprocess_images(images, preprocess_function):
    """
    Process a list of images using a specified preprocessing function to prepare features for each image.

    Args:
        images (list): A list of images to be processed. Each image is expected to be a NumPy array.
        preprocess_function (callable): A function that preprocesses a single image. It should take a single image as input
                                        and return a processed feature vector.

    Returns:
        ndarray: A 2D array where each row is a feature vector corresponding to an image.
    """
    # Process each image in the list using the provided preprocessing function
    processed_images = [
        preprocess_function(img) for img in tqdm(images, desc="Processing images")
    ]

    # Stack the processed image features vertically to create a feature matrix
    X = np.vstack(processed_images)
    return X


def get_sobel_features(image):
    """
    Compute Sobel edge detection on an image.

    This function applies the Sobel filter to an image to highlight edges. It returns the edge
    intensity image which can be used as a feature for further analysis or processing.

    Args:
        image (ndarray): An image array in RGB format.

    Returns:
        ndarray: Edge intensity image derived from applying the Sobel filter.

    Read more about Sobel edge detection: https://en.wikipedia.org/wiki/Sobel_operator and https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html
    """
    image_gray = color.rgb2gray(image)
    edges = sobel(image_gray)
    return edges


def generate_gabor_kernel(theta, sigma, frequency):
    """
    Generate a collection of Gabor filter kernels with specified parameters.

    This function creates a list of Gabor kernels each with different orientations and properties
    defined by the input parameter ranges for theta (orientation), sigma (bandwidth), and frequency.

    Args:
        theta_vals (list or array): Range of orientations for the Gabor filter kernels.
        sigma_vals (list or array): Range of sigma values (bandwidths) for the kernels.
        frequency_vals (list or array): Range of frequencies for the kernels.

    Returns:
        list: A list of Gabor kernels with specified parameters.

    Read more about Gabor filters:
    https://en.wikipedia.org/wiki/Gabor_filter and https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html
    for more filters: https://scikit-image.org/docs/stable/api/skimage.filters.html
    """

    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
    return kernel


def get_gabor_features(image, kernel):
    """
    Extract features from an image using a single Gabor filter kernel.

    This function applies the provided Gabor kernel to the input image and returns the filtered image.
    The filtered image represents the Gabor features extracted using the specified kernel, which can capture
    specific texture information based on the kernel's parameters.

    Args:
        image (ndarray): An image array in RGB format.
        kernel (ndarray): A single Gabor kernel.

    Returns:
        ndarray: Filtered image representing Gabor features extracted with the kernel.
    """
    image_gray = color.rgb2gray(image)
    filtered = ndi.convolve(image_gray, kernel, mode="wrap")
    return filtered


def get_local_binary_pattern(image, radius=1, method="uniform"):
    """
    Compute the Local Binary Pattern (LBP) descriptor for an image.

    This function calculates the LBP descriptor for the input image. LBP is a powerful texture descriptor
    which can be used for further image analysis or classification. The function returns the LBP image
    that can be further processed or used directly as a feature.

    Args:
        image (ndarray): An image array in RGB format.
        radius (int): Radius of circular LBP. Defaults to 1.
        method (str): Method to compute LBP. Defaults to 'uniform'.

    Returns:
        ndarray: LBP image.

    Read more about LBP: https://en.wikipedia.org/wiki/Local_binary_patterns and https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    """
    image_gray = color.rgb2gray(image.astype(np.float32))
    n_points = 8 * radius
    lbp = local_binary_pattern(image_gray.astype(np.uint8), n_points, radius, method)
    return lbp

def gabor_filter(img):
    # Define Gabor filter parameters
    ksize = (16, 16)  # size of the filter
    sigma = 3.0       # standard deviation of the Gaussian function
    theta = 1.0       # orientation of the normal to the parallel stripes
    lambd = 25.0      # wavelength of the sinusoidal factor
    gamma = 0.02      # spatial aspect ratio
    psi = 0.0         # phase offset

    # Apply Gabor filter to each color layer
    filtered_img = np.zeros_like(img)
    for i in range(3):
        filtered_img[:, :, i] = cv2.filter2D(img[:, :, i], cv2.CV_32F, cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi))

    return filtered_img
