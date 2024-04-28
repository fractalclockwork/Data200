import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import zipfile
import re
import io
from tqdm import tqdm

import pickle

#Let's make some enhancements! 
from tensorflow.keras.models import load_model

def load_model_data(name):
    '''
    Loads model, history, notes
    '''
    path = '../Data'
    model_path = path + '/models/'+ f'{name}'
    model_history_path = model_path + '/history.pkl'

    print(model_path)
    print(model_history_path)
    
    # Note: Delete models directory to rebuild.
    if (os.path.exists(model_path) & os.path.isfile(model_history_path)): 
        model = load_model(model_path)
        with open(model_history_path, 'rb') as db_file:
            #history = pickle.load(db_file)
            db_pkl = pickle.load(db_file)
            history = db_pkl['history']
            notes = db_pkl['notes']
            print('Gherkin injested.')
        return model, history, notes
    else:
        print("Model not found.")
        return None, None, None
    

def save_model_data(name, model, history, notes=[] ):
    '''
    Save model for transfer learning
    '''
    path = '../Data'
    model_path = path + '/models/'+ f'{name}'
    model_history_path = model_path + '/history.pkl'
    
    model.save(model_path)
    with open(model_history_path, 'wb') as db_file:
        #pickle.dump(history, file = db_file)
        pickle.dump(obj={'history':history,
                    'notes':notes}, file=db_file)    
        print('Gherkin created.')


def show_balance(df):
    display(df.groupby('type')['label'].value_counts())

def read_files_from_zip(zip_file_path, regex_pattern):
    """
    This function reads files from a zip file that matches a given regex pattern.
    It assumes the regex pattern contains two groups (train|test) and a nominative group (some_name).
    
    Parameters:
    zip_file_path (str): The path to the zip file.
    regex_pattern (str): The regex pattern to match the files.

    Returns:
    dict: A dictionary with the name as key, and a sub-dictionary of type and file as values.
    """
    data_dict = {}  # Dictionary with name as key, and sub-dict of type and file as value

   # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # List all files
        all_files = zip_ref.namelist()

        # Filter files based on the regex pattern and create a dictionary
        for file in all_files:
            match = re.search(regex_pattern, file)
            if match:
                filtered = list(match.groups())
                data_type = [s for s in filtered if re.search(r'(test|train)', s)][0]
                filtered.remove(data_type)
                data_name = filtered[0]
                data_dict[data_name] = {'type':data_type, 'file':file}

        # Read in the data for each file
        for name in tqdm(data_dict.keys(), ascii=False, desc=f"Importing data from zip"):         
            with zip_ref.open(data_dict[name]['file']) as f:
                bytes_io = io.BytesIO(f.read())
                
                match = re.search(r'_images', data_dict[name]['file'])
                if match:
                    #print('Reading Image:')
                    data = np.load(bytes_io, allow_pickle=True)
                    images=[image[1] for image in data.items()]
                    data_dict[name]['images'] = images
                else:
                    #print('Reading Label:')
                    data_dict[name]['labels'] = np.load(bytes_io, allow_pickle=True)
    print('Found the following datasets: ', list(data_dict.keys()))
    return data_dict

def data2pd(image_dataset, label_dataset=None):
    # replace type with disaster type
    for key in image_dataset.keys():
        #print(key)
        regex = r'(?:fire|flood|hurricane)'
        matches = re.findall(r'(fire|flood|hurricane)', key)
        disaster_type = matches[0] if matches else None
        image_dataset[key]['type'] = disaster_type
        if label_dataset != None:
            image_dataset[key]['labels'] = label_dataset[key]['labels'] # add labels
    
    df = pd.DataFrame(image_dataset)
    if label_dataset != None:
        df = df.T.explode(['images', 'labels']).drop(columns = ['file']).rename(columns={'labels':'label', 'images':'image'})
    else:
        df = df.T.explode(['images']).drop(columns = ['file']).rename(columns={'images':'image'})
    return df.reset_index()

def load_images(images_path):
    """
    Load images from a specified .npz file.

    Parameters:
    - images_path (str): The file path to the .npz file containing the images.

    Returns:
    - images (list): A list of numpy arrays, each representing an image loaded from the .npz file.
    """
    data = np.load(images_path, allow_pickle=True)
    images = [data[f"image_{i}"] for i in range(len(data.files))]
    return images

def load_labels(labels_path):
    """
    Load labels from a specified .npy file.

    Parameters:
    - labels_path (str): The file path to the .npy file containing the labels.

    Returns:
    - labels (numpy.ndarray): An array of labels loaded from the .npy file.
    """
    labels = np.load(labels_path, allow_pickle=True)
    return labels

def get_images(data_dir, disaster, split="train"):
    """
    Load images from a specified disaster dataset split.

    Args:
        data_dir (str): The directory where the dataset is stored.
        disaster (str): The disaster type of the dataset.
        split    (str): The train or test split (default train).

    Returns:
        list: A list of images (as numpy arrays) from the specified dataset split.
    """
    images_path = os.path.join(data_dir, disaster, f"{split}_images.npz")
    return load_images(images_path)


def get_labels(data_dir, disaster, split="train"):
    """
    Load labels for a specified disaster dataset split.

    Args:
        data_dir (str): The directory where the dataset is stored.
        disaster (str): The disaster type of the dataset.
        split    (str): The train or test split (default train).

    Returns:
        ndarray: The labels for the images in the specified dataset split.
    """
    labels_path = os.path.join(data_dir, disaster, f"{split}_labels.npy")
    return load_labels(labels_path)


def convert_dtype(images, dtype=np.float32):
    """
    Convert the data type of a collection of images.

    Args:
        images (list or dict): The images to convert, either as a list or dictionary of numpy arrays.
        dtype (data-type): The target data type for the images. Defaults to np.float32.

    Returns:
        The converted collection of images, in the same format (list or dict) as the input.
    """
    if isinstance(images, dict):
        return {k: v.astype(dtype) for k, v in images.items()}
    elif isinstance(images, list):
        return [img.astype(dtype) for img in images]
    else:
        raise TypeError("Unsupported type for images. Expected list or dict.")


def plot_label_distribution(labels, ax=None, title="Label Distribution"):
    """
    Plot the distribution of labels.

    Args:
        labels (ndarray): An array of labels to plot the distribution of.
        ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot.
                                             If None, a new figure and axis are created.
        title (str, optional): The title for the plot. Defaults to "Label Distribution".
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True  # Flag indicating a figure was created within this function
    else:
        created_fig = False

    sns.countplot(x=labels, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")

    if created_fig:
        plt.tight_layout()
        plt.show()
