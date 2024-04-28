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

from tensorflow.keras.models import load_model

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

def show_balance(df):
    display(df.groupby('type')['label'].value_counts())



def load_model_data(name, path='../Data'):
    '''
    Loads model, history, notes
    '''
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


def save_model_data(name, model, history, notes=[], path = '../Data' ):
    '''
    Save model for transfer learning
    '''
    model_path = path + '/models/'+ f'{name}'
    model_history_path = model_path + '/history.pkl'
    
    model.save(model_path)
    with open(model_history_path, 'wb') as db_file:
        #pickle.dump(history, file = db_file)
        pickle.dump(obj={'history':history,
                    'notes':notes}, file=db_file)    
        print('Gherkin created.')
