import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

"""
disaster = disaster_list[0]
images = data[disaster]["images"]
labels = data[disaster]["labels"]

# Since we are doing EDA let's use a panda dataframe to collect our results

# Create a list of our dictionaries
data_list = [{'disaster': disaster, 
              'idx': idx, 
              'label': label, 
              'height': img.shape[0], 
              'width': img.shape[1],
              'image_size': img.shape[0] * img.shape[1],
              'img': np.array(img)} 
             for disaster in disaster_list 
             for idx, (img, label) in enumerate(zip(data[disaster]["images"], data[disaster]["labels"]))]

# Convert our list of dictionaries into a DataFrame
df = pd.DataFrame(data_list)
disaster_regex = r'(fire|hurricane|flood)'
df['disaster_type'] = df['disaster'].str.extract(disaster_regex)
df.sample(5)
"""

def data2df(disaster_list, data):    
    # Create a list of our dictionaries
    data_list = [{'disaster': disaster, 
                  'idx': idx, 
                  'label': label, 
                  'height': img.shape[0], 
                  'width': img.shape[1],
                  'image_size': img.shape[0] * img.shape[1],
                  'img': np.array(img)} 
                 for disaster in disaster_list 
                 for idx, (img, label) in enumerate(zip(data[disaster]["images"], data[disaster]["labels"]))]
    
    # Convert our list of dictionaries into a DataFrame
    df = pd.DataFrame(data_list)
    disaster_regex = r'(fire|hurricane|flood)'
    df['disaster_type'] = df['disaster'].str.extract(disaster_regex)
    return df

def show_image(img):
    plt.imshow(img.astype(np.uint8))

def show_df(dd, seed=5707):
    random_state = seed
    l = dd.shape[0]
    
    #print('len: ', l)
    assert l>0, 'zero length dataframe'

    # plot up to 4x4 images
    if l > 16:
        df = dd.sample(16, random_state=random_state)
    else:
        df = dd
    l = df.shape[0]
    #print('len: ', l)
    if (l > 4):
        rows = l//4
        cols = 4
    else:
        rows = 1
        cols = l
    # print(cols, rows)

    #fig, ax = plt.subplots(rows,cols, figsize=(12,3*rows))
    fig, ax = plt.subplots(rows,cols)
    #print(ax)
    for i, ax in enumerate(ax.flat):
        #display(df.iloc[i])
        img = df.iloc[i].img
        disaster_type = df.iloc[i].disaster_type
        damage = df.iloc[i].label
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'{disaster_type}:{damage}')
        ax.set_xticks([])  # Disable x ticks
        ax.set_yticks([])  # Disable y ticks
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()