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
# Calculate average color of an image
def avg_color(img):
    return np.mean(img, axis=(0, 1))

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
    df['avg_color'] = df['img'].apply(avg_color)
    return df

def show_image(img):
    plt.imshow(img.astype(np.uint8))

def reshape_image(image):
    return image.reshape(-1, 3)

# This is a rather slow and ugly function.
# this is really a dumb thing to do anyway.
def show_color_density(dd):
    '''
    # this is going to be rather slow
    pixels = []
    for img in dd.img:  
        reshaped_img = img.reshape(-1,3)
        pixels.append(reshaped_img)
    pixels = np.vstack(pixels)
    '''
    '''
    packed_pixels = np.array(dd['img'].tolist())
    pixels = packed_pixels.reshape(-1,3)
    
    '''
    #pixels = np.concatenate([img.reshape(-1, 3) for img in dd['img']])
    pixels = np.concatenate(dd['img'].apply(reshape_image).tolist())
    
    non_black_pixels = pixels[np.any(pixels > [0, 0, 0], axis=-1)]
    df_pixels = pd.DataFrame(non_black_pixels, columns=['Red', 'Green', 'Blue'])

    for color in ['Red', 'Green', 'Blue']:
        sns.kdeplot(df_pixels[color], color=color.lower(),  alpha=0.5, label=color)
    plt.xlabel('Color Value (0-255)')
    plt.legend()
    plt.grid()

def show_df(dd, seed=None):
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