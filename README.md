---

# Computer Vision

In disaster situations, it is important for emergency response efforts to have access to quick and accurate information about an area in order to respond effectively. This project will explore how data science techniques can be useful for such efforts.


# Useage
Makefile targets have been provided for convenience of the user. To install and run the EDA notebook type 'make setup run'.


```
$ make 
clean        data         environment  run          setup

setup:
    Configures a environment and downloads data. 

data: 
    Uses 'gdown' to retrive the dataset.

environment:
    Configures a conda environment with all the required tools and libraries.

run:
    Starts the conda enironment and runs EDA notebook.

clean:
    Removes all temporary files.
```


# Files and Directories
```
.
├── Data
├── Docs
├── Figures
├── Source
└── Utils

Data:
    Downloaded and decompressed data
Doc:
    Helpful documentation.
Figure:
    Our generated figure.
Source:
    Our Python libraries and Notebooks.
Utils:
    Helpful utils.
```
