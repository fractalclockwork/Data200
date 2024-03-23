---

# Computer Vision

In disaster situations, it is important for emergency response efforts to have access to quick and accurate information about an area in order to respond effectively. This project will explore how data science techniques can be useful for such efforts.



# Usage

Clone the repo and then run 'make setup run'.
This will setup a conda environment then launch a Jupyer Lab session.
If your browser doesn't launch automatically the server is typically avalable at 'localhot:8888'.

```
$ git clone https://github.com/fractalclockwork/Data200.git
```

A number of other makefile targets have been provided for convenience of the user. 

```
$ make 
clean        data         environment  release  run          setup

setup:
    Configures a environment and downloads data. 

data: 
    Uses 'gdown' to retrive the dataset.

environment:
    Configures a conda environment with all the required tools and libraries.

release:
    Package main branch for submission to gradescope.

run:
    Starts the conda enironment and runs EDA notebook.

clean:
    Removes all temporary files.
```


# Directories
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
