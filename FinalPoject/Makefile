# See the README.md in this directory
# and the Source diectory for a detailed decription.
#
# Project Report, Presention Slides and Presentation Video
# maybe found at the following paths...
#
# narrative/
# ├── Final_Project_Presentation.pdf
# ├── Final_Project_Report.pdf
# └── README.md
#
# Video: https://youtu.be/UwdOYl2VrcI

.PHONY: environment setup

ENV_NAME = data200s
DATA_FILE = 'data/sp24_grad_project_data.zip'
MODEL_FILE = 'data/model_data.zip'

environment:
ifneq (,$(shell conda list --name $(ENV_NAME)))
	@echo '$(ENV_NAME) is already installed.'
else
	@echo '$(ENV_NAME) needs to be installed.'
	echo 'y' | conda env create -f environment.yaml
endif

env_clean:
	conda env remove --name data200s

setup: environment $(DATA_FILE) $(MODEL_FILE)

data: $(DATA_FILE)

$(DATA_FILE):
	bash 'utils/get_data.sh'

model: $(MODEL_FILE)

$(MODEL_FILE):
	bash 'utils/get_model.sh'

run:
	bash 'utils/run_conda_model.sh' 

clean:
	$(MAKE) -C Source clean 

release:
	bash 'utils/do_release.sh'

pack_model:
	bash 'utils/pack_model.sh'

