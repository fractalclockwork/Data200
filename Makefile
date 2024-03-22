# See the README.md in this directory
# and the Source diectory for a detailed decription.
.PHONY: environment setup

ENV_NAME = data200s
DATA_FILE = 'Data/sp24_grad_project_data.zip'

environment:
ifneq (,$(shell conda list --name $(ENV_NAME)))
	@echo '$(ENV_NAME) is already installed.'
else
	@echo '$(ENV_NAME) needs to be installed.'
	conda env create -f environment.yaml
endif

setup: environment $(DATA_FILE)
data: $(DATA_FILE)
	(cd Data; unzip ../$(DATA_FILE))

$(DATA_FILE):
	gdown https://drive.google.com/uc?id=1-be6hZE61fYibBQ23VUUAg8oE8AfpPq1 -cO $(DATA_FILE) 

#test:
#	$(MAKE) -C Source test

run:
	bash 'Utils/run_conda.sh' 

clean:
	$(MAKE) -C Source clean 

