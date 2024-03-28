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
	echo 'y' | conda env create -f environment.yaml
endif

setup: environment $(DATA_FILE)
data: $(DATA_FILE)
	(cd Data; unzip ../$(DATA_FILE))

$(DATA_FILE):
	bash 'Utils/get_data.sh'

#test:
#	$(MAKE) -C Source test

run_eda:
	bash 'Utils/run_conda_eda.sh' 

run_model:
	bash 'Utils/run_conda_model.sh' 

clean:
	$(MAKE) -C Source clean 

release:
	bash 'Utils/do_release.sh'
