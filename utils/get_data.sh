conda run --no-capture-output -n data200s gdown https://drive.google.com/uc?id=1-be6hZE61fYibBQ23VUUAg8oE8AfpPq1 -O 'data/sp24_grad_project_data.zip'
cd data; conda run --no-capture-output -n data200s unzip 'sp24_grad_project_data.zip' 
