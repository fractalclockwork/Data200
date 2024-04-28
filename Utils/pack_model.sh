#ls task_*.npy models/task_*_dropout

#conda run --no-capture-output -n data200s gdown https://drive.google.com/uc?id=1-be6hZE61fYibBQ23VUUAg8oE8AfpPq1 -O 'Data/sp24_grad_project_data.zip'
#cd Data; conda run --no-capture-output -n data200s unzip 'sp24_grad_project_data.zip' 

#cd Data;conda run --no-capture-output -n data200s zip -r model_data.zip train_df.pkl test_df.pkl task_*.npy models/task_*_dropout 
cd Data;conda run --no-capture-output -n data200s zip -r model_data.zip task_*.npy models/task_*_dropout 
