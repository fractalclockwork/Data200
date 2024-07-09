conda run --no-capture-output -n data200s gdown https://drive.google.com/uc?id=19SJmHIrGtEVVlsWaiXmjt_7-uh6wBkdq -O 'data/model_data.zip'
cd data; conda run --no-capture-output -n data200s unzip 'model_data.zip'
