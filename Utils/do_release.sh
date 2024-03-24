mkdir release
cd release
git clone https://github.com/fractalclockwork/Data200.git
rm -rf  Data200/.git/
conda run -n data200s zip -r data200_grad_proj.zip Data200/
cd ../
cp release/data200_grad_proj.zip ./
rm -rf release
