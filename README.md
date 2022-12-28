# Anomaly-Detection-using-Surveillance-Footage
### 
Dataset:
https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0
###
For Large files like pretrained weights:
https://drive.google.com/drive/folders/1dsiSA7TPDUAqGwBm5YUI5-WyOeaSB7dj?usp=sharing
###
Follow ipynb anomaly detection file, setup a lower version of python like 3.6 for tensorflow scipy libraries
then you'll get c3d feature raw files then after that too short files will be eliminated and we'll get the uploaded preprocessed c3d normal and anomaly files respectively.
We have obtained c3d model feature extraction directly in fc6 layer and then built our own deep neural network with MIL(Multiple instance learning) parameters.
###
We followed this paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf
for implementation.

.npy files are numpy binary files the array is saved in binary format, i've attached a ipynb file to view the npy files or C3D features obtained.

