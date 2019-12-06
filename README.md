# Advanced_Machine_Learning_Project

Please read the Proposal.pdf and Report.pdf first. These two files will give a basic introduction of the project. 

To run the demo of CAM with our 4-class dataset

1. Download the dataset from the google drive: https://drive.google.com/open?id=14c0y9KhCKGC5DdeMGKZFVeITlNGVxOc7
2. "CNN_CAM_NP_4_Class.py" is used to train a model with 4 conv2D layers, 1 GAP layer and 1 Dense layer. You can train the model with the dataset from Step 1. It will save the model and weights in .h5 format.
3. "CNN_CAM_weight_Maps.py" is used to generate all CAMs for all classes. It will save the heat-maps and superposition of maps and original images in .jpg and .npy
4. "Heat_Map_Show.py" is used to show cams those are saved in Step 3.
