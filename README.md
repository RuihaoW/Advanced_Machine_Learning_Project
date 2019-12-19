# Advanced_Machine_Learning_Project

Please read the Proposal.pdf and Report.pdf first. These two files will give a basic introduction of the project. 

To run the demo of CAM with our 4-class dataset

1. Download the dataset from the google drive: https://drive.google.com/open?id=1XPj8Ya02eH3EV4VHRJgu1b-UBaGUamcv
2. "CNN_CAM_NP_4_Class.py" is used to train a model with 4 conv2D layers, 1 GAP layer and 1 Dense layer. You can train the model with the dataset from Step 1. It will save the model and weights in .h5 format.
3. "CNN_CAM_weight_Maps.py" is used to generate all CAMs for all classes. It will save the heat-maps and superposition of maps and original images in .jpg and .npy
4. "Heat_Map_Show.py" is used to show cams those are saved in Step 3.

File "model_weight.h5" is the well-trained model for 4-class dataset. You can directly use it in "CNN_CAM_weight_Maps.py" if you do not want to train your own model in "CNN_CAM_NP_4_Class.py".


Updated on 12/19/2019:
# Part 1: Summary of the original paper
In the paper, the authors provide a mapping approach to highlight the region in the image the prediction relies on. In the CNN model, feature maps are the product of convolutional layers. These feature maps correspond to different features, textures and patterns. The core of Grad-CAM is to use the gradient of the score for the target class, with respect to each feature maps to generate the weight “Wa”, The weight “Wa” represents a partial linearization of the deep network downstream from overall feature maps and captures the ‘importance’ of each feature map for the target class. After the weighted combination based on weight “Wa” and feature maps, the ReLU activation function is used to obtain the heat-map of the desired class. Coordinating with the heat-map, the Guided Backpropagation is used by multiplication to perform the fine-grained importance like pixel-space gradient visualization with a relatively high resolution. 

# Part 2: Reproduced result
We search online to build our own dataset. In this dataset, we have 4 classes images: dog, cat, horse and bird. The numbers of images of each class are: 12500, 12500, 8452, 8671. We use vgg-16 to classify the images in our dataset and then use grad-cam approach to generate the heat-maps. We run our model on a desktop with a Nvidia RTX 2080 with the IDE PyCharm. Our code is inspired by this tutorial(https://github.com/insikk/Grad-CAM-tensorflow). The TensorFlow is used as a machine learning library.  Here are some examples of the reproduction results:
Class: Dog
![Reproduced_Dog](/images/reproduce_dog.png)
Class: Cat
 
Class: Horse
 
Class: Bird
 






















