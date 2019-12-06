# Pedestrian-Detector
PIPELINE OR THE FLOW OF THE PROJECT:

1) Procure a dataset, either download a public dataset or make a datatset of your own.
   	Make sure the dataset contains images of both Pedestrians and non-Pedestrians for training.

2) Divide the dataset into trainig, cross-validation (if you have the number) and test, either by slicing the list or
	using test-train split from sci-kit learn.

3) Convert the images to arrays ad then tensors using the appropriate functions, methods.

4) Create a Convolution Neural Network Model for the dataset using Keras/Pytorch or any other suitable API for the training and testing.

5) Train the model and save the weights.

6) Select an image(test-image) on which the Pedestrian is to be detected and run it through the Convolution Model.

7) The image will be scaled down in accordance with the depth and the kernel size of the CNN, and the resulting output will be tensor with each value of it being a probability. Make a copy of resulting tensor and apply threshold on the tensor(rudimentary Non-Max Suppression),i.e. all the tensor entries below the threshold are reassgned value 0 and those above the threshold are converted to 1. 

8) Generate a heatmap from the resulting tensor ( by reconverting array into image)

9) Generate a heatmap with the threshold applied.

10) Use PIL to draw rectangles and rescale according to the dimensions of the original image.

11) Use cv2 library to group rectangles.

12) Display the image. 

ARCHITECTURE OF THE MODEL(summary of the same model using Keras API):

Layer (type)                      Output Shape                   Param # 
=================================================================
conv2d_1 (Conv2D)                 (None, 48, 96, 16)             160 
_________________________________________________________________
conv2d_2 (Conv2D)                 (None, 48, 96, 32)             4640 
_________________________________________________________________
max_pooling2d_1(MaxPooling2)      (None, 8, 16, 32)              0 
_________________________________________________________________
dropout_1 (Dropout)               (None, 8, 16, 32)              0 
_________________________________________________________________
conv2d_3 (Conv2D)                 (None, 4, 8, 64)               92224 
_________________________________________________________________
max_pooling2d_2(MaxPooling2)      (None, 2, 4, 64)               0 
_________________________________________________________________
conv2d_4 (Conv2D)                 (None, 1, 1, 64)               32832 
_________________________________________________________________
dropout_2 (Dropout)               (None, 1, 1, 64)               0 
_________________________________________________________________
conv2d_5 (Conv2D)                 (None, 1, 1, 128)              8320 
_________________________________________________________________
conv2d_6 (Conv2D)                 (None, 1, 1, 1)                129 
        
=================================================================
Total params: 138,305

Trainable params: 138,305

Non-trainable params: 0

