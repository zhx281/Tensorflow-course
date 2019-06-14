# Tensorflow-course
Practice using Tensorflow 2.0 to reproduce projects and results from the deeplearning.ai course on coursera.org

## CNN Directory:

File in this directory is projects based on the course of [Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow) from coursera.org. 
(Note: The ipython notebooks are reproduction of the projects not direct copies.)
- Week-1: [Dog_vs_Cat_v1.ipynb](https://github.com/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v1.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
	- This ipython notebook demonstrate the skills of how to preprocess image dataset using the keras preprocess library and building a simple Convolutional Neural Network with 3 Conv2D layers with ReLU activations and 3 MaxPooling2D layers, 1 Flatten layer, 1 Fully Connected layer with ReLU activation, and 1 Sigmoid output layer to predict the images to be either a cat or a dog. I also demonstrate understanding of overfitting from analysing plot history of training loss and accuracy, and validation loss and accuracy.

- Week-2: [Dog_vs_Cat_v2.ipynb](https://github.com/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v2.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  	- In this Ipython notebook, I have improve the dataset by using augmentation techniques in preprocessing and loading the data with the ImageDataGenerator method from tensorflow.keras.preprocessing.image library. I also clean up some of the error that cause by the formats or data type of the downloaded datasets. After training for 15 epochs, the model's accuracy and loss in both training and testing stay relativily close and least likely to be overfitting.   

- Week-3: [Horses_vs_Humans.ipynb](https://github.com/zhx281/Tensorflow-course/blob/master/CNN/Horses_vs_Humans_v1.ipynb) or <a href="https://github.com/zhx281/Tensorflow-course/blob/master/CNN/Horses_vs_Humans_v1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  	- In this Ipython notebook, I going to demonstrate how to use a pre-trained model to train on our Horses vs Humans model. First, I am going to load the save model and weights into the notebook and freeze the layers. Second, I am going to strip the final layers from the model, and add my own output layers to create my own model. Third, I train the model with the Horses-v-Humans dataset. Finally, you will be able to test the model with some randomly selected images from the validation directories.