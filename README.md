# Tensorflow-course
Practice using Tensorflow 2.0 to reproduce projects and results from the deeplearning.ai course on coursera.org

## Intro_to_tensorflow Directory:

File in this directory are excerise based on the course of [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow) form coursera.org.
- [Basic Exercise v1](https://github.com/zhx281/Tensorflow-course/blob/master/Intro_to_tensorflow/Basic_Exercise_v1.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/Intro_to_tensorflow/Basic_Exercise_v1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  	- This notebook contain multiply excerise from the week 1 to week 3 of the course. The first excerise is a simple neural netwrok for prediction house price base on rooms the house have. The second excerise is a classification model trying to classify the correct hand written digits from 0 - 9. Finally, the third excerise is an improvement on the second excerise with multiply convolution layers and maxpooling layers.
- [Happy or Sad v1](https://github.com/zhx281/Tensorflow-course/blob/master/Intro_to_tensorflow/Happy_or_Sad_v1.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/Intro_to_tensorflow/Happy_or_Sad_v1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  	- In this notebook, we are going to create a model to predict if the person is happy or sad. It is the 4th exercise in the course, and the goal is to use multiply convolution layers to reaches a 100% accucary on training of 80 images. In this exercise we also experiment with the ImageDataGenerator function from keras.preprocessing.image to help us easily preprocess the images for training.

## CNN Directory:

File in this directory are projects based on the course of [Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow) from coursera.org. 
(Note: The ipython notebooks are reproduction of the projects not direct copies.)
- Week-1: [Dog_vs_Cat_v1.ipynb](https://github.com/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v1.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
	- This ipython notebook demonstrate the skills of how to preprocess image dataset using the keras preprocess library and building a simple Convolutional Neural Network with 3 Conv2D layers with ReLU activations and 3 MaxPooling2D layers, 1 Flatten layer, 1 Fully Connected layer with ReLU activation, and 1 Sigmoid output layer to predict the images to be either a cat or a dog. I also demonstrate understanding of overfitting from analysing plot history of training loss and accuracy, and validation loss and accuracy.

- Week-2: [Dog_vs_Cat_v2.ipynb](https://github.com/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v2.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/CNN/Dog_vs_Cat_v2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  	- In this Ipython notebook, I have improve the dataset by using augmentation techniques in preprocessing and loading the data with the ImageDataGenerator method from tensorflow.keras.preprocessing.image library. I also clean up some of the error that cause by the formats or data type of the downloaded datasets. After training for 15 epochs, the model's accuracy and loss in both training and testing stay relativily close and least likely to be overfitting.   

- Week-3: [Horses_vs_Humans_v1.ipynb](https://github.com/zhx281/Tensorflow-course/blob/master/CNN/Horses_vs_Humans_v1.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/CNN/Horses_vs_Humans_v1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  	- In this Ipython notebook, I am going to demonstrated how to use a pre-trained model to train on our Horses vs Humans model. First, I am going to load the save model and weights into the notebook and freeze the layers. Second, I am going to strip the final layers from the model, and add my own output layers to create my own model. Third, I train the model with the Horses-v-Humans dataset. Finally, you will be able to test the model with some randomly selected images from the validation directories.

- Week-4: [American_sign_language_v1.ipynb](https://github.com/zhx281/Tensorflow-course/blob/master/CNN/American_sign_language_v1.ipynb) or <a href="https://colab.research.google.com/github/zhx281/Tensorflow-course/blob/master/CNN/American_sign_language_v1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  	- In this Ipython notebook, I am going to demonstrated how to use a simple cnn to do multi-classification on the sign-language-mnist dataset from kaggle. I also demonstrated the use of kaggle api within colab, and authenication, searching, and downloading dataset using the kaggle api. For the model, it contains 2 Conv2D layer, 2 MaxPooling2D, and an output layer with softmax activation and size of the total letters of alphabet. The dataset was splitted into 3 different sets, a large set for training, a small set for validation while training, and a tiny set or hidden set for testing after the model was fully trained. The final result was pleasible with an above 90% accuarcy. 
