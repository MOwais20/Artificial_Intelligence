# Udacity_AIProgramming

# AI Programming with Python Project. Image classification (102 flower categories) using Pytorch models.
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, code developed for an image classifier built with PyTorch, then converted into a command line applications: train.py, predict.py.

The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.

In Image Classifier Project.ipynb VGG16 from torchvision.models pretrained models was used. It was loaded as a pre-trained network, based on which defined a new, untrained feed-forward network as a classifier, using ReLU activations and dropout. Trained the classifier layers using backpropagation using the pre-trained network to get the features. The loss and accuracy on the validation set were tracked to determine the best hyperparameters.
