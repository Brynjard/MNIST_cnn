## A Convolutional Neural Network created from scratch using Numpy for experiments on the MNIST dataset.
---
### Usage: 
- Create conda environment for all package dependencies with conda: 
```
conda env create --name env_name --file=environment.yml
```
- Modify hyper-parameters (learning rate, filter-size, number of filters, epochs etc..) from main.py
- The program logs results to "log.txt" when finished with training/testing. 
- Start training/testing the model: 
```
python3 main.py
```
---
### About: 
This project was a part of the research for my masters thesis, and was originally a competition between me and my thesis-partner to see who could make the best CNN model for MNIST using only Numpy. 
The winner was awarded a 6-pack of beer and eternal glory. 

The point of this project was to learn the finer details of how a CNN works, and the code/model reflects my efforts of the time that the competition ended. It is therefore not perfect, but can hopefully be a source of reference for others learning about CNN's. 
The best achieved results were an accuracy of 87.8%, training on 5000 images, for 20 epochs, 1 5x5 filter and a learning rate of 0.005.
#### Credits: 
Some parts of the code is heavily inspired and at some points copied from other sources, namely this great tutorial from Victor Zhou (https://victorzhou.com): 
https://victorzhou.com/blog/intro-to-cnns-part-1/
https://towardsdatascience.com/training-a-convolutional-neural-network-from-scratch-2235c2a25754

#### Specs: 
This model was tested with several different specifications and architectures, including early stopping to avoid overfitting, ReLU for non-linearity, and different filter-sizes/number of filters. What eventually yielded the best results, were a simple model consisting of: 
- 1 Convolution layer (5x5 filter size, 1 filter)
- 1 maxpooling layer
- 1 Softmax layer
- One prediction layer with cross-entropy loss.




