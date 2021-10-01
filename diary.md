#### 30.09 
- Branch dev is most up to date. 
- Current best model is training on 2000 * 3 images, and testing on 2000 images. This model does not seem to converge though, so might try to train on more images. 
- The model works because i deactivated the relu-layers. I tried adjust relu to leaky, but this does not work. Experiment with other activation functions? 
##### TODO: 
- Implement optimizing mechanisms, see articles: 
https://medium.com/@dipti.rohan.pawar/improving-performance-of-convolutional-neural-network-2ecfe0207de7
https://towardsdatascience.com/the-quest-of-higher-accuracy-for-cnn-models-42df5d731faf
- Experiment with hyper-params such as learning rate, filter size, training sample-size, validation sample-size and filter-size. While experimenting, 5 filters seems to be best.
- Experiment with model architecture?!
