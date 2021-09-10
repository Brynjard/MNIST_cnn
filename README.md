conda env: mnist_cnn
---
### Useful sources: 
https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710 <- detailed description of how backprop works for different layers
https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
https://towardsdatascience.com/training-a-convolutional-neural-network-from-scratch-2235c2a25754 <- modelling of a complete (but simplified cnn) which i used heavily for my model.

### Notes:
- Using relu instead of leaky relu 
- calculating backprop for d_L with respects to input in convolutional layer: not 100% this is correct.