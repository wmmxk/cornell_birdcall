1. Linear layer also accept 3d input, the last dimension must match with the in_features argument 
   you specified in the constructor. Linear operation is applied on the last dimension.
https://stackoverflow.com/questions/58587057/multi-dimensional-inputs-in-pytorch-linear-method

2. input to 1d convolution
   [batch_size, input_channel, time_step]
   convolution is done on all the channels over time. In 2d convolution, each filter is appllied 
     on all the input channels over width and height.

3. batch2d, input [N, C, H, w]
   Normalization is done per channel. So the channel dimension is needed: https://stackoverflow.com/questions/62041724/batchnorm2d-pytorch-why-pass-number-of-channels-to-batchnorm
