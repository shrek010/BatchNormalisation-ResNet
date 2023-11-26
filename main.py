class EnhancedConvolutionalProcessingBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(EnhancedConvolutionalProcessingBlock, self).__init__()
        # Initialize attributes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        # Build the module
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)

        # First convolutional layer with batch normalization and Leaky ReLU
        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=x.shape[1], out_channels=self.num_filters,
                                              kernel_size=self.kernel_size, padding=self.padding, 
                                              bias=self.bias, dilation=self.dilation, stride=1)
        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=self.num_filters)

        # Second convolutional layer with batch normalization
        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters,
                                              kernel_size=self.kernel_size, padding=self.padding, 
                                              bias=self.bias, dilation=self.dilation, stride=1)
        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=self.num_filters)

    def forward(self, x):
        identity = x

        # Applying the first conv layer, BN, and Leaky ReLU
        out = F.leaky_relu(self.layer_dict['bn_0'](self.layer_dict['conv_0'](x)))

        # Applying the second conv layer and BN
        out = self.layer_dict['bn_1'](self.layer_dict['conv_1'](out))

        # Adding the residual connection
        out += identity

        # Final Leaky ReLU
        out = F.leaky_relu(out)

        return out
