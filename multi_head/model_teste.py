""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- CNN model
- CBAM Reference: https://arxiv.org/abs/1807.06521
    - Code Reference: https://tinyurl.com/25wyxnb8
    - Code Resnet CBAM: https://tinyurl.com/2b8zkaol

"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for capturing channel-wise dependencies.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the channel attention block. Default is 16.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        init.kaiming_normal_(self.fc1.weight,nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight,nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through the Channel Attention Module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying channel attention.
        """
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        channel_attention = torch.sigmoid(avg_out + max_out)

        return channel_attention

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for capturing spatial dependencies.

    Args:
        kernel_size (int, optional): Size of the convolutional kernel for spatial attention. Default is 7.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        init.kaiming_normal_(self.conv.weight,nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through the Spatial Attention Module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying spatial attention.
        """
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled_tensor = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention = self.sigmoid(self.conv(pooled_tensor))

        return spatial_attention

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) combining Channel Attention Module (CAM)
    and Spatial Attention Module (SAM).

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the channel attention block. Default is 16.
        kernel_size (int, optional): Size of the convolutional kernel for spatial attention. Default is 7.
    """

    def __init__(self, in_channels=1, reduction_ratio=16, kernel=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel)

    def forward(self, x):
        """
        Forward pass through the CBAM block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying CBAM.
        """
        residual = x
        out = self.channel_attention(x) * x     # Channel attention output
        out = self.spatial_attention(out) * out # Spatial attention - cbam output
        return out + residual

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the SE block. Default is 16.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initializes the Squeeze-and-Excitation block.

        Args:
            in_channels (int): Number of input channels.
            reduction_ratio (int, optional): Reduction ratio for the SE block. Default is 16.
        """
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        init.kaiming_normal_(self.fc1.weight,nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight,nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through the Squeeze-and-Excitation block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the SE block.
        """
        x_se = self.pool(x)
        x_se = F.relu(self.fc1(x_se))
        x_se = torch.sigmoid(self.fc2(x_se))
        return x * x_se

class ConvBlock(nn.Module):
    """ Convolutional Block with Conv -> BatchNorm -> ReLU """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, channels_last=False):
        """ Initialize ConvBlock.

        Args:
            in_channel : int
                Represents the number of channels in the input image (default 3 for RGB)
            kernel : int
                Represents the size of the convolutional window (3 means [3,3])
            filters : int
                Number of filters
            strides : int
                Represents the stride of the convolutional window (3 means [3,3])
            mu : float
                Mean for the batch normalization
            epsilon : float
                Epsilon for the batch normalization
        """
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        self.activation = nn.ReLU()
        self.channels_last = channels_last
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.filters, 
                            kernel_size=self.kernel_size, 
                            stride=self.strides, 
                            padding= self.padding)
        init.kaiming_normal_(self.conv.weight,nonlinearity='relu')
        self.batch_norm = nn.BatchNorm2d(num_features=self.filters)
            
    def forward(self, inputs):
        """ Convolutional block with convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        #print(f'ConvBlock input.shape: {inputs.shape}')
        tensor = self.conv(inputs)
        #print(f'layer1 output.shape: {tensor.shape}')
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor

class Conv1DBlock(nn.Module):
    """1D Conv Block com opção de padding 'same' igual ao Keras"""

    def __init__(self, kernel=3, in_channels=1, filters=16, strides=1, padding_mode='same'):
        """
        Args:
            kernel (int): tamanho do filtro convolucional
            in_channels (int): canais de entrada
            filters (int): canais de saída (filtros)
            strides (int): stride da convolução
            padding_mode (str): 'same' ou 'valid'. Se 'same', faz padding manual
        """
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.padding_mode = padding_mode.lower()
        
        # Sempre padding=0 na camada Conv1d, o padding será feito manualmente se necessário
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=0
        )
        init.kaiming_normal_(self.conv.weight, nonlinearity='relu')

        self.batch_norm = nn.BatchNorm1d(num_features=self.filters, eps=1e-3, momentum=0.99)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor (batch_size, in_channels, seq_len)
        
        Returns:
            Tensor (batch_size, filters, seq_len') com padding 'same' ou 'valid'
        """
        if self.padding_mode == 'same':
            # Calcula o padding total necessário para manter o tamanho de seq_len
            # Formula do padding total:
            # padding_total = (kernel_size - 1) * dilation (dilation=1 aqui)
            padding_total = self.kernel_size - 1
            # Padding à esquerda e direita (assimetricos para kernel par)
            pad_left = padding_total // 2
            pad_right = padding_total - pad_left
            # Aplica padding manual com F.pad
            x = F.pad(inputs, (pad_left, pad_right))
        else:
            # padding_mode == 'valid', sem padding
            x = inputs

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class DefConvBlock(nn.Module):
    """ Deformable Convolutional Block with DeformConv -> BatchNorm -> ReLU """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, channels_last=False):
        """ Initialize DeformableConvBlock.

        Args:
            in_channel : int
                Represents the number of channels in the input image (default 3 for RGB)
            kernel : int
                Represents the size of the convolutional window (3 means [3,3])
            filters : int
                Number of filters
            strides : int
                Represents the stride of the convolutional window (3 means [3,3])
            mu : float
                Mean for the batch normalization
            epsilon : float
                Epsilon for the batch normalization
        """
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        self.activation = nn.ReLU()
        self.channels_last = channels_last
        
        # Deformable convolution
        self.offsets = nn.Conv2d(in_channels=in_channels, out_channels=2*kernel*kernel, 
                                 kernel_size=kernel, stride=strides, padding=self.padding)
        self.deform_conv = DeformConv2d(in_channels=in_channels, out_channels=self.filters, 
                                        kernel_size=self.kernel_size, stride=self.strides, 
                                        padding=self.padding)
        init.kaiming_normal_(self.deform_conv.weight, nonlinearity='relu')
        
        self.batch_norm = nn.BatchNorm2d(num_features=self.filters)
            
    def forward(self, inputs):
        """ Deformable convolutional block with deformable convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        # Calculate offsets
        offsets = self.offsets(inputs)
        
        # Apply deformable convolution
        tensor = self.deform_conv(inputs, offsets)
        
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor

class SEConvBlock(nn.Module):
    """
    Squeeze-and-Excitation Convolution Block.
    """
    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, channels_last=False, reduction_ratio=16):
        """
        Initializes the Squeeze-and-Excitation Convolution block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
            stride (int, optional): Stride for the convolution. Default is 1.
            padding (int, optional): Padding for the convolution. Default is 1.
            reduction_ratio (int, optional): Reduction ratio for the SE block. Default is 16.
        """
        super(SEConvBlock, self).__init__()
        self.conv_block = ConvBlock(kernel, in_channels, filters, strides, channels_last)
        self.se_block = SEBlock(filters, reduction_ratio)

    def forward(self, inputs):
        """
        Forward pass through the Squeeze-and-Excitation Convolution block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Squeeze-and-Excitation Convolution block.
        """
        conv_output = self.conv_block(inputs)
        se_output = self.se_block(conv_output)
        return se_output

class CBAMConvBlock(nn.Module):
    """
    Convolutional Block with Convolutional Block Attention Module (CBAM).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int, optional): Size of the convolutional kernel. Default is 3.
        stride (int, optional): Stride of the convolutional operation. Default is 1.
        padding (int, optional): Padding for the convolutional operation. Default is 1.
        reduction_ratio (int, optional): Reduction ratio for the channel attention block. Default is 16.
    """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, channels_last=False, reduction_ratio=16):
        super(CBAMConvBlock, self).__init__()

        self.conv_block = ConvBlock(kernel, in_channels, filters, strides, channels_last)
        self.cbam_block = CBAMBlock(filters, reduction_ratio)

    def forward(self, x):
        """
        Forward pass through the CBAM Convolutional Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying CBAM Convolutional Block.
        """
        # check if the channel is equal to 3 
        
        ##print(f'CBAMConvBlock input.shape: {x.shape}')
        x = self.conv_block(x)
        ##print(f'layer1 output.shape: {x.shape}')
        x = self.cbam_block(x)
        ##print(f'layer2 output.shape: {x.shape}')
        return x

class DWConvBlock(nn.Module):
    """ Depth Wise Separable Convolutional Block with Conv -> DepthwiseConv -> BatchNorm -> ReLU """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, channels_last=False):
        """ Initialize DepthwiseSeparableConvBlock.

        Args:
            in_channel : int
                Represents the number of channels in the input image (default 3 for RGB)
            kernel : int
                Represents the size of the convolutional window (3 means [3,3])
            filters : int
                Number of filters
            strides : int
                Represents the stride of the convolutional window (3 means [3,3])
            mu : float
                Mean for the batch normalization
            epsilon : float
                Epsilon for the batch normalization
        """
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        self.activation = nn.ReLU()
        self.channels_last = channels_last

        # Depthwise Separable Convolution: Depthwise Convolution + Pointwise Convolution
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.kernel_size,
                                        stride=self.strides, padding=self.padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,out_channels= self.filters, kernel_size=1)

        init.kaiming_normal_(self.depthwise_conv.weight,nonlinearity='relu')
        init.kaiming_normal_(self.pointwise_conv.weight,nonlinearity='relu')

        self.batch_norm = nn.BatchNorm2d(num_features=self.filters)
            
    def forward(self, inputs):
        """ Depthwise Separable Convolutional block with depthwise convolution + pointwise convolution
            + batch normalization + ReLU activation.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        tensor = self.depthwise_conv(inputs)
        tensor = self.pointwise_conv(tensor)
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor

class MBConv(nn.Module):
    """
    MobileNetV3 Bottleneck Block with Squeeze-and-Excitation (SE) Block.
    """
    def __init__(self,kernel=1, in_channels=1, filters=1, strides=1, expand_ratio=6, reduction_ratio=16):
        super(MBConv, self).__init__()
        mid_channels = in_channels * expand_ratio

        # Expand
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=strides, padding=(kernel - 1) // 2, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        # Squeeze-and-Excitation
        self.se_block = SEBlock(mid_channels, reduction_ratio)

        # Project - Pointwise
        self.project_conv = nn.Conv2d(mid_channels, filters, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(filters)
        
        self.use_residual = (in_channels == filters and strides == 1)

    def forward(self, x):
        identity = x

        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.expand_relu(out)

        # Depthwise
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.depthwise_relu(out)

        # Squeeze-and-Excitation
        out = self.se_block(out) # skip is inside the block

        # Project - project the features back to the original number of channels
        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_residual:
            out = out + identity

        return out
    
class MBConv_V2(nn.Module):
    """
    MobileNetV2 Bottleneck Block
    Ref 1: Effective Data Augmentation and Training Techniques for Improving Deep Learning in Plant Leaf Disease Recognition
    link: https://tinyurl.com/2axqr4cl
    """
    def __init__(self,kernel=1, in_channels=1, filters=1, strides=1, expand_ratio=6, reduction_ratio=16):
        super(MBConv_V2, self).__init__()
        mid_channels = in_channels * expand_ratio

        # Expand
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=strides, padding=(kernel - 1) // 2, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        # Project
        self.project_conv = nn.Conv2d(mid_channels, filters, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(filters)
        
        self.use_residual = (in_channels == filters and strides == 1)

    def forward(self, x):
        identity = x

        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.expand_relu(out)

        # Depthwise
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.depthwise_relu(out)

        # Project
        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_residual:
            out = out + identity

        return out
    
class MBConv_EPPGA(nn.Module):
    """
    EPPGA block structure: MobileNetV3 style Bottleneck Block with Squeeze-and-Excitation (SE) Block.
    Ref: An evolutionary neural architecture search method based on performance prediction and weight inheritance
    link: https://www.sciencedirect.com/science/article/pii/S0020025524003797
    """
    def __init__(self,kernel=1, in_channels=1, filters=1, strides=1, expand_ratio=6, reduction_ratio=16):
        super(MBConv_EPPGA, self).__init__()
        mid_channels = in_channels * expand_ratio

        # Expand
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=strides, padding=(kernel - 1) // 2, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise_relu = nn.ReLU6(inplace=True)
        
        # Pointwise - Project
        self.pointwise_conv = nn.Conv2d(mid_channels, filters, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(filters)

        # Squeeze-and-Excitation
        self.se_block = SEBlock(filters, reduction_ratio)
        
        self.use_residual = (in_channels == filters and strides == 1)

    def forward(self, x):
        identity = x

        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.expand_relu(out)

        # Depthwise
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.depthwise_relu(out)
        
        # Pointwise - to project the features back to the original number of channels
        out = self.pointwise_conv(out)
        out = self.pointwise_bn(out)

        # Squeeze-and-Excitation
        out = self.se_block(out) # skip is inside the block

        if self.use_residual:
            out = out + identity

        return out
    
class ResidualV1(nn.Module):
    def __init__(self, in_channels=1, kernel=1, filters=1, strides=1, channels_last=False):
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.channels_last = channels_last
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                                kernel_size=self.kernel_size,stride=strides, 
                                padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                                kernel_size=self.kernel_size, stride=1, 
                                padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, stride=strides, bias=False),
            nn.BatchNorm2d(num_features=self.filters)
        ) if strides != 1 or in_channels != filters else nn.Identity()
        
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if not isinstance(self.projection, nn.Identity):
            init.kaiming_normal_(self.projection[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs):
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        #print(f'inputs.shape V1: {inputs.shape}')            
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
        #print(f'tensor.shape Layer 1: {tensor.shape}')
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
              
        #print(f'tensor.shape Layer 2: {tensor.shape}')
              
        tensor += self.projection(inputs)
        tensor = F.relu(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
        
        #print(f'output.shape: {tensor.shape}')
        return tensor

class ResidualV1CBAM(nn.Module):
    """ Residual Block with Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> CBAM -> Add -> ReLU """
    def __init__(self, in_channels=1, kernel=1, filters=1, strides=1, channels_last=False):
        """ Initialize ResidualV1.

        Args:
            in_channel : int
                Represents the number of channels in the input image (default 3 for RGB)
            kernel : int
                Represents the size of the convolutional window (3 means [3,3])
            filters : int
                Number of filters
            strides : int
                Represents the stride of the convolutional window (3 means [3,3])
            mu : float
                Mean for the batch normalization
            epsilon : float
                Epsilon for the batch normalization
        """
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.channels_last = channels_last
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                                kernel_size=self.kernel_size,stride=strides, 
                                padding= self.padding ,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                                kernel_size=self.kernel_size,stride=strides, 
                                padding= self.padding ,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        
        init.kaiming_normal_(self.conv1.weight,nonlinearity='relu')  # He Normal initialization
        init.kaiming_normal_(self.conv2.weight,nonlinearity='relu')  # He Normal initialization
        
        self.channel_attention = ChannelAttention(filters)
        self.spatial_attention = SpatialAttention()
                # Shortcut connection
        if strides != 1 or in_channels != filters:
            #print("Shortcut connection")
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, padding=0,stride=strides, bias=False),
                nn.BatchNorm2d(num_features=self.filters)
            )
        else:
            self.shortcut = nn.Identity()
        

    def forward(self, inputs):
        """ Residual block with convolution op + batch normalization op + add op.

        Args:
            inputs: input tensor to the
        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        
        # Apply CBAM
        tensor = self.channel_attention(tensor) * tensor
        tensor = self.spatial_attention(tensor) * tensor

        
        tensor = tensor + self.shortcut(inputs)
        tensor = F.relu(tensor)
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor
    
class ResidualV1Pr(nn.Module):
    """ Residual V1 block with projection shortcut """
    def __init__(self, in_channels=1, kernel=1, filters=1, strides=1, channels_last=False):
        """ Initialize ResidualV1.

        Args:
            in_channels : int
                Represents the number of channels in the input image (default 3 for RGB)
            kernel : int
                Represents the size of the convolutional window (3 means [3,3])
            filters : int
                Number of filters
            strides : int
                Represents the stride of the convolutional window (3 means [3,3])
            mu : float
                Mean for the batch normalization
            epsilon : float
                Epsilon for the batch normalization
        """
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.channels_last = channels_last
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                                kernel_size=self.kernel_size,stride=strides, 
                                padding= self.padding ,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                                kernel_size=self.kernel_size,stride=strides, 
                                padding= self.padding ,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        
        # Shortcut connection
        if strides != 1 or in_channels != filters:
            #print("Shortcut connection")
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, padding=0,stride=strides, bias=False),
                nn.BatchNorm2d(num_features=self.filters)
            )
        else:
            self.shortcut = nn.Identity()


    def forward(self, inputs):
        """ Residual block with convolution op + batch normalization op + add op.

        Args:
            inputs: input tensor to the
        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        #print(f'inputs.shape: {inputs.shape}')            
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
        #print(f'tensor.shape Layer 1: {tensor.shape}')
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        #print(f'tensor.shape Layer 2: {tensor.shape}')
        tensor = tensor + self.shortcut(inputs)
        tensor = F.relu(tensor)
        #print(f'output.shape: {tensor.shape}')
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor
        
class MaxPooling(nn.Module):
    """ Max Pooling layer """

    def __init__(self, kernel=1, strides=1, channels_last=False):
        """ Initialize MaxPooling.

        Args:
            kernel : int
                Represents the size of the pooling window (3 means [3,3])
            strides : int
                Represents the stride of the pooling window (3 means [3,3])
        """
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = 0 # 'valid' no padding
        self.channels_last = channels_last

        self.max_pool = nn.MaxPool2d(kernel_size=self.kernel, 
                                    stride=self.strides, 
                                    padding=self.padding)

    def forward(self, inputs):
        """ Max Pooling layer.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        # check of the image size    
        if inputs.shape[2] >= self.kernel and inputs.shape[3] >= self.kernel:
            tensor = self.max_pool(inputs)
        else:
            ##print("Warning: MaxPooling layer not applied because the image size is smaller than the kernel size")
            return inputs
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor

class MaxPooling1D(nn.Module):
    """1D Max Pooling layer"""

    def __init__(self, kernel=2, strides=2):
        """
        Args:
            kernel : int
                Size of the pooling window
            strides : int
                Stride of the pooling operation
        """
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = 0  # default 'valid' pooling

        self.max_pool = nn.MaxPool1d(
            kernel_size=self.kernel,
            stride=self.strides,
            padding=self.padding
        )

    def forward(self, inputs):
        """
        Forward pass for MaxPooling1D.

        Args:
            inputs: Tensor of shape (batch_size, channels, seq_len)

        Returns:
            Tensor after max pooling.
        """
        if inputs.shape[-1] >= self.kernel:
            return self.max_pool(inputs)
        else:
            # Optional: print or log warning
            return inputs

class AvgPooling(nn.Module):
    """ Average Pooling layer """

    def __init__(self, kernel=1, strides=1, channels_last=False):
        """ Initialize AvgPooling.

        Args:
            kernel : int
                Represents the size of the pooling window (3 means [3,3])
            strides : int
                Represents the stride of the pooling window (3 means [3,3])
        """
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = 0 # 'valid' no padding
        self.channels_last = channels_last

        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel, 
                                    stride=self.strides, 
                                    padding=self.padding)

    def forward(self, inputs):
        """ Average Pooling layer.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        ##print(f'inputs.shape avg: {inputs.shape}')
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        # check of the image size    
        if inputs.shape[2] >= self.kernel and inputs.shape[3] >= self.kernel:
            tensor = self.avg_pool(inputs)
        else:
            #print("Warning: AvgPooling layer not applied because the image size is smaller than the kernel size")
            return inputs
            #return inputs.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor

class AvgPooling1D(nn.Module):
    """1D Average Pooling layer"""

    def __init__(self, kernel=2, strides=2):
        """
        Args:
            kernel : int
                Size of the pooling window
            strides : int
                Stride of the pooling operation
        """
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = 0  # 'valid' padding (no padding)

        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel,
            stride=self.strides,
            padding=self.padding
        )

    def forward(self, inputs):
        """
        Applies average pooling operation.

        Args:
            inputs: Tensor in the format (batch_size, channels, seq_len)

        Returns:
            Tensor after average pooling.
        """
        if inputs.shape[-1] >= self.kernel:
            return self.avg_pool(inputs)
        else:
            return inputs

class StochasticPooling(nn.Module):
    """ Stochastic Pooling layer """

    def __init__(self, kernel=2, strides=2, channels_last=False):
        """
        Args:
            kernel : int
                Tamaño de la ventana de pooling (p.ej., 2 implica [2,2])
            strides : int
                Stride de la ventana de pooling
            channels_last : bool
                Indica si el tensor de entrada está en formato NHWC
        """
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.channels_last = channels_last
        self.padding = 0  # 'valid' sin padding

    def forward(self, inputs):
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2)  # Convertir de NHWC a NCHW

        # Verificar tamaño de la imagen
        if inputs.shape[2] < self.kernel or inputs.shape[3] < self.kernel:
            return inputs

        # Extraer ventanas con unfold
        patches = F.unfold(inputs, kernel_size=self.kernel, stride=self.strides, padding=self.padding)
        batch, flat_size, L = patches.shape
        channels = inputs.shape[1]
        patches = patches.view(batch, channels, self.kernel * self.kernel, L)

        # Calcular probabilidades normalizadas para cada parche
        probs = F.softmax(patches, dim=2)  # [B, C, K*K, L]

        # Reorganizar para muestreo: colapsar dimensiones batch, canal y patch
        probs_reshaped = probs.permute(0, 1, 3, 2).reshape(-1, self.kernel * self.kernel)
        patches_reshaped = patches.permute(0, 1, 3, 2).reshape(-1, self.kernel * self.kernel)

        # Muestrear índices según las probabilidades
        indices = torch.multinomial(probs_reshaped, num_samples=1).squeeze(-1)
        pooled = patches_reshaped.gather(1, indices.unsqueeze(1)).view(batch, channels, L)

        # Reconstruir forma espacial
        out_H = (inputs.shape[2] - self.kernel) // self.strides + 1
        out_W = (inputs.shape[3] - self.kernel) // self.strides + 1
        tensor = pooled.view(batch, channels, out_H, out_W)

        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1)  # Convertir de NCHW a NHWC

        return tensor

class FullyConnected(nn.Module):
    def __init__(self,input_features=1, units=1):
        """ Initialize FullyConnected.

        Args:
            inputs_features : int
                Represents the number of inputs features of the layer
            units : int
                Represents the number of neurons in the layer

        """
        super().__init__()
        self.inputs__features = input_features
        self.units = units                
        self.fc = nn.Linear(in_features=self.inputs__features,
                            out_features=self.units)
        init.kaiming_normal_(self.fc.weight,nonlinearity='relu')                   
        
    def forward(self, inputs):
        """ FullyConnected layer.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        tensor = self.fc(inputs)
        return tensor
    
# ==============================
# New Fixed Blocks Definitions
# ==============================

class StemBlock(nn.Module):
    """
    Fixed Stem block: a 3x3 convolution with downsampling, followed by BatchNorm and ReLU.
    This block is intended to extract low-level features and reduce spatial dimensions.
    """
    def __init__(self, in_channels=3, filters=32, stride=2):
        super(StemBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, filters, kernel_size=3, stride=stride, padding=1)
        init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TailBlock(nn.Module):
    """
    Fixed Tail block: a 3x3 convolution that aggregates extracted features.
    Optionally applies Global Average Pooling (GAP) to reduce each feature map to a single value.
    """
    def __init__(self, in_channels, filters=None, use_gap=True):
        super(TailBlock, self).__init__()
        # If no specific number of filters is provided, maintain the same channel size.
        if filters is None:
            filters = in_channels
        self.conv = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        self.use_gap = use_gap
        if self.use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_gap:
            x = self.gap(x)
        return x
    
class NoOp(nn.Module):
    """ NoOp layer.
    """
    pass

functions_dict = {
    'StemBlock': StemBlock,
    'TailBlock': TailBlock,
    'ConvBlock': ConvBlock,
    'DWConvBlock': DWConvBlock,
    'SEConvBlock': SEConvBlock,
    'MBConv': MBConv,
    'MBConvV2': MBConv_V2,
    'MBConv_EPPGA': MBConv_EPPGA,
    'ResidualV1': ResidualV1,
    'ResidualV1Pr': ResidualV1Pr,
    'CBAMConvBlock': CBAMConvBlock,
    'ResidualV1CBAM': ResidualV1CBAM,
    'CBAMBlock' : CBAMBlock,
    'MaxPooling': MaxPooling,
    'AvgPooling': AvgPooling,
    'Conv1DBlock' : Conv1DBlock,
    'MaxPooling1D': MaxPooling1D,
    'AvgPooling1D': AvgPooling1D,
    'StochasticPooling': StochasticPooling,
    'FullyConnected': FullyConnected,
    'no_op': NoOp}


class LSTMRegressor(nn.Module):
    def __init__(self,
                 input_features: int,
                 units: int,
                 seq_splits: int = 4,
                 output_dim: int = 1):
        """
        LSTM-based regression layer.

        Args:
            input_features : int
                Flattened feature size (e.g. 64).
            units : int
                Hidden size of the LSTM.
            seq_splits : int
                How many timesteps to split the flattened vector into.
            output_dim : int
                Size of the final output (number of outputs).
        """
        super().__init__()
        assert input_features % seq_splits == 0, "input_features must be divisible by seq_splits"
        self.seq_len = seq_splits
        self.feature_dim = input_features // seq_splits
        self.units = units
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=units, batch_first=True)
        self.regressor = nn.Linear(units, output_dim)

    def forward(self,
                inputs):
        """
        Forward pass of the LSTM regressor.

        Args:
            inputs: Tensor of shape [batch_size, input_features]

        Returns:
            output: Tensor of shape [batch_size, output_dim]
        """
        batch_size, total_features = inputs.shape
        assert total_features == self.seq_len * self.feature_dim, "Input shape mismatch"

        x = inputs.view(batch_size, self.seq_len, self.feature_dim)
        lstm_out, _ = self.lstm(x)  # Shape: [batch_size, seq_len, units]
        output = self.regressor(lstm_out[:, -1, :])  # Use last timestep
        return output


class TwoLayerLSTMRegressor(nn.Module):
    def __init__(self,
                 input_features: int,
                 seq_splits: int = 4,
                 hidden_sizes: list = [128, 64],
                 output_dim: int = 1,
                 dropout: float = 0.5):
        """
        LSTM-based regression layer with two stacked LSTM layers and dropout.

        Args:
            input_features (int): 
                Total number of input features after flattening (e.g., from a feature map).
            seq_splits (int): 
                Number of timesteps to split the input into. The input will be reshaped
                to shape [batch_size, seq_splits, input_features // seq_splits].
                Must divide input_features evenly.
            hidden_sizes (list): 
                List with two integers specifying the hidden sizes of the first and
                second LSTM layers, respectively.
            output_dim (int): 
                Dimension of the final output (e.g., 1 for univariate regression,
                or more for multivariate).
            dropout (float): 
                Dropout probability applied between the two LSTM layers.
        Raises:
            AssertionError: If input_features is not divisible by seq_splits.
        """
        super().__init__()
        assert input_features % seq_splits == 0, "input_features must be divisible by seq_splits"
        self.seq_len = seq_splits
        self.feature_dim = input_features // seq_splits
        self.output_dim = output_dim

        self.lstm1 = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_sizes[0],
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[1],
            batch_first=True
        )

        self.regressor = nn.Linear(hidden_sizes[1], output_dim)

    def forward(self, inputs):
        batch_size, total_features = inputs.shape
        x = inputs.view(batch_size, self.seq_len, self.feature_dim)

        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        output = self.regressor(x[:, -1, :])
        return output


class NetworkGraph(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=1,
                 network_gap=False,
                 network_config = 'default',
                 num_sensors=None):
        """ Initialize NetworkGraph.

        Args:
            num_classes: int 
                number of classes for classification model.
            in_channels: int
                number of input channels.
            network_gap: bool
                flag to apply Global Average Pooling (GAP) in the Tail block.
            network_config: str
                network configuration to use for the model.
        Returns:
            output logits tensor.
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_gap = network_gap
        self.network_config = network_config
        self.layers = None
        self.fc = None

        
    def create_functions(self, net_list, fn_dict, cbam=False):
        """ Dynamically create network layers.

        Args:
            net_list: List of layer names.
            fn_dict: dict with definitions of the functions (name and parameters);
                format --> {'fn_name': ['FNClass', {'param1': value1, 'param2': value2}]}.
            cbam: Boolean flag to modify network for CBAM.
        """
        # Define sets for blocks that need special handling.
        primary_blocks = {
            'ConvBlock', 'Conv1DBlock', 'DWConvBlock', 'SEConvBlock', 'ResidualV1CBAM',
            'MBConv', 'MBConvV2', 'MBConv_EPPGA', 'ResidualV1', 'ResidualV1Pr'
        }
        if self.network_config == 'default':
            in_channels = self.in_channels
            self.layers = []
            # Optionally insert a 1x1 convolution for CBAM in default config.
            if cbam:
                net_list.insert(0, 'conv_1_1_32')
                conv_1_1_info = {'conv_1_1_32': {'function': 'ConvBlock', 'params': {'kernel': 1, 'strides': 1, 'filters': 32}}}
                fn_dict.update(conv_1_1_info)


            for name in net_list:
                parameters = fn_dict[name]
                func = parameters['function']  # Cache the function name
                if func == 'NoOp':
                    continue

                if func in primary_blocks:
                    parameters['params']['in_channels'] = in_channels
                    in_channels = parameters['params']['filters']
                elif func == 'CBAMBlock':
                    parameters['params']['in_channels'] = in_channels

                self.layers.append(functions_dict[func](**parameters['params']))
                
            self.model = nn.Sequential(*self.layers)
            self.fc = None
        elif self.network_config == 'dense':
            # Dense connectivity: use a ModuleList and build cumulative channels.
            cumulative_channels = self.in_channels  # e.g., initial channels (e.g., 3)
            self.layers = []  # List to store layers

            # Insert Stem block if desired.
            stem_params = {
                'in_channels': cumulative_channels,
                'filters': 32,
                'stride': 1,
            }
            self.layers.append(StemBlock(**stem_params))
            cumulative_channels += stem_params['filters']  # Add stem's output channels

            # For each layer, update parameters based on whether it is a pooling operation.
            for name in net_list:
                parameters = fn_dict[name]
                #print(f'Function name: {parameters["function"]}')
                if parameters['function'] == 'NoOp':
                    continue

                # Make a copy of the parameters to avoid modifying the original dictionary.
                params = parameters['params'].copy()

                # If the function name indicates a pooling operation, remove "in_channels"
                if "pool" in parameters['function'].lower():
                    # Remove in_channels if it exists (pooling layers don't expect it)
                    params.pop('in_channels', None)
                    # Do not update cumulative_channels (pooling doesn't change channel count)
                else:
                    # For non-pooling blocks, override in_channels with the current cumulative count.
                    params['in_channels'] = cumulative_channels
                    # If the block defines an output channel count, update cumulative_channels.
                    if 'filters' in params:
                        cumulative_channels += params['filters']

                #print(f'Instantiating {parameters["function"]} with params: {params}')
                self.layers.append(functions_dict[parameters['function']](**params))

            # Optionally add a Tail block.
            tail_params = {
                'in_channels': cumulative_channels,
                'filters': cumulative_channels,
                'use_gap': self.use_gap
            }
            self.layers.append(TailBlock(**tail_params))

            # Wrap layers in a ModuleList for custom forward pass.
            self.layers = nn.ModuleList(self.layers)
            self.model = None  # Not using a sequential model.
            self.fc = None     # Fully connected layer will be initialized later.
        else:
            raise ValueError(f"Invalid network configuration: {self.network_config}")
    def forward(self, inputs, debug=False):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor.
            debug: Boolean flag for printing debug information.
        
        Returns:
            Logits tensor.
        """
        if self.network_config == 'default':
            # Standard forward using self.model.
            if debug:
                for layer in self.model:
                    inputs = layer(inputs)
                    print(f'Layer output shape: {inputs.shape}')
            else:
                inputs = self.model(inputs)
        elif self.network_config == 'dense':
            # Dense connectivity: accumulate features in a list.
            features = [inputs]
            for layer in self.layers:
                # Compute common spatial size among all current features.
                common_h = min(feat.shape[2] for feat in features)
                common_w = min(feat.shape[3] for feat in features)
                # Resize all features to the common size.
                resized_features = [F.interpolate(feat, size=(common_h, common_w),
                                                mode='bilinear', align_corners=False)
                                    for feat in features]
                # Concatenate along the channel dimension.
                concatenated = torch.cat(resized_features, dim=1)
                out = layer(concatenated)
                # If the current layer is a pooling operation, reset the features list.
                if isinstance(layer, (AvgPooling, MaxPooling, StochasticPooling)):
                    features = [out]
                else:
                    features.append(out)
                if debug:
                    print(f"{layer.__class__.__name__}: concatenated shape = {concatenated.shape}, "
                        f"output shape = {out.shape}")
            
            # Before the FC layer, adjust all features to a common spatial size.
            common_h = min(feat.shape[2] for feat in features)
            common_w = min(feat.shape[3] for feat in features)
            adjusted_features = [F.interpolate(feat, size=(common_h, common_w),
                                            mode='bilinear', align_corners=False)
                                for feat in features]
            inputs = torch.cat(adjusted_features, dim=1)
        
        inputs = torch.flatten(inputs, 1)
        if self.fc is None:
            seq_splits = 4
            input_features = inputs.size(1)
            for seq_splits in range(min(4, input_features), 0, -1):
                if input_features % seq_splits == 0:
                    break
            # print(f"[INFO] Using seq_splits: {seq_splits}")

            # self.fc = TwoLayerLSTMRegressor(
            #     input_features=input_features,
            #     seq_splits=seq_splits,
            #     hidden_sizes=[128, 64],
            #     output_dim=self.num_classes,
            #     dropout=0.5
            # )

            self.fc = LSTMRegressor(
                input_features=input_features,
                units=128,
                seq_splits=seq_splits,
                output_dim=self.num_classes
            )
            # self.fc = FullyConnected(input_features=inputs.size(1), units=self.num_classes)
        logits = self.fc(inputs)
        return logits


class MultiSensorNetworkGraphOld(nn.Module):
    def __init__(self, num_classes, network_config, network_gap, in_channels=1):
        """
        Multi-Sensor Neural Network: One CNN per sensor -> concat -> LSTM -> FC
        
        Args:
            num_classes (int): Number of output classes.
            network_config (dict): Network configuration from QNAS.
            network_gap (bool): Whether to use Global Average Pooling (if applicable).
            in_channels (int): Number of input channels per sensor (size along window dimension, e.g., 3).
        """
        super(MultiSensorNetworkGraphOld, self).__init__()
        self.num_classes = num_classes
        self.network_config = network_config
        self.network_gap = network_gap
        self.in_channels = in_channels

        self.sensor_cnns = nn.ModuleList()
        self.lstm = None
        self.fc = None

    def create_functions(self, fn_dict, net_list, cbam=False, num_sensors=None):
        """
        Creates a CNN (from fn_dict) for each sensor.

        Args:
            fn_dict (dict): Dictionary with CNN layers/functions per net.
            net_list (list): List of network names to use (e.g., ['net1', 'net2']).
            cbam (bool): Optional attention module (ignored for now).
            num_sensors (int): Number of sensors (determined from input data shape at runtime if not provided).
        """
        self.fn_dict = fn_dict
        self.net_list = net_list
        self.cbam = cbam
        self.num_sensors = num_sensors

        if self.num_sensors is None:
            raise ValueError("num_sensors must be provided in create_functions")

        for _ in range(self.num_sensors):
            layers = []
            in_channels = self.in_channels
            for key in net_list:
                if key in fn_dict:
                    layer_fn = fn_dict[key]
                    layer = layer_fn(in_channels)
                    layers.append(layer)
                    if hasattr(layer, 'out_channels'):
                        in_channels = layer.out_channels
            self.sensor_cnns.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Forward pass for multi-sensor data.

        Args:
            x (Tensor): Shape (num_sensors, num_samples, num_windows, window_size)

        Returns:
            Tensor: Model output (batch_size, num_classes)
        """
        num_sensors, num_samples, num_windows, window_size = x.shape

        # Verificação se a quantidade de sensores bate com o número de CNNs criadas
        if len(self.sensor_cnns) != num_sensors:
            raise ValueError(f"Expected {len(self.sensor_cnns)} CNNs for {num_sensors} sensors.")

        sensor_outputs = []

        for sensor_idx in range(num_sensors):
            # Dados do sensor: (num_samples, num_windows, window_size)
            sensor_input = x[sensor_idx]  # Shape: (num_samples, num_windows, window_size)

            # Reformata para (num_samples, window_size, num_windows) para se adaptar ao Conv1d
            sensor_input = sensor_input.permute(0, 2, 1)  # Shape: (num_samples, window_size, num_windows)
            
            # Ajusta para (num_samples, channels, seq_len)
            sensor_input = sensor_input  # Aqui, channels = window_size, seq_len = num_windows
            
            # Aplica a CNN correspondente ao sensor
            cnn = self.sensor_cnns[sensor_idx]
            sensor_out = cnn(sensor_input)  # Output: (num_samples, out_channels, new_seq_len)

            # Transforma para (num_samples, new_seq_len, out_channels)
            sensor_out = sensor_out.permute(0, 2, 1)

            sensor_outputs.append(sensor_out)

        # Ajusta o comprimento de sequência (seq_len) se houver alteração pelo CNN
        min_seq_len = min([out.shape[1] for out in sensor_outputs])
        sensor_outputs = [out[:, :min_seq_len, :] for out in sensor_outputs]

        # Concatena as saídas ao longo da dimensão de features
        x_concat = torch.cat(sensor_outputs, dim=2)  # Shape: (num_samples, min_seq_len, total_features)

        # Cria a LSTM na primeira passada (tamanho dinâmico)
        if self.lstm is None:
            input_size_lstm = x_concat.shape[2]
            self.lstm = nn.LSTM(
                input_size=input_size_lstm,
                hidden_size=128,
                num_layers=2,
                batch_first=True
            )

        # Passa pela LSTM
        lstm_out, _ = self.lstm(x_concat)  # Shape: (num_samples, min_seq_len, 128)

        # Pega a última saída temporal
        last_timestep_out = lstm_out[:, -1, :]  # Shape: (num_samples, 128)

        # Cria a FC na primeira passada
        if self.fc is None:
            self.fc = nn.Linear(last_timestep_out.shape[1], self.num_classes)

        # Saída final
        output = self.fc(last_timestep_out)  # Shape: (num_samples, num_classes)

        return output
    

class MultiHeadNetworkGraphNew(nn.Module):
    def __init__(self,
                 num_classes,
                 network_config,
                 network_gap,
                 num_lstm_cells_1,
                 num_lstm_cells_2,
                 in_channels=1,
                 num_sensors=None):
        """
        Multi-Sensor CNN-LSTM Network: One CNN per sensor -> Concat -> LSTM -> FC.

        Args:
            num_classes (int): Number of output classes.
            network_config (str): Network architecture type ('default', etc.).
            network_gap (bool): Use GAP or not (ignored here).
            in_channels (int): Number of channels per window (should be 1 for your case).
            num_sensors (int): Number of sensors (e.g., 14).
        """
        super(MultiHeadNetworkGraphNew, self).__init__()
        self.num_classes = num_classes
        self.network_config = network_config
        self.network_gap = network_gap
        self.in_channels = in_channels
        self.num_sensors = num_sensors
        self.num_lstm_cells_1 = num_lstm_cells_1
        self.num_lstm_cells_2 = num_lstm_cells_2

        self.sensor_cnns = nn.ModuleList()
        self.lstm1 = None
        self.lstm2 = None
        self.dropout = nn.Dropout(p=0.0)
        self.fc = None

    def create_functions(self, fn_dict, net_list, cbam=False):
        """
        Build a CNN (from QNAS) for each sensor.

        Args:
            fn_dict (dict): Block definitions from QNAS.
            net_list (list): List of block names for the network.
            cbam (bool): Optional CBAM block (not used here).
        """
        if self.num_sensors is None:
            raise ValueError("num_sensors must be set!")

        primary_blocks = {
            'ConvBlock', 'Conv1DBlock', 'DWConvBlock', 'SEConvBlock', 'ResidualV1CBAM',
            'MBConv', 'MBConvV2', 'MBConv_EPPGA', 'ResidualV1', 'ResidualV1Pr'
        }

        for _ in range(self.num_sensors):
            layers = []
            in_channels = self.in_channels
            for name in net_list:
                parameters = fn_dict[name]
                func = parameters['function']
                if func == 'NoOp':
                    continue

                params = parameters['params'].copy()

                if func in primary_blocks:
                    params['in_channels'] = in_channels
                    in_channels = params['filters']
                elif func == 'CBAMBlock':
                    params['in_channels'] = in_channels

                layer_class = functions_dict[func]
                layers.append(layer_class(**params))

            layers.append(nn.Flatten(start_dim=1))
            self.sensor_cnns.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Forward pass for the MultiHeadNetworkGraphNew model.

        Args:
            x (list[Tensor]): List of sensor inputs, each of shape 
                              (batch_size, num_windows, window_size, channels=1)

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes)
        """
        features = []
        for i, sensor_input in enumerate(x):
            b, t, w, c = sensor_input.shape  # batch, windows, window_length, channels

            # Reorganiza para (batch, time, channels, window_length)
            sensor_input = sensor_input.permute(0, 1, 3, 2)  # (b, t, c=1, w)

            # Junta batch e time steps: (b * t, c, w)
            sensor_input_reshaped = sensor_input.reshape(b * t, c, w)

            # Passa apenas o bloco Conv (sem Flatten)
            conv_out = self.sensor_cnns[i][0](sensor_input_reshaped)  # (b * t, num_filters, new_w)

            # Aplica Flatten manualmente: (b * t, num_filters * new_w)
            conv_out = conv_out.view(b, t, -1)  # (b, t, features_per_sensor)

            features.append(conv_out)

        # Concatena as features dos sensores na última dimensão
        x_concat = torch.cat(features, dim=2)  # (batch, time, total_features)

         # Inicializa LSTM1 dinamicamente (se necessário)
        if self.lstm1 is None:
            input_size = x_concat.shape[2]
            self.lstm1 = nn.LSTM(input_size=input_size,
                                 hidden_size=self.num_lstm_cells_1,
                                 batch_first=True)
            self.lstm2 = nn.LSTM(input_size=self.num_lstm_cells_1,
                                 hidden_size=self.num_lstm_cells_2,
                                 batch_first=True)
                                 
        # Passa pelas LSTMs
        lstm_out1, _ = self.lstm1(x_concat)         # (batch, time, lstm1_units)
        lstm_out2, (hn, _) = self.lstm2(lstm_out1)  # (batch, time, lstm2_units)
        final_output = hn.squeeze(0)                # (batch, lstm2_units)

        dropped = self.dropout(final_output)

        # Inicializa FC na primeira passagem, se necessário
        if self.fc is None:
            self.fc = nn.Linear(self.num_lstm_cells_2, self.num_classes)

        # Aplica camada totalmente conectada à saída final da LSTM (último hidden state)
        output = self.fc(dropped)  # (batch, num_classes)

        return output