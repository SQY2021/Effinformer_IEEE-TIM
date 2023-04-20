import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activation import Swish, Relu, Gelu, SRelu,Elu,Lrelu,ReGLU,GEGLU,SWGLU
class DilatedCausalConv1d(nn.Conv1d):
    """
        Dilated Causal Convolutional layer implementation.
        It combines two concepts:
            - causal convolution -> a convolutional layer that is able to respect the ordering of the data
            - dilated convolution -> a convolutional layer where the filter is applied over an area larger than its length by skipping input values with a certain step
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Define padding
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(input, (self.__padding, 0)))

class ConvLayer(nn.Module):
    def __init__(self, c_in,d):
        super(ConvLayer, self).__init__()
        self.dilation = d
        #padding = 1 if torch.__version__>='1.5.0' else 2

        # self.downConv = nn.Conv1d(in_channels=c_in,
        #                           out_channels=c_in,
        #                           kernel_size=3,
        #                           padding=0,
        #                           dilation=self.dilation)
        self.downConv = DilatedCausalConv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  dilation=self.dilation)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
# Implementation of the second detail Causal Conv1d layer
# class ConvLayer(nn.Module):
#     def __init__(self, c_in, d):
#         super(ConvLayer, self).__init__()
#         self.downConv = nn.Conv1d(in_channels=c_in,
#                                   out_channels=c_in,
#                                   kernel_size=3,
#                                   padding=0,
#                                   stride=1
#                                   )
#         self.pad1 = nn.Conv1d(in_channels=c_in,
#                               out_channels=c_in,
#                               kernel_size=1,
#                               padding=0,
#                               stride=1
#                               )
#         self.norm = nn.BatchNorm1d(c_in)
#         self.activation = nn.ELU()
#         self.d = d
#         self.dropout = nn.Dropout(0.1)
#         self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         # print(self.d)
#         if self.d == 1:
#             x_i = x.clone()
#             x_p1 = self.downConv(x.permute(0, 2, 1))
#             x_p2 = self.pad1(x_i[:, 0:2, :].permute(0, 2, 1))
#             x_p = torch.cat((x_p1, x_p2), 2)
#             x = self.norm(x_p)
#             x = self.dropout(self.activation(x))
#             x = x + x_i.permute(0, 2, 1)
#             x = self.maxPool(x)
#             x = x.transpose(1, 2)
#             return x
#         elif self.d == 2:
#             x_i = x.clone()
#             x_p = x.permute(0, 2, 1)
#             x1 = x[:, 0::2, :]
#             x1_p1 = self.downConv(x1.permute(0, 2, 1))
#             x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
#             x1_p = torch.cat((x1_p1, x1_p2), 2)
#             x2 = x[:, 1::2, :]
#             x2_p1 = self.downConv(x2.permute(0, 2, 1))
#             x2_p2 = self.pad1(x2[:, 0:2, :].permute(0, 2, 1))
#             x2_p = torch.cat((x2_p1, x2_p2), 2)
#             for i in range(x_p.shape[2]):
#                 if i % 2 == 0:
#                     x_p[:, :, i] = x1_p[:, :, i // 2]
#                 else:
#                     x_p[:, :, i] = x2_p[:, :, i // 2]
#             x = self.norm(x_p)
#             x = self.dropout(self.activation(x))
#             x = x + x_i.permute(0, 2, 1)
#             x = self.maxPool(x)
#             x = x.transpose(1, 2)
#             return x
#         else:
#             x_i = x.clone()
#             x_p = x.permute(0, 2, 1)
#             for i in range(self.d):
#                 x1 = x[:, i::self.d, :]
#                 x1_p1 = self.downConv(x1.permute(0, 2, 1))
#                 x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
#                 x1_p = torch.cat((x1_p1, x1_p2), 2)
#                 for j in range(x_p.shape[2]):
#                     if j % self.d == i:
#                         x_p[:, :, j] = x1_p[:, :, j // self.d]
#             x = self.norm(x_p)
#             x = self.dropout(self.activation(x))
#             x = x + x_i.permute(0, 2, 1)
#             x = self.maxPool(x)
#             x = x.transpose(1, 2)
#             return x
#
#
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #self.activation = F.relu if activation == "relu" else F.gelu
        if activation == "relu":
            self.activation = Relu()
        elif activation == "gelu":
            self.activation = Gelu()
        if activation == "srelu":
            self.activation = SRelu()
        if activation == "Elu":
            self.activation = Elu()
        elif activation == "Lrelu":
            self.activation = Lrelu()
        if activation == "ReGLU":
            self.activation = ReGLU()
        if activation == "GEGLU":
            self.activation = GEGLU()
        if activation == "SWGLU":
            self.activation = SWGLU()
        else:
            self.activation = Swish()

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
