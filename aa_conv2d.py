#!/usr/bin/env python
"""
aa_conv2d.py

This module implements the AAConv2d class, an attention-augmented convolutional layer that
combines standard convolution with self-attention. The layer computes QKV projections and
optionally incorporates relative positional encodings. A sample run is provided in the __main__
block to test the functionality of the layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import conv1x1, conv3x3

class AAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dk, dv, nh, relative, input_dims, **kwargs):
        """
        Initializes the AAConv2d layer.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Total number of output channels; note that a part is reserved for attention.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            dk (int): Dimension of query and key vectors.
            dv (int): Dimension of the value vectors.
            nh (int): Number of attention heads.
            relative (bool): Whether to use relative positional encodings.
            input_dims (tuple): Spatial dimensions (H, W) of the input feature map.
            **kwargs: Additional keyword arguments for convolution.
        """
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.nh = nh
        self.relative = relative

        # Ensure that number of heads divides both dk and dv.
        assert dk % nh == 0, 'nh must divide dk'
        assert dv % nh == 0, 'nh must divide dv'

        # Determine padding for "same" convolution (if not provided, default to kernel_size//2)
        padding = kwargs.pop('padding', None)
        if not padding:
            padding = kernel_size // 2

        # Standard convolution: only used if out_channels is greater than dv.
        self.conv = nn.Conv2d(in_channels, out_channels - dv, kernel_size, stride, padding, bias=False, **kwargs) \
            if out_channels > dv else None
        # Projection layer for query, key, and value (QKV).
        self.in_proj_qkv = nn.Conv2d(in_channels, 2 * dk + dv, kernel_size=1, stride=stride, bias=False)
        # Projection layer for attention output.
        self.out_proj = nn.Conv2d(dv, dv, kernel_size=1, bias=False)

        # If relative positional encoding is enabled, initialize relative key parameters.
        if relative:
            H, W = input_dims
            self.key_rel_h = nn.Parameter(dk**-0.5 + torch.randn(dk // nh, 2 * H - 1))
            self.key_rel_w = nn.Parameter(dk**-0.5 + torch.randn(dk // nh, 2 * W - 1))

    def rel_to_abs(self, x):
        """
        Converts relative position representations to absolute ones.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, nh, L, 2*L-1).

        Returns:
            torch.Tensor: Tensor of shape (B, nh, L, L) with absolute position logits.
        """
        B, nh, L, _ = x.shape  # x shape: (B, nh, L, 2*L-1)
        x = F.pad(x, (0, 1))   # Pad to shape (B, nh, L, 2*L)
        x = x.flatten(2)       # Flatten to (B, nh, L*2*L)
        x = F.pad(x, (0, L - 1))  # Pad further to (B, nh, L*2*L + L-1)
        x = x.reshape(B, nh, L + 1, 2 * L - 1)  # Reshape to (B, nh, L+1, 2*L-1)
        return x[:, :, :L, L - 1:]

    def relative_logits_1d(self, q, rel_k):
        """
        Computes the relative attention logits along one spatial dimension.

        Parameters:
            q (torch.Tensor): Query tensor of shape (B, nh, H, W, dkh).
            rel_k (torch.Tensor): Relative key tensor of shape (dkh, 2*W-1).

        Returns:
            torch.Tensor: Relative logits tensor expanded to shape (B, nh, H, H, W, W).
        """
        B, nh, H, W, dkh = q.shape
        rel_logits = torch.matmul(q, rel_k)  # (B, nh, H, W, 2*W-1)
        # Collapse height and head dimensions.
        rel_logits = rel_logits.reshape(B, nh * H, W, 2 * W - 1)
        rel_logits = self.rel_to_abs(rel_logits)  # (B, nh*H, W, W)
        return rel_logits.reshape(B, nh, H, 1, W, W).expand(-1, -1, -1, H, -1, -1)

    def forward(self, x):
        """
        Performs the forward pass through the AAConv2d layer.

        Parameters:
            x (torch.Tensor): Input feature map of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output feature map after applying convolution and attention.
        """
        # Compute QKV projections.
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv.split([self.dk, self.dk, self.dv], dim=1)
        B, _, H, W = qkv.shape

        # Reshape and flatten for multi-head attention.
        flat_q = q.reshape(B, self.nh, self.dk // self.nh, H, W).flatten(3) * (self.dk // self.nh) ** -0.5
        flat_k = k.reshape(B, self.nh, self.dk // self.nh, H, W).flatten(3)
        flat_v = v.reshape(B, self.nh, self.dv // self.nh, H, W).flatten(3)

        # Compute attention logits.
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)  # (B, nh, HW, HW)
        if self.relative:
            # Reshape q for relative logits computation.
            q_reshaped = flat_q.reshape(B, self.nh, self.dk // self.nh, H, W).permute(0, 1, 3, 4, 2)
            # Compute relative logits for width dimension.
            w_rel_logits = self.relative_logits_1d(q_reshaped, self.key_rel_w)
            # Compute relative logits for height dimension.
            h_rel_logits = self.relative_logits_1d(q_reshaped.transpose(2, 3), self.key_rel_h)
            # Permute and reshape to add to the attention logits.
            w_rel_logits = w_rel_logits.permute(0, 1, 2, 4, 3, 5).reshape(B, self.nh, H * W, H * W)
            h_rel_logits = h_rel_logits.permute(0, 1, 4, 2, 5, 3).reshape(B, self.nh, H * W, H * W)
            logits += h_rel_logits + w_rel_logits

        # Apply softmax to get attention weights.
        self.weights = F.softmax(logits, dim=-1)
        # Compute attention output.
        attn_out = torch.matmul(self.weights, flat_v.transpose(2, 3))  # (B, nh, HW, dvh)
        attn_out = attn_out.transpose(2, 3)  # (B, nh, dvh, HW)
        attn_out = attn_out.reshape(B, -1, H, W)  # (B, dv, H, W)
        attn_out = self.out_proj(attn_out)

        # If a standard convolution branch is defined, concatenate its output.
        if self.conv is not None:
            return torch.cat([self.conv(x), attn_out], dim=1)
        else:
            return attn_out

    def extra_repr(self):
        """
        Provides an extra string representation of the AAConv2d layer parameters.

        Returns:
            str: String representation including dk, dv, nh, and whether relative encoding is used.
        """
        return 'dk={dk}, dv={dv}, nh={nh}, relative={relative}'.format(**self.__dict__)


def main():
    """
    Demonstrates a sample run of the AAConv2d layer using a dummy input.
    """
    # Sample configuration parameters.
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 1
    dk = 16
    dv = 8
    nh = 4
    relative = False  # Set to True to test relative positional encoding (requires valid input_dims)
    input_dims = (32, 32)  # Dummy spatial dimensions

    # Instantiate the AAConv2d layer.
    aa_conv = AAConv2d(in_channels, out_channels, kernel_size, stride, dk, dv, nh, relative, input_dims)
    # Create a dummy input tensor with batch size 1.
    dummy_input = torch.randn(1, in_channels, input_dims[0], input_dims[1])
    # Run the forward pass.
    output = aa_conv(dummy_input)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
