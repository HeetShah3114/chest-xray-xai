# aa_conv2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import conv1x1, conv3x3

class AAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dk, dv, nh, relative, input_dims, **kwargs):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.nh = nh
        self.relative = relative

        assert dk % nh == 0, 'nh must divide dk'
        assert dv % nh == 0, 'nh must divide dv'

        # `same` conv since conv and attn are concatenated in the output
        padding = kwargs.pop('padding', None)
        if not padding: padding = kernel_size//2

        self.conv = nn.Conv2d(in_channels, out_channels - dv, kernel_size, stride, padding, bias=False, **kwargs) if out_channels > dv else None
        self.in_proj_qkv = nn.Conv2d(in_channels, 2*dk + dv, kernel_size=1, stride=stride, bias=False)
        self.out_proj = nn.Conv2d(dv, dv, kernel_size=1, bias=False)

        if relative:
            H, W = input_dims
            self.key_rel_h = nn.Parameter(dk**-0.5 + torch.randn(dk//nh, 2*H-1))
            self.key_rel_w = nn.Parameter(dk**-0.5 + torch.randn(dk//nh, 2*W-1))

    def rel_to_abs(self, x):
        B, nh, L, _ = x.shape   # (B, nh, L, 2*L-1)

        # pad to shift from relative to absolute indexing
        x = F.pad(x, (0,1))     # (B, nh, L, 2*L)
        x = x.flatten(2)        # (B, nh, L*2*L)
        x = F.pad(x, (0,L-1))   # (B, nh, L*2*L + L-1)

        # reshape and slice out the padded elements
        x = x.reshape(B, nh, L+1, 2*L-1)
        return x[:,:,:L,L-1:]

    def relative_logits_1d(self, q, rel_k):
        B, nh, H, W, dkh = q.shape

        rel_logits = torch.matmul(q, rel_k)                                     # (B, nh, H, W, 2*W-1)
        # collapse height and heads
        rel_logits = rel_logits.reshape(B, nh*H, W, 2*W-1)
        rel_logits = self.rel_to_abs(rel_logits)                                # (B, nh*H, W, W)
        # shape back and tile height times
        return rel_logits.reshape(B, nh, H, 1, W, W).expand(-1,-1,-1,H,-1,-1)   # (B, nh, H, H, W, W)

    def forward(self, x):
        # compute qkv
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv.split([self.dk, self.dk, self.dv], dim=1)
        # split channels into multiple heads, flatten H,W dims and scale q; out (B, nh, dkh or dvh, HW)
        B, _, H, W = qkv.shape
        flat_q = q.reshape(B, self.nh, self.dk//self.nh, H, W).flatten(3) * (self.dk//self.nh)**-0.5
        flat_k = k.reshape(B, self.nh, self.dk//self.nh, H, W).flatten(3)
        flat_v = v.reshape(B, self.nh, self.dv//self.nh, H, W).flatten(3)

        logits = torch.matmul(flat_q.transpose(2,3), flat_k)    # (B, nh, HW, HW)
        if self.relative:
            q = flat_q.reshape(B, self.nh, self.dk//self.nh, H, W).permute(0,1,3,4,2)  # (B, nh, H, W, dkh)
            # compute relative logits in width dim
            w_rel_logits = self.relative_logits_1d(q, self.key_rel_w)                  # (B, nh, H, H, W, W)
            # repeat for heigh dim by transposing H,W and then permuting output
            h_rel_logits = self.relative_logits_1d(q.transpose(2,3), self.key_rel_h)   # (B, nh, W, W, H, H)
            # permute and reshape for adding to the attention logits
            w_rel_logits = w_rel_logits.permute(0,1,2,4,3,5).reshape(B, self.nh, H*W, H*W)
            h_rel_logits = h_rel_logits.permute(0,1,4,2,5,3).reshape(B, self.nh, H*W, H*W)
            # add to attention logits
            logits += h_rel_logits + w_rel_logits
        self.weights = F.softmax(logits, -1)

        attn_out = torch.matmul(self.weights, flat_v.transpose(2,3)) # (B, nh, HW, dvh)
        attn_out = attn_out.transpose(2,3)                           # (B, nh, dvh, HW)
        attn_out = attn_out.reshape(B, -1 , H, W)                    # (B, dv, H, W)
        attn_out = self.out_proj(attn_out)

        if self.conv is not None:
            return torch.cat([self.conv(x), attn_out], dim=1)
        else:
            return attn_out

    def extra_repr(self):
        return 'dk={dk}, dv={dv}, nh={nh}, relative={relative}'.format(**self.__dict__)