import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

# Dash CNN architecture
class AstroDashPyTorchNet(nn.Module):
    """
    PyTorch implementation of the AstroDash CNN.
    """
    def __init__(self, n_types, im_width=32):
        super().__init__()
        self.im_width = im_width
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        pooled_size = (self.im_width // 4)
        self.classifier = nn.Sequential(
            nn.Linear(64 * pooled_size * pooled_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, n_types),
        )

    def forward(self, x):
        x = x.view(-1, 1, self.im_width, self.im_width)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)

# Transformer blocks
class SinusoidalMLPPositionalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sine = torch.sin(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        cosine = torch.cos(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        encoding = torch.cat([sine, cosine], dim=-1)
        encoding = F.relu(self.fc1(encoding))
        encoding = self.fc2(encoding)
        return encoding

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1,
                 context_self_attn: bool = False):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        if context_self_attn:
            self.context_self_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
            self.layernorm_context = nn.LayerNorm(embed_dim)
        else:
            self.context_self_attn = nn.Identity()
            self.layernorm_context = nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        if context is not None:
            if not isinstance(self.context_self_attn, nn.Identity):
                context_attn_output, _ = self.context_self_attn(context, context, context, key_padding_mask=context_mask)
                context = self.layernorm_context(context + self.dropout(context_attn_output))
            cross_attn_output, _ = self.cross_attn(x, context, context, key_padding_mask=context_mask)
            x = self.layernorm2(x + self.dropout(cross_attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output))
        return x

# Transformer Classifier Architecture
class spectraTransformerEncoder(nn.Module):
    def __init__(self,
                 bottleneck_length: int,
                 model_dim: int,
                 num_heads: int,
                 num_layers: int,
                 num_classes: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 selfattn: bool = False):
        super().__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.redshift_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.flux_embd = nn.Linear(1, model_dim)
        self.transformerblocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, ff_dim, dropout, selfattn)
            for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, num_classes)
        )
    def forward(self, wavelength, flux, redshift, mask=None):
        flux_embd = self.flux_embd(flux[:, :, None]) + self.wavelength_embd_layer(wavelength)
        redshift_embd = self.redshift_embd_layer(redshift[:, None])
        if redshift_embd.dim() == 4 and redshift_embd.shape[2] == 1:
            redshift_embd = redshift_embd.squeeze(2)
        context = torch.cat([flux_embd, redshift_embd], dim=1)
        if mask is not None:
            mask = torch.cat([mask, torch.zeros(mask.shape[0], 1).to(torch.bool).to(mask.device)], dim=1)
        x = self.initbottleneck[None, :, :].repeat(context.shape[0], 1, 1)
        h = x
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, context, context_mask=mask)
        final_bottleneck = x + h
        pooled_features = self.pooling(final_bottleneck.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(pooled_features)
        return logits
