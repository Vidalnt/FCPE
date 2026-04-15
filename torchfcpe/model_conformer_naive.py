import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from local_attention import LocalAttention
from torch import nn

# From https://github.com/CNChTu/Diffusion-SVC/ by CNChTu
# License: MIT


class ConformerNaiveEncoder(nn.Module):
    """
    Conformer Naive Encoder

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
        use_pre_norm (bool): Whether to use pre-norm, default False
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        dim_model: int,
        use_norm: bool = False,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1  # Deprecated code, kept for compatibility only
        self.attention_dropout = 0.1  # Deprecated code, kept for compatibility only

        self.encoder_layers = nn.ModuleList(
            [
                CFNEncoderLayer(
                    dim_model,
                    num_heads,
                    use_norm,
                    conv_only,
                    conv_dropout,
                    atten_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, mask)
        return x  # (#batch, length, dim_model)


class CFNEncoderLayer(nn.Module):
    """
    Conformer Naive Encoder Layer

    Args:
        dim_model (int): Dimension of model
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.1
        atten_dropout (float): Dropout rate of attention module, default 0.1
        use_pre_norm (bool): Whether to use pre-norm, default False
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int = 8,
        use_norm: bool = False,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.0,
    ):
        super().__init__()

        # We reincorporate the exact if to match the original FCPE logic.
        if conv_dropout > 0.0:
            self.conformer = nn.Sequential(
                LynxNet2ConvModule(dim_model), nn.Dropout(conv_dropout)
            )
        else:
            self.conformer = LynxNet2ConvModule(dim_model)

        self.norm = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(0.1)  # Deprecated code, kept for compatibility only

        # selfatt -> fastatt: performer!
        if not conv_only:
            self.attn = SelfAttention(
                dim=dim_model,
                heads=num_heads,
                causal=False,
                use_norm=use_norm,
                dropout=atten_dropout,
            )
        else:
            self.attn = None

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        if self.attn is not None:
            x = x + (self.attn(self.norm(x), mask=mask))

        x = x + (self.conformer(x))

        return x  # (#batch, length, dim_model)


class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            out_min, out_max = torch.aminmax(out.detach())
            gate_min, gate_max = torch.aminmax(gate.detach())
            max_abs_out = torch.max(-out_min, out_max).float()
            max_abs_gate = torch.max(-gate_min, gate_max).float()
            max_abs_value = max_abs_out * max_abs_gate
            if max_abs_value > 1000:
                ratio = (1000 / max_abs_value).half()
                gate *= ratio
                return (out * gate).clamp(-1000 * ratio, 1000 * ratio) / ratio
        return out * gate


class LynxNet2ConvModule(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.0,
    ):
        super().__init__()

        inner_dim = int(dim * expansion_factor)

        if float(dropout) > 0.0:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()

        # We use nn.Sequential to maintain an idiomatic network architecture faithful to the original code
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(
                dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
            ),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            _dropout,
        )

    def forward(self, x):
        return self.net(x)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        qr_uniform_q=False,
        dropout=0.0,
        no_projection=False,
        use_norm=False,
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            qr_uniform_q=qr_uniform_q,
            no_projection=no_projection,
            use_norm=use_norm,
        )

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = (
            LocalAttention(
                window_size=local_window_size,
                causal=causal,
                autopad=True,
                dropout=dropout,
                look_forward=int(not causal),
                rel_pos_emb_config=(dim_head, local_heads),
            )
            if local_heads > 0
            else None
        )

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        name=None,
        inference=False,
        **kwargs,
    ):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.0)
            if cross_attend:
                pass
            else:
                out = self.fast_attention(q, k, v)
                attn_outs.append(out)

        if not empty(lq):
            assert (
                not cross_attend
            ), 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class FastAttention(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        qr_uniform_q=False,
        no_projection=False,
        use_norm=False,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
            qr_uniform_q=qr_uniform_q,
        )
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        self.no_projection = no_projection

        self.causal = causal
        self.use_norm = use_norm

        if self.causal or self.generalized_attention:
            raise NotImplementedError(
                'Causal and generalized attention not implemented yet'
            )

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.use_norm:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            raise NotImplementedError('generalized attention not implemented yet')

        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device
            )

            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out


def linear_attention(q, k, v):
    if v is None:
        out = torch.einsum('...ed,...nd->...ne', k, q)
        return out

    else:
        k_cumsum = k.sum(dim=-2)
        D_inv = 1.0 / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-8)

        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out


def softmax_kernel(
    data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data**2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer**2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True).values
            )
            + eps
        )
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data + eps))

    return data_dash.type_as(data)


def gaussian_orthogonal_random_matrix(
    nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None
):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device
        )
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device
        )
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones(
            (nb_rows,), device=device
        )
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))

    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def default(val, d):
    return val if exists(val) else d


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0
