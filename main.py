from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import math
from pathlib import Path
import glob
import logging

class MultiHeadAttention(nn.Module):
    """
    Identical to [nn.MultiHeadAttention](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html)
    but with support for `key_padding_mask`
    """
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.query_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.key_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.value_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None, key_padding_mask=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        queries = mx.unflatten(queries, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        keys = mx.unflatten(keys, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        values = mx.unflatten(values, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        scale = math.sqrt(1 / queries.shape[-1])

        if key_padding_mask is not None:
            # key_padding_mask is (batch_size, seq_len)
            key_padding_mask = key_padding_mask[:, None, None, :]
            key_padding_mask_expanded = mx.repeat(key_padding_mask, num_heads, axis=1)
            if mask is not None:
                merged_mask = mask + key_padding_mask_expanded
            else:
                merged_mask = key_padding_mask_expanded
        else:
            merged_mask = mask
        merged_mask = merged_mask.transpose(3, 1, 2, 0)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=merged_mask
        )
        output = output.transpose(0, 2, 1, 3).flatten(-2, -1)
        return self.out_proj(output)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * mx.finfo(dtype).min
        return mask


class Conv(nn.Module):
    r"""Conformer convolution module.

        Args:
            input_dim (int): input dimension.
            num_channels (int): number of depthwise convolution layer input channels.
    depthwise_kernel_size (int): kernel size of depthwise convolution layer.
            dropout (float, optional): dropout probability. (Default: 0.0)
            bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
            use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError(
                "depthwise_kernel_size must be odd to achieve 'SAME' padding."
            )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.GLU(axis=2),
            nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, dims=num_channels)
            if use_group_norm
            else nn.BatchNorm(num_channels),
            nn.SiLU(),
            nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        x = self.layer_norm(x)
        x = self.sequential(x)
        return x.transpose(0, 2, 1)


class FeedForward(nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim, bias=True),
            nn.Dropout(dropout),
        )

    def __call__(self, input: mx.array) -> mx.array:
        r"""
        Args:
            input (mx.array): with shape `(*, D)`.

        Returns:
            Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = FeedForward(input_dim, ffn_dim, dropout)

        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.self_attn = MultiHeadAttention(input_dim, num_attention_heads)  # nodropout
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv = Conv(
            input_dim,
            input_dim,
            depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = FeedForward(input_dim, ffn_dim, dropout)
        self.final_layer_norm = nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_conv(self, x: mx.array) -> mx.array:
        residual = x
        x = x.transpose(1, 0, 2)
        x = self.conv(x)
        x = x.transpose(2, 0, 1)
        x = x + residual
        return x

    def __call__(
        self, input: mx.array, key_padding_mask: Optional[mx.array]
    ) -> mx.array:
        r"""
        Args:
            input (mx.array): input, with shape `(T, B, D)`.
            key_padding_mask (mx.array or None): key padding mask to use in self attention layer.

        Returns:
            mx.array: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_conv(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_conv(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
         conformer = Conformer(
             input_dim=80,
             num_heads=4,
             ffn_dim=128,
             num_layers=4,
             depthwise_conv_kernel_size=31,
         )
         lengths = mx.random.randint(1, 400, (10,))  # (batch,)
         input = mx.random.uniform(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
         output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = [
            ConformerLayer(
                input_dim,
                ffn_dim,
                num_heads,
                depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
                convolution_first=convolution_first,
            )
            for _ in range(num_layers)
        ]

    def _lengths_to_padding_mask(self, lengths: mx.array) -> mx.array:
        batch_size = lengths.shape[0]
        max_length = int(mx.max(lengths).item())
        padding_mask = mx.repeat(mx.arange(max_length)[None, :], batch_size, axis=0)
        return padding_mask >= lengths[:, None]

    def __call__(self, input: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        r"""
        Args:
            input (mx.array): with shape `(B, T, input_dim)`.
            lengths (mx.array): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (mx.array, mx.array)
                mx.array
                    output frames, with shape `(B, T, input_dim)`
                mx.array
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = self._lengths_to_padding_mask(lengths)

        x = input.transpose(1, 0, 2)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return input.transpose(1, 0, 2), lengths


    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        model = Conformer(
            input_dim=80,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # pytorch conv2d expects the weight tensor to be of shape [out_channels, in_channels, kH, KW]
                # mlx conv2d expects the weight tensor to be of shape [out_channels, kH, KW, in_channels]
                sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights


if __name__ == "__main__":
    input_dim = 80
    conformer = Conformer.from_pretrained(".")
    lengths = mx.random.randint(1, 400, (10,))  # (batch,)
    input = mx.random.uniform(
        low=0, high=1, shape=[10, int(lengths.max()), input_dim]
    )  # (batch, num_frames, input_dim)
    output = conformer(input, lengths)
    print(f"Output: {output}")
