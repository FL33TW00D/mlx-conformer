from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import math
from pathlib import Path
import glob
import logging
import re

class MultiHeadAttentionFused(nn.Module):
    """
    Multi-Head Attention layer with fused QKV projection.

    This layer computes scaled dot-product attention similarly to
    [nn.MultiHeadAttention](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html),
    but fuses the linear projections for queries, keys, and values into a
    single `in_proj` layer for potential efficiency gains. It also supports
    `key_padding_mask`.

    Note: This implementation assumes that the input dimensions for queries,
    keys, and values are the same (`dims`).
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
        bias: bool = True,
    ):
        super().__init__()

        # Ensure divisibility
        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        # For fused projection, input dimensions must match 'dims'
        if query_input_dims is not None and query_input_dims != dims:
             raise ValueError(f"query_input_dims ({query_input_dims}) must equal dims ({dims}) for fused projection")
        if key_input_dims is not None and key_input_dims != dims:
             raise ValueError(f"key_input_dims ({key_input_dims}) must equal dims ({dims}) for fused projection")
        if value_input_dims is not None and value_input_dims != dims:
             raise ValueError(f"value_input_dims ({value_input_dims}) must equal dims ({dims}) for fused projection")

        # Set dimensions, using defaults if necessary
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.dims = dims # Store dims for splitting
        self.value_dims = value_dims # Store value_dims for splitting

        # Fused input projection for Q, K, V
        # Total output dimension = dims (Q) + dims (K) + value_dims (V)
        self.in_proj = nn.Linear(dims, dims + dims + value_dims, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, key_padding_mask: Optional[mx.array] = None):
        """
        Apply multi-head attention.

        Args:
            x (mx.array): Input tensor expected to be of shape
                (batch_size, sequence_length, dims). This tensor serves as the
                source for queries, keys, and values.
            mask (Optional[mx.array]): An additive mask for the attention scores.
                Typically used for causal masking. Shape should be broadcastable
                to (batch_size, num_heads, query_seq_len, key_seq_len).
            key_padding_mask (Optional[mx.array]): A boolean mask indicating
                padded positions in the keys. Shape (batch_size, key_seq_len).
                `True` indicates a padded position that should be masked.

        Returns:
            mx.array: The output tensor after attention and output projection.
        """
        B, L, _ = x.shape # Batch size, sequence length, dimensions

        # 1. Fused Projection and Split
        qkv = self.in_proj(x)
        queries, keys, values = mx.split(qkv, [self.dims, self.dims, self.value_dims], axis=-1)

        # 2. Reshape for Multi-Head Attention
        # Reshape from (B, L, D) to (B, num_heads, L, head_dim)
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        # Use value_dims for reshaping V
        value_head_dim = self.value_dims // self.num_heads
        values = values.reshape(B, L, self.num_heads, value_head_dim).transpose(0, 2, 1, 3)

        # 3. Scale
        head_dim = queries.shape[-1]
        scale = math.sqrt(1 / head_dim)

        # 4. Prepare Masking
        merged_mask = mask # Start with the causal/attention mask if provided

        if key_padding_mask is not None:
            # Convert boolean key_padding_mask to additive float mask
            # key_padding_mask is (B, L) -> needs to be (B, 1, 1, L) for broadcasting
            additive_key_padding_mask = key_padding_mask[:, None, None, :].astype(mx.float32) * mx.finfo(mx.float32).min

            if merged_mask is None:
                merged_mask = additive_key_padding_mask
            else:
                # Ensure mask has compatible float type before adding
                if merged_mask.dtype != additive_key_padding_mask.dtype:
                     # This might happen if causal mask was created with different dtype
                     merged_mask = merged_mask.astype(additive_key_padding_mask.dtype)
                merged_mask = merged_mask + additive_key_padding_mask

        # Note: The original code had a transpose here (merged_mask.transpose(3, 1, 2, 0))
        # which seemed incorrect for the expected mask shape of scaled_dot_product_attention.
        # The mask should broadcast to (B, num_heads, query_len, key_len).
        # The created merged_mask (from key_padding_mask and/or causal mask)
        # should already align with these dimensions or be broadcastable.

        # 5. Scaled Dot-Product Attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=merged_mask
        )

        # 6. Reshape and Output Projection
        # Reshape from (B, num_heads, L, value_head_dim) -> (B, L, num_heads * value_head_dim) = (B, L, value_dims)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1) # Use self.value_dims implied by flatten
        return self.out_proj(output)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        """
        Creates an additive causal mask for attention.

        Args:
            N (int): The sequence length.
            dtype (mx.Dtype, optional): The data type of the mask. Defaults to mx.float32.

        Returns:
            mx.array: A mask of shape (N, N) where future positions are masked.
        """
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        # Add dimensions for broadcasting: (1, 1, N, N) or just (N,N) if handled by attention fn
        mask = mask.astype(dtype) * mx.finfo(dtype).min
        # Often needs expansion for batch/heads, but depends on attention implementation.
        # mx.fast.scaled_dot_product_attention handles broadcasting from (N, N).
        return mask

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
        bias: bool = True,
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
                bias=True,
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
        self.self_attn = MultiHeadAttentionFused(input_dim, num_attention_heads)  # nodropout
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = Conv(
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
        x = self.conv_module(x)
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
        print("Model: ", model)
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
            if "in_proj_weight" in k:
                new_key = k.replace("in_proj_weight", "in_proj.weight")
                sanitized_weights[new_key] = v
            elif "in_proj_bias" in k:
                new_key = k.replace("in_proj_bias", "in_proj.bias")
                sanitized_weights[new_key] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights

if __name__ == "__main__":
    input_dim = 80
    conformer = Conformer.from_pretrained("./weights")
    lengths = mx.random.randint(1, 400, (10,))  # (batch,)
    input = mx.random.uniform(
        low=0, high=1, shape=[10, int(lengths.max()), input_dim]
    )  # (batch, num_frames, input_dim)
    output = conformer(input, lengths)
    print(f"Output: {output}")
