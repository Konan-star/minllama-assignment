from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    # 10000 ^ (-2i/d) (i=0, 1, 2, ..., d/2-1)
    positions = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    # 1*theta1  1*theta2  1*theta3  1*theta4
    # 2*theta1  2*theta2  2*theta3  2*theta4
    # 3*theta1  3*theta2  3*theta3  3*theta4
    # 4*theta1  4*theta2  4*theta3  4*theta4 みたいなやつ
    cos = freqs.cos()[:seqlen]
    sin = freqs.sin()[:seqlen]
    # queryとkeyの次元について、batch_sizeとn_local_headsは本質的に重要ではなく、並列計算することができればいいので、一つのcosとsin
    # を作成して、それをbroadcastすることで、queryとkeyの次元に合わせて計算することができる。
    cos = cos[None, :, None, :] # (1, seqlen, 1, head_dim/2)
    sin = sin[None, :, None, :]
    # | 次元     | query_real | cos[None,:] | 結果                |
    # | ------ | ---------- | ----------- | ----------------- |
    # | batch  | B          | 1           | 自動で B に broadcast |
    # | seqlen | S          | S           | 一致                |
    # | head   | H          | 1           | 自動で H に broadcast |
    # | dim    | D/2        | D/2         | 一致                |
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    q_real_out = query_real * cos - query_imag * sin #奇数行目
    q_imag_out = query_real * sin + query_imag * cos #偶数行目
    k_real_out = key_real * cos - key_imag * sin #奇数行目
    k_imag_out = key_real * sin + key_imag * cos #偶数行目

    # Return the rotary position embeddings for the query and key tensors
    query_out = torch.stack([q_real_out, q_imag_out], dim=-1).reshape(query.shape).to(query.dtype)
    key_out = torch.stack([k_real_out, k_imag_out], dim=-1).reshape(key.shape).to(key.dtype)
    return query_out, key_out