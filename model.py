# -*- coding: utf-8 -*-
"""
Created on Thu May 12 23:25:54 2022

@author: Yuanhang Zhang
Adapted from https://github.com/pytorch/examples/blob/main/word_language_model/model.py

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pos_encoding import TQSPositionalEncoding1D, TQSPositionalEncoding2D
from model_utils import sample, sample_without_weight
from torch.nn import TransformerEncoderLayer
# from custom_transformer_layer import TransformerEncoderLayer

pi = np.pi


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        system_sizes,
        param_dim,
        embedding_size,
        n_head,
        n_hid,
        n_layers,
        phys_dim=2,
        dropout=0.5,
        minibatch=None,
    ):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder
        except:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            )

        self.system_sizes = torch.tensor(
            system_sizes, dtype=torch.int64
        )  # (n_size, n_dim)
        assert len(self.system_sizes.shape) == 2
        self.n = self.system_sizes.prod(dim=1)  # (n_size, )
        self.n_size, self.n_dim = self.system_sizes.shape
        max_system_size, _ = self.system_sizes.max(dim=0)  # (n_dim, )

        self.size_idx = None
        self.system_size = None
        self.param = None
        self.prefix = None

        self.param_dim = param_dim
        self.phys_dim = phys_dim

        # input consists of: [phys_dim_0 phys_dim_1 log(system_size[0]) log(system_size[1]) parity(system_size) mask_token params]
        input_dim = phys_dim + self.n_dim + 2 + param_dim
        self.input_dim = input_dim

        # sequence consists of: [log(system_size[0]) log(system_size[1]) params spins]
        self.seq_prefix_len = self.n_dim + param_dim

        self.param_range = None

        self.n_head = n_head
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.dropout = dropout
        self.minibatch = minibatch

        self.src_mask = None

        pos_encoder = (
            TQSPositionalEncoding1D if self.n_dim == 1 else TQSPositionalEncoding2D
        )

        self.pos_encoder = pos_encoder(
            embedding_size, self.seq_prefix_len, dropout=dropout
        )
        # max_length = n + param_dim
        # self.pos_embedding = nn.Parameter(torch.empty(max_length, 1, embedding_size).normal_(std=0.02))

        encoder_layers = TransformerEncoderLayer(embedding_size, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Linear(input_dim, embedding_size)
        self.embedding_size = embedding_size
        self.amp_head = nn.Linear(embedding_size, phys_dim)
        self.phase_head = nn.Linear(embedding_size, phys_dim)
        # self.param_head = nn.Linear(embedding_size, 1)
        # param_head: (n_param, batch, embedding_size) -> (n_param, 1, batch/16)
        # perform conv-pooling on the batch dimension
        # hidden_size_1 = int(embedding_size / 2)
        # hidden_size_2 = int(embedding_size / 4)
        # self.param_head = nn.Sequential(nn.Conv1d(embedding_size, hidden_size_1, kernel_size=1),
        #                                 nn.ReLU(),
        #                                 nn.BatchNorm1d(hidden_size_1),
        #                                 nn.AvgPool1d(kernel_size=4),
        #                                 nn.Conv1d(hidden_size_1, hidden_size_2, kernel_size=1),
        #                                 nn.ReLU(),
        #                                 nn.BatchNorm1d(hidden_size_2),
        #                                 nn.AvgPool1d(kernel_size=4),
        #                                 nn.Conv1d(hidden_size_2, 1, kernel_size=1))
        # self.param_head = ParamHead(embedding_size)
        self.init_weights()

    def set_param(self, system_size=None, param=None):
        self.size_idx = torch.randint(self.n_size, [])
        if system_size is None:
            self.system_size = self.system_sizes[self.size_idx]
        else:
            self.system_size = system_size
            self.size_idx = None
        if param is None:
            self.param = self.param_range[0] + torch.rand(self.param_dim) * (
                self.param_range[1] - self.param_range[0]
            )
        else:
            self.param = param
        self.prefix = self.init_seq()

    def init_seq(self):
        system_size = self.system_size
        param = self.param
        parity = (system_size % 2).to(torch.get_default_dtype())  # (n_dim, )
        size_input = torch.diag(system_size.log())  # (n_dim, n_dim)

        init = torch.zeros(self.seq_prefix_len, 1, self.input_dim)

        # sequence consists of: [log(system_size[0]) log(system_size[1]) params spins]
        # input consists of: [phys_dim_0 phys_dim_1 log(system_size[0]) log(system_size[1]) parity(system_size) mask_token params]

        init[: self.n_dim, :, self.phys_dim : self.phys_dim + self.n_dim] = (
            size_input.unsqueeze(1)
        )  # (n_dim, 1, n_dim)
        init[: self.n_dim, :, self.phys_dim + self.n_dim] = parity.unsqueeze(
            1
        )  # (n_dim, 1)

        param_offset = self.phys_dim + self.n_dim + 2
        for i in range(self.param_dim):
            init[self.n_dim + i, :, param_offset + i] += param[i]
        return init  # (prefix_len, 1, input_dim)

    def wrap_spins(self, spins):
        """
        prefix: (prefix_len, 1, input_dim)
        spins: (n, batch)
        """
        prefix = self.prefix
        prefix_len, _, input_dim = prefix.shape
        n, batch = spins.shape
        src = torch.zeros(prefix_len + n, batch, input_dim)
        src[:prefix_len, :, :] = prefix
        src[prefix_len:, :, : self.phys_dim] = F.one_hot(
            spins.to(torch.int64), num_classes=self.phys_dim
        )
        return src

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.encoder.bias)
        nn.init.uniform_(self.amp_head.weight, -initrange, initrange)
        nn.init.zeros_(self.amp_head.bias)
        nn.init.uniform_(self.phase_head.weight, -initrange, initrange)
        nn.init.zeros_(self.phase_head.bias)

    @staticmethod
    def softsign(x):
        """
        Defined in Hibat-Allah, Mohamed, et al.
                    "Recurrent neural network wave functions."
                    Physical Review Research 2.2 (2020): 023358.
        Used as the activation function on the phase output
        range: (-2pi, 2pi)
        NOTE: this function outputs 2\phi, where \phi is the phase
              an additional factor of 2 is included, to ensure \phi\in(-\pi, \pi)
        """
        return 2 * pi * (1 + x / (1 + x.abs()))

    def forward(self, spins, compute_phase=True):
        # src: (seq, batch, input_dim)
        # use_symmetry: has no effect in this function
        # only included to be consistent with the symmetric version
        src = self.wrap_spins(spins)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
            self.src_mask = mask

        system_size = src[
            : self.n_dim, 0, self.phys_dim : self.phys_dim + self.n_dim
        ].diag()  # (n_dim, )
        system_size = system_size.exp().round().to(torch.int64)  # (n_dim, )

        result = []
        if self.minibatch is None:
            src = self.encoder(src) * math.sqrt(
                self.embedding_size
            )  # (seq, batch, embedding)
            # src = src + self.pos_embedding[:len(src)]  # (seq, batch, embedding)
            src = self.pos_encoder(src, system_size)  # (seq, batch, embedding)
            output = self.transformer_encoder(
                src, self.src_mask
            )  # (seq, batch, embedding)
            psi_output = output[
                self.seq_prefix_len - 1 :
            ]  # only use the physical degrees of freedom
            amp = F.log_softmax(
                self.amp_head(psi_output), dim=-1
            )  # (seq, batch, phys_dim)
            result.append(amp)
            if compute_phase:
                phase = self.softsign(
                    self.phase_head(psi_output)
                )  # (seq, batch, phys_dim)
                result.append(phase)
        else:
            batch = src.shape[1]
            minibatch = self.minibatch
            repeat = int(np.ceil(batch / minibatch))
            amp = []
            phase = []
            for i in range(repeat):
                src_i = src[:, i * minibatch : (i + 1) * minibatch]
                src_i = self.encoder(src_i) * math.sqrt(
                    self.embedding_size
                )  # (seq, batch, embedding)
                # src_i = src_i + self.pos_embedding[:len(src_i)]  # (seq, batch, embedding)
                src_i = self.pos_encoder(src_i, system_size)  # (seq, batch, embedding)
                output_i = self.transformer_encoder(
                    src_i, self.src_mask
                )  # (seq, batch, embedding)
                psi_output = output_i[
                    self.seq_prefix_len - 1 :
                ]  # only use the physical degrees of freedom
                amp_i = F.log_softmax(
                    self.amp_head(psi_output), dim=-1
                )  # (seq, batch, phys_dim)
                amp.append(amp_i)
                if compute_phase:
                    phase_i = self.softsign(
                        self.phase_head(psi_output)
                    )  # (seq, batch, phys_dim)
                    phase.append(phase_i)
            amp = torch.cat(amp, dim=1)
            result.append(amp)
            if compute_phase:
                phase = torch.cat(phase, dim=1)
                result.append(phase)
        return result

    @torch.no_grad()
    def forward_step(self, tokens, cache, start_pos, compute_phase=False):
        """
        Incremental (KV-cached) analogue of forward, for autoregressive sampling. Instead of
        re-encoding the whole prefix+spins sequence every step, it pushes only the `tokens`
        at absolute positions [start_pos, start_pos + T) through the stack, reusing each
        layer's cached keys/values for all earlier positions.

        This is mathematically identical to forward for a causal (lower-triangular) mask:
        with start_pos == 0 the whole prefix is processed as one causal chunk (priming the
        cache); every later call passes the single newly sampled token. The trained weights
        are used unchanged -- `_sdpa_attn_step` slices the stock fused `in_proj_weight`, so
        checkpoints load with no remap. Requires model.eval() (dropout must be inactive for
        exact equivalence) and the deduplication-free, fixed-batch sampling path.

        tokens : (T, batch, input_dim) raw input (same encoding as wrap_spins produces)
        cache  : KVCache (model_utils.KVCache), mutated in place
        start_pos : absolute sequence index of tokens[0]
        returns log_amp (and log_phase if compute_phase) of shape (T, batch, phys_dim)
        """
        x = self.encoder(tokens) * math.sqrt(self.embedding_size)  # (T, batch, embedding)
        # Absolute-position PE: forward adds self.pe[:seq_len], so position p uses pe[p].
        x = x + self.pos_encoder.pe[start_pos : start_pos + x.size(0)]
        is_prefill = start_pos == 0
        for layer_idx, layer in enumerate(self.transformer_encoder.layers):
            x = _encoder_layer_step(layer, x, cache, layer_idx, self.n_head, is_prefill)
        log_amp = F.log_softmax(self.amp_head(x), dim=-1)  # (T, batch, phys_dim)
        if compute_phase:
            return log_amp, self.softsign(self.phase_head(x))
        return log_amp


def _sdpa_attn_step(mha, x, cache, layer_idx, n_head, is_prefill):
    """
    Cache-aware self-attention using stock nn.MultiheadAttention parameters
    (fused `in_proj_weight`/`in_proj_bias` + `out_proj`), computed with
    F.scaled_dot_product_attention. Appends this step's keys/values to `cache` and attends
    over the full cached history.

    x : (T, batch, embed_dim). is_prefill=True processes a causal chunk (the prefix); a
    single-token step (T=1) is not masked because every cached key is a strictly earlier
    (hence valid) position. SDPA applies the 1/sqrt(head_dim) scaling itself, matching stock.
    """
    T, B, E = x.shape
    head_dim = E // n_head
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)  # (T, B, 3E), order [q, k, v]
    q, k, v = qkv.chunk(3, dim=-1)

    def to_heads(t):  # (T, B, E) -> (B, n_head, T, head_dim)
        return t.contiguous().view(T, B, n_head, head_dim).permute(1, 2, 0, 3)

    q, k, v = to_heads(q), to_heads(k), to_heads(v)

    if cache.k[layer_idx] is None:
        cache.k[layer_idx], cache.v[layer_idx] = k, v
    else:
        cache.k[layer_idx] = torch.cat([cache.k[layer_idx], k], dim=2)
        cache.v[layer_idx] = torch.cat([cache.v[layer_idx], v], dim=2)

    out = F.scaled_dot_product_attention(
        q, cache.k[layer_idx], cache.v[layer_idx], is_causal=is_prefill
    )  # (B, n_head, T, head_dim)
    out = out.permute(2, 0, 1, 3).contiguous().view(T, B, E)  # (T, B, E)
    return mha.out_proj(out)


def _encoder_layer_step(layer, x, cache, layer_idx, n_head, is_prefill):
    """
    One stock TransformerEncoderLayer forward, cache-aware. Honors the layer's own
    norm_first / activation so it matches whatever the checkpoint was trained with. Dropout
    submodules are intentionally omitted -- exact only in eval(), where they are identities.
    """
    act = getattr(layer, "activation", F.relu)
    if getattr(layer, "norm_first", False):
        a = _sdpa_attn_step(layer.self_attn, layer.norm1(x), cache, layer_idx, n_head, is_prefill)
        x = x + a
        x = x + layer.linear2(act(layer.linear1(layer.norm2(x))))
    else:
        a = _sdpa_attn_step(layer.self_attn, x, cache, layer_idx, n_head, is_prefill)
        x = layer.norm1(x + a)
        x = layer.norm2(x + layer.linear2(act(layer.linear1(x))))
    return x
