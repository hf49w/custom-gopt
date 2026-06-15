# -*- coding: utf-8 -*-

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask):
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~attn_mask.unsqueeze(1), torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StreamingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class StreamingGOPT(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim, seq_len, phn_num, use_phone_embedding=True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.phn_num = phn_num
        self.use_phone_embedding = use_phone_embedding

        self.blocks = nn.ModuleList([StreamingBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 5, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.in_proj = nn.Linear(self.input_dim, embed_dim)
        self.phn_proj = nn.Linear(self.phn_num, embed_dim)
        self.cls_tokens = nn.Parameter(torch.zeros(1, 5, embed_dim))
        trunc_normal_(self.cls_tokens, std=.02)

        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    def build_streaming_mask(self, visible_lens, max_phone_tokens, main_context_tokens, right_context_tokens, device):
        total_len = max_phone_tokens + 5
        mask = torch.zeros((visible_lens.shape[0], total_len, total_len), dtype=torch.bool, device=device)
        for row, visible_len in enumerate(visible_lens.tolist()):
            visible_len = int(visible_len)
            cls_start = visible_len
            valid_len = visible_len + 5

            for tok_idx in range(visible_len):
                block_end = ((tok_idx // main_context_tokens) + 1) * main_context_tokens - 1
                max_key = min(visible_len - 1, block_end + right_context_tokens)
                mask[row, tok_idx, :max_key + 1] = True

            for cls_offset in range(5):
                query_idx = cls_start + cls_offset
                mask[row, query_idx, :query_idx + 1] = True

            for pad_idx in range(valid_len, total_len):
                mask[row, pad_idx, pad_idx] = True

        return mask

    def pack_sequence(self, token_embed, phn_ids):
        batch_size, max_phone_tokens, embed_dim = token_embed.shape
        visible_mask = phn_ids >= 0
        visible_lens = visible_mask.sum(dim=1)
        total_len = max_phone_tokens + 5

        packed = token_embed.new_zeros(batch_size, total_len, embed_dim)
        cls_positions = torch.zeros(batch_size, 5, dtype=torch.long, device=token_embed.device)

        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        for row, visible_len in enumerate(visible_lens.tolist()):
            visible_len = int(visible_len)
            if visible_len > 0:
                packed[row, :visible_len] = token_embed[row, :visible_len]
            cls_start = visible_len
            packed[row, cls_start:cls_start + 5] = cls_tokens[row]
            cls_positions[row] = torch.arange(cls_start, cls_start + 5, device=token_embed.device)

        packed = packed + self.pos_embed[:, :total_len]
        return packed, visible_lens, cls_positions

    def unpack_phone_states(self, hidden, visible_lens, max_phone_tokens):
        phone_hidden = hidden.new_zeros(hidden.shape[0], max_phone_tokens, hidden.shape[-1])
        for row, visible_len in enumerate(visible_lens.tolist()):
            visible_len = int(visible_len)
            if visible_len > 0:
                phone_hidden[row, :visible_len] = hidden[row, :visible_len]
        return phone_hidden

    def forward(self, x, phn, main_context_tokens=8, right_context_tokens=2):
        main_context_tokens = max(int(main_context_tokens), 1)
        right_context_tokens = max(int(right_context_tokens), 0)

        batch_size, max_phone_tokens, _ = x.shape
        visible_mask = phn >= 0
        x = self.in_proj(x) if self.embed_dim != self.input_dim else x

        phn_clamped = torch.clamp(phn.long() + 1, min=0, max=self.phn_num - 1)
        phn_one_hot = F.one_hot(phn_clamped, num_classes=self.phn_num).float()
        phn_embed = self.phn_proj(phn_one_hot) * visible_mask.unsqueeze(-1)
        token_embed = x + phn_embed if self.use_phone_embedding else x
        packed, visible_lens, cls_positions = self.pack_sequence(token_embed, phn)

        attn_mask = self.build_streaming_mask(
            visible_lens=visible_lens,
            max_phone_tokens=max_phone_tokens,
            main_context_tokens=main_context_tokens,
            right_context_tokens=right_context_tokens,
            device=x.device,
        )

        hidden = packed
        for blk in self.blocks:
            hidden = blk(hidden, attn_mask)

        phone_hidden = self.unpack_phone_states(hidden, visible_lens, max_phone_tokens)
        batch_index = torch.arange(batch_size, device=x.device).unsqueeze(1)
        cls_hidden = hidden[batch_index, cls_positions]

        u1 = self.mlp_head_utt1(cls_hidden[:, 0])
        u2 = self.mlp_head_utt2(cls_hidden[:, 1])
        u3 = self.mlp_head_utt3(cls_hidden[:, 2])
        u4 = self.mlp_head_utt4(cls_hidden[:, 3])
        u5 = self.mlp_head_utt5(cls_hidden[:, 4])

        p = self.mlp_head_phn(phone_hidden)
        w1 = self.mlp_head_word1(phone_hidden)
        w2 = self.mlp_head_word2(phone_hidden)
        w3 = self.mlp_head_word3(phone_hidden)
        return u1, u2, u3, u4, u5, p, w1, w2, w3


class StreamingGOPTNoPhn(StreamingGOPT):
    def __init__(self, embed_dim, num_heads, depth, input_dim, seq_len, phn_num):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            input_dim=input_dim,
            seq_len=seq_len,
            phn_num=phn_num,
            use_phone_embedding=False,
        )
