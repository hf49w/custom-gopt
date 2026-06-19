# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming_gopt import Mlp, StreamingBlock, trunc_normal_


def masked_mean(values, mask, dim, keepdim=False):
    mask = mask.to(dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return (values * mask).sum(dim=dim, keepdim=keepdim) / denom


class ReliabilityGate(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cn_stats, acoustic_stats):
        # cn_stats: eps, entropy, top1, margin, prefix_stability
        # acoustic_stats: entropy, margin, duration, PCN-Charsiu JS divergence
        gate_input = torch.stack(
            [
                cn_stats[..., 1],
                acoustic_stats[..., 0],
                acoustic_stats[..., 3],
                cn_stats[..., 4],
            ],
            dim=-1,
        )
        return torch.sigmoid(self.net(gate_input))


class PCNStreamingScorer(nn.Module):
    """
    Lightweight stateful streaming scorer for PCN/Charsiu/prosody features.

    Inputs follow streaming_pcn_gopt_v1:
    - cn_post: [B, T, phone_dim]
    - cn_stats: [B, T, 5]
    - acoustic_post: [B, T, phone_dim]
    - acoustic_stats: [B, T, 4]
    - prosody: [B, prosody_dim]
    - visible_len: [B]
    - commit_mask: [B, T]
    - word_ids: [B, T], optional, used for online word pooling
    """

    def __init__(
        self,
        phone_dim,
        seq_len,
        prosody_dim=14,
        embed_dim=40,
        num_heads=2,
        depth=2,
        pcn_embed_dim=16,
        acoustic_embed_dim=16,
        prosody_embed_dim=8,
        gru_dim=32,
        main_context_tokens=16,
        dropout=0.0,
    ):
        super().__init__()
        self.phone_dim = int(phone_dim)
        self.seq_len = int(seq_len)
        self.prosody_dim = int(prosody_dim)
        self.embed_dim = int(embed_dim)
        self.gru_dim = int(gru_dim)
        self.main_context_tokens = max(int(main_context_tokens), 1)

        self.pcn_proj = nn.Sequential(
            nn.Linear(self.phone_dim + 5, pcn_embed_dim),
            nn.LayerNorm(pcn_embed_dim),
            nn.GELU(),
        )
        self.acoustic_proj = nn.Sequential(
            nn.Linear(self.phone_dim + 4, acoustic_embed_dim),
            nn.LayerNorm(acoustic_embed_dim),
            nn.GELU(),
        )
        self.prosody_proj = nn.Sequential(
            nn.Linear(self.prosody_dim, max(16, prosody_embed_dim * 2)),
            nn.GELU(),
            nn.Linear(max(16, prosody_embed_dim * 2), prosody_embed_dim),
            nn.LayerNorm(prosody_embed_dim),
        )
        if pcn_embed_dim != acoustic_embed_dim:
            raise ValueError('pcn_embed_dim and acoustic_embed_dim must match for gated interpolation.')
        fused_dim = pcn_embed_dim + prosody_embed_dim
        self.reliability_gate = ReliabilityGate()
        self.fused_proj = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.blocks = nn.ModuleList(
            [
                StreamingBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    qkv_bias=True,
                    drop=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.word_pool_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.sentence_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_dim, batch_first=True)

        self.phone_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.word_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.asr_correct_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.uncertainty_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.confidence_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.abstention_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.utt_head = nn.Sequential(nn.LayerNorm(gru_dim), nn.Linear(gru_dim, 5))
        self.state_projection = nn.Sequential(nn.LayerNorm(gru_dim), nn.Linear(gru_dim, 128))

    def build_local_causal_mask(self, valid_mask):
        batch_size, seq_len = valid_mask.shape
        idx = torch.arange(seq_len, device=valid_mask.device)
        causal = idx.view(1, -1) <= idx.view(-1, 1)
        local = idx.view(1, -1) >= (idx.view(-1, 1) - self.main_context_tokens + 1)
        base = causal & local
        mask = base.unsqueeze(0).expand(batch_size, -1, -1).clone()
        key_valid = valid_mask.unsqueeze(1).expand(-1, seq_len, -1)
        query_valid = valid_mask.unsqueeze(2).expand(-1, -1, seq_len)
        mask = mask & key_valid & query_valid
        diag = torch.eye(seq_len, dtype=torch.bool, device=valid_mask.device).unsqueeze(0)
        return mask | diag

    def encode_slots(self, cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, valid_mask):
        z_pcn = self.pcn_proj(torch.cat([cn_post, cn_stats], dim=-1))
        z_acoustic = self.acoustic_proj(torch.cat([acoustic_post, acoustic_stats], dim=-1))
        z_prosody = self.prosody_proj(prosody).unsqueeze(1).expand(-1, cn_post.shape[1], -1)
        gate = self.reliability_gate(cn_stats, acoustic_stats)
        # gate=1 trusts Charsiu acoustic evidence; gate=0 trusts PCN posterior.
        evidence = gate * z_acoustic + (1.0 - gate) * z_pcn
        token = self.fused_proj(torch.cat([evidence, z_prosody], dim=-1))
        token = token + self.pos_embed[:, : token.shape[1]]
        token = token * valid_mask.unsqueeze(-1).to(token.dtype)
        return token, gate

    def local_causal_transformer(self, token, valid_mask):
        attn_mask = self.build_local_causal_mask(valid_mask)
        hidden = token
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)

    def pool_committed_words(self, hidden, commit_mask, word_ids=None):
        batch_size, seq_len, dim = hidden.shape
        pooled_rows = []
        pooled_masks = []
        max_words = 0
        row_groups = []
        for row in range(batch_size):
            active = torch.nonzero(commit_mask[row] > 0, as_tuple=False).squeeze(1).tolist()
            groups = []
            if active:
                if word_ids is None:
                    groups = [[idx] for idx in active]
                else:
                    cur_group = [active[0]]
                    cur_word = int(word_ids[row, active[0]].item())
                    for idx in active[1:]:
                        next_word = int(word_ids[row, idx].item())
                        if next_word == cur_word and next_word >= 0:
                            cur_group.append(idx)
                        else:
                            groups.append(cur_group)
                            cur_group = [idx]
                            cur_word = next_word
                    groups.append(cur_group)
            row_groups.append(groups)
            max_words = max(max_words, len(groups))

        max_words = max(max_words, 1)
        for row, groups in enumerate(row_groups):
            pooled = hidden.new_zeros(max_words, dim)
            mask = hidden.new_zeros(max_words)
            for group_idx, positions in enumerate(groups):
                pos = torch.tensor(positions, dtype=torch.long, device=hidden.device)
                pooled[group_idx] = self.word_pool_proj(hidden[row, pos].mean(dim=0))
                mask[group_idx] = 1.0
            pooled_rows.append(pooled)
            pooled_masks.append(mask)
        return torch.stack(pooled_rows, dim=0), torch.stack(pooled_masks, dim=0)

    def update_sentence_state(self, committed_words, word_mask, prev_state):
        batch_size = committed_words.shape[0]
        states = []
        for row in range(batch_size):
            h0 = (
                prev_state[:, row : row + 1, :]
                if prev_state is not None
                else committed_words.new_zeros(1, 1, self.gru_dim)
            )
            word_count = int(word_mask[row].sum().item())
            if word_count <= 0:
                states.append(h0)
                continue
            _, h_row = self.sentence_gru(committed_words[row : row + 1, :word_count], h0)
            states.append(h_row)
        return torch.cat(states, dim=1)

    def forward(
        self,
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len,
        commit_mask,
        word_ids=None,
        prev_state=None,
    ):
        batch_size, seq_len, _ = cn_post.shape
        device = cn_post.device
        token_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = token_idx < visible_len.to(device).unsqueeze(1)
        commit_mask = (commit_mask > 0) & valid_mask

        token, gate = self.encode_slots(cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, valid_mask)
        slot_hidden = self.local_causal_transformer(token, valid_mask)
        committed_words, word_mask = self.pool_committed_words(slot_hidden, commit_mask, word_ids=word_ids)

        if prev_state is not None and prev_state.dim() == 2:
            prev_state = prev_state.unsqueeze(0)
        final_state = self.update_sentence_state(committed_words, word_mask, prev_state)
        sentence_state = final_state[-1]

        phone_score = self.phone_head(slot_hidden)
        word_scores = self.word_head(slot_hidden)
        asr_correct_logits = self.asr_correct_head(slot_hidden)
        uncertainty_logits = self.uncertainty_head(slot_hidden)
        confidence = torch.sigmoid(self.confidence_head(slot_hidden))
        abstention_logit = self.abstention_head(slot_hidden)
        utt_scores = self.utt_head(sentence_state)

        return {
            'phone_score': phone_score,
            'word_scores': word_scores,
            'utt_scores': utt_scores,
            'asr_correct_logits': asr_correct_logits,
            'uncertainty_logits': uncertainty_logits,
            'confidence': confidence,
            'abstention_logit': abstention_logit,
            'reliability_gate': gate,
            'slot_hidden': slot_hidden,
            'word_states': committed_words,
            'word_mask': word_mask,
            'sentence_state': sentence_state,
            'state_projection': self.state_projection(sentence_state),
            'next_state': final_state.detach(),
        }

    @torch.inference_mode()
    def stream_step(self, batch, prev_state=None):
        return self.forward(
            cn_post=batch['cn_post'],
            cn_stats=batch['cn_stats'],
            acoustic_post=batch['acoustic_post'],
            acoustic_stats=batch['acoustic_stats'],
            prosody=batch['prosody'],
            visible_len=batch['visible_len'],
            commit_mask=batch['commit_mask'],
            word_ids=batch.get('pcn_word_id'),
            prev_state=prev_state,
        )
