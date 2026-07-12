# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .streaming_gopt import StreamingBlock, trunc_normal_


def masked_mean(values, mask, dim, keepdim=False):
    mask = mask.to(dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return (values * mask).sum(dim=dim, keepdim=keepdim) / denom


class ReliabilityGate(nn.Module):
    def __init__(self, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(output_dim)),
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
    Stateful streaming scorer for streaming_pcn_gopt_v2_stateful.

    Local phone/word heads read all visible PCN slots. The sentence GRU is
    updated only from slots selected by new_commit_mask. cumulative_commit_mask
    is retained for losses/diagnostics and must not be used for GRU updates.
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
        use_state_projection=False,
        teacher_state_dim=128,
        utt_pooling_head='gru',
        fusion_mode='scalar_gate',
        slot_prosody_dim=0,
        slot_prosody_embed_dim=8,
        stress_branch='none',
        stress_grad_scale=0.2,
    ):
        super().__init__()
        self.phone_dim = int(phone_dim)
        self.seq_len = int(seq_len)
        self.prosody_dim = int(prosody_dim)
        self.embed_dim = int(embed_dim)
        self.gru_dim = int(gru_dim)
        self.main_context_tokens = max(int(main_context_tokens), 1)
        self.use_state_projection = bool(use_state_projection)
        self.utt_pooling_head = str(utt_pooling_head)
        self.fusion_mode = str(fusion_mode)
        self.slot_prosody_dim = int(slot_prosody_dim or 0)
        self.stress_branch = str(stress_branch)
        self.stress_grad_scale = float(stress_grad_scale)

        if self.utt_pooling_head not in {'gru', 'gru_visible'}:
            raise ValueError("utt_pooling_head must be 'gru' or 'gru_visible'.")
        if self.fusion_mode not in {'scalar_gate', 'concat_vector_gate'}:
            raise ValueError("fusion_mode must be 'scalar_gate' or 'concat_vector_gate'.")
        if self.stress_branch not in {'none', 'detached', 'gradscale'}:
            raise ValueError("stress_branch must be 'none', 'detached', or 'gradscale'.")

        if pcn_embed_dim != acoustic_embed_dim:
            raise ValueError('pcn_embed_dim and acoustic_embed_dim must match for gated interpolation.')

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
        gate_dim = pcn_embed_dim if self.fusion_mode == 'concat_vector_gate' else 1
        self.reliability_gate = ReliabilityGate(output_dim=gate_dim)
        self.slot_prosody_proj = None
        slot_prosody_fused_dim = 0
        if self.slot_prosody_dim > 0:
            slot_prosody_fused_dim = int(slot_prosody_embed_dim)
            self.slot_prosody_proj = nn.Sequential(
                nn.Linear(self.slot_prosody_dim, max(16, slot_prosody_fused_dim * 2)),
                nn.GELU(),
                nn.Linear(max(16, slot_prosody_fused_dim * 2), slot_prosody_fused_dim),
                nn.LayerNorm(slot_prosody_fused_dim),
            )
        evidence_dim = pcn_embed_dim * 4 if self.fusion_mode == 'concat_vector_gate' else pcn_embed_dim
        self.fused_proj = nn.Sequential(
            nn.Linear(evidence_dim + prosody_embed_dim + slot_prosody_fused_dim, embed_dim),
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
        self.stress_head = None
        if self.stress_branch != 'none':
            self.stress_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.asr_correct_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.uncertainty_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.confidence_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.abstention_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.utt_head = nn.Sequential(nn.LayerNorm(gru_dim), nn.Linear(gru_dim, 5))
        self.utt_visible_head = None
        if self.utt_pooling_head == 'gru_visible':
            visible_dim = gru_dim + embed_dim + embed_dim + 2
            self.utt_visible_head = nn.Sequential(nn.LayerNorm(visible_dim), nn.Linear(visible_dim, 5))
        self.state_projection = (
            nn.Sequential(nn.LayerNorm(gru_dim), nn.Linear(gru_dim, int(teacher_state_dim)))
            if self.use_state_projection
            else None
        )

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

    def encode_slots(self, cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, valid_mask, slot_prosody=None):
        z_pcn = self.pcn_proj(torch.cat([cn_post, cn_stats], dim=-1))
        z_acoustic = self.acoustic_proj(torch.cat([acoustic_post, acoustic_stats], dim=-1))
        z_prosody = self.prosody_proj(prosody).unsqueeze(1).expand(-1, cn_post.shape[1], -1)
        gate = self.reliability_gate(cn_stats, acoustic_stats)
        interpolation = gate * z_acoustic + (1.0 - gate) * z_pcn
        if self.fusion_mode == 'concat_vector_gate':
            evidence = torch.cat([z_pcn, z_acoustic, z_acoustic - z_pcn, interpolation], dim=-1)
        else:
            evidence = interpolation
        fused_parts = [evidence, z_prosody]
        if self.slot_prosody_proj is not None:
            if slot_prosody is None:
                slot_prosody = cn_post.new_zeros(cn_post.shape[0], cn_post.shape[1], self.slot_prosody_dim)
            fused_parts.append(self.slot_prosody_proj(slot_prosody))
        token = self.fused_proj(torch.cat(fused_parts, dim=-1))
        token = token + self.pos_embed[:, : token.shape[1]]
        token = token * valid_mask.unsqueeze(-1).to(token.dtype)
        return token, gate

    def local_causal_transformer(self, token, valid_mask):
        attn_mask = self.build_local_causal_mask(valid_mask)
        hidden = token
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)

    def pool_new_committed_words(self, hidden, new_commit_mask, word_ids=None):
        batch_size, _, dim = hidden.shape
        row_groups = []
        max_words = 0
        for row in range(batch_size):
            active = torch.nonzero(new_commit_mask[row] > 0, as_tuple=False).squeeze(1).tolist()
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
        pooled_rows = []
        pooled_masks = []
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

    def update_sentence_state(self, new_words, word_mask, prev_state):
        batch_size = new_words.shape[0]
        if prev_state is not None and prev_state.dim() == 2:
            prev_state = prev_state.unsqueeze(0)
        states = []
        for row in range(batch_size):
            if prev_state is None:
                h0 = new_words.new_zeros(1, 1, self.gru_dim)
            else:
                h0 = prev_state[:, row : row + 1, :]
            word_count = int(word_mask[row].sum().item())
            if word_count <= 0:
                states.append(h0)
                continue
            _, h_row = self.sentence_gru(new_words[row : row + 1, :word_count], h0)
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
        cumulative_commit_mask=None,
        new_commit_mask=None,
        commit_mask=None,
        word_ids=None,
        slot_prosody=None,
        prev_state=None,
        detach_next_state=False,
    ):
        batch_size, seq_len, _ = cn_post.shape
        device = cn_post.device
        token_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = token_idx < visible_len.to(device).unsqueeze(1)

        if cumulative_commit_mask is None:
            cumulative_commit_mask = commit_mask
        if cumulative_commit_mask is None:
            cumulative_commit_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if new_commit_mask is None:
            # Legacy fallback: safe only for stateless/full-sentence use.
            new_commit_mask = cumulative_commit_mask
        cumulative_commit_mask = (cumulative_commit_mask > 0) & valid_mask
        new_commit_mask = (new_commit_mask > 0) & valid_mask

        token, gate = self.encode_slots(cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, valid_mask, slot_prosody=slot_prosody)
        slot_hidden = self.local_causal_transformer(token, valid_mask)
        new_words, new_word_mask = self.pool_new_committed_words(slot_hidden, new_commit_mask, word_ids=word_ids)
        final_state = self.update_sentence_state(new_words, new_word_mask, prev_state)
        next_state = final_state.detach() if detach_next_state else final_state
        sentence_state = final_state[-1]

        confidence_logit = self.confidence_head(slot_hidden)
        abstention_logit = self.abstention_head(slot_hidden)
        confidence = torch.sigmoid(confidence_logit)
        abstention_probability = torch.sigmoid(abstention_logit)
        if self.utt_pooling_head == 'gru_visible':
            committed_pool = masked_mean(slot_hidden, cumulative_commit_mask, dim=1)
            visible_pool = masked_mean(slot_hidden, valid_mask, dim=1)
            mean_confidence = masked_mean(confidence, valid_mask, dim=1)
            mean_abstention = masked_mean(abstention_probability, valid_mask, dim=1)
            utt_input = torch.cat([sentence_state, committed_pool, visible_pool, mean_confidence, mean_abstention], dim=-1)
            utt_scores = self.utt_visible_head(utt_input)
        else:
            utt_scores = self.utt_head(sentence_state)
        base_word_scores = self.word_head(slot_hidden)
        if self.stress_head is None:
            word_scores = base_word_scores
        else:
            if self.stress_branch == 'detached':
                stress_input = slot_hidden.detach()
            else:
                scale = float(self.stress_grad_scale)
                stress_input = scale * slot_hidden + (1.0 - scale) * slot_hidden.detach()
            stress_score = self.stress_head(stress_input)
            word_scores = torch.cat([base_word_scores[..., 0:1], stress_score, base_word_scores[..., 2:3]], dim=-1)

        output = {
            'phone_score': self.phone_head(slot_hidden),
            'word_scores': word_scores,
            'utt_scores': utt_scores,
            'asr_correct_logits': self.asr_correct_head(slot_hidden),
            'uncertainty_logits': self.uncertainty_head(slot_hidden),
            'confidence_logit': confidence_logit,
            'confidence': confidence,
            'abstention_logit': abstention_logit,
            'abstention_probability': abstention_probability,
            'reliability_gate': gate,
            'slot_hidden': slot_hidden,
            'new_word_states': new_words,
            'new_word_mask': new_word_mask,
            'word_states': new_words,
            'word_mask': new_word_mask,
            'sentence_state': sentence_state,
            'next_state': next_state,
            'cumulative_commit_mask': cumulative_commit_mask,
            'new_commit_mask': new_commit_mask,
        }
        if self.state_projection is not None:
            output['state_projection'] = self.state_projection(sentence_state)
        return output

    @torch.inference_mode()
    def stream_step(self, batch, prev_state=None):
        return self.forward(
            cn_post=batch['cn_post'],
            cn_stats=batch['cn_stats'],
            acoustic_post=batch['acoustic_post'],
            acoustic_stats=batch['acoustic_stats'],
            prosody=batch['prosody'],
            visible_len=batch['visible_len'],
            cumulative_commit_mask=batch.get('cumulative_commit_mask', batch.get('commit_mask')),
            new_commit_mask=batch.get('new_commit_mask'),
            word_ids=batch.get('pcn_word_id'),
            slot_prosody=batch.get('slot_prosody'),
            prev_state=prev_state,
            detach_next_state=True,
        )
