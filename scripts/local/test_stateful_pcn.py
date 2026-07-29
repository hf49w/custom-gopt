import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'prep_data'))

from models import PCNStreamingScorer
from build_streaming_pcn_gopt_data import build_pcn_from_hypotheses
from train_streaming_pcn import PCNUtteranceDataset, effective_supervision_weight, make_loader, reset_state_where_needed


def make_chunk(phone_dim=5, seq_len=4):
    torch.manual_seed(7)
    cn_post = torch.softmax(torch.randn(1, seq_len, phone_dim), dim=-1)
    acoustic_post = torch.softmax(torch.randn(1, seq_len, phone_dim), dim=-1)
    cn_stats = torch.rand(1, seq_len, 5)
    acoustic_stats = torch.rand(1, seq_len, 4)
    prosody = torch.randn(1, 14)
    word_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    return cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, word_ids


def test_gru_new_commits_only():
    phone_dim = 5
    seq_len = 4
    model = PCNStreamingScorer(phone_dim=phone_dim, seq_len=seq_len, embed_dim=16, gru_dim=12, depth=1, num_heads=1)
    model.eval()
    tensors = make_chunk(phone_dim, seq_len)
    cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, word_ids = tensors

    out_a = model(
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len=torch.tensor([1]),
        cumulative_commit_mask=torch.tensor([[1, 0, 0, 0]], dtype=torch.float32),
        new_commit_mask=torch.tensor([[1, 0, 0, 0]], dtype=torch.float32),
        word_ids=word_ids,
    )
    out_b = model(
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len=torch.tensor([2]),
        cumulative_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        new_commit_mask=torch.tensor([[0, 1, 0, 0]], dtype=torch.float32),
        word_ids=word_ids,
        prev_state=out_a['next_state'],
    )
    out_ab = model(
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len=torch.tensor([2]),
        cumulative_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        new_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        word_ids=word_ids,
    )
    assert torch.allclose(out_b['next_state'], out_ab['next_state'], atol=1e-5), 'A then B must equal one-shot [A,B]'

    out_repeat = model(
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len=torch.tensor([2]),
        cumulative_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        new_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        word_ids=word_ids,
        prev_state=out_a['next_state'],
    )
    assert not torch.allclose(out_repeat['next_state'], out_ab['next_state'], atol=1e-5), 'prev_state plus cumulative mask would repeat A'


def test_state_reset():
    state = torch.ones(1, 2, 3)
    reset = torch.tensor([1, 0])
    out = reset_state_where_needed(state, reset)
    assert torch.all(out[:, 0] == 0), 'first utterance state should reset'
    assert torch.all(out[:, 1] == 1), 'second utterance state should be preserved'


def test_gru_visible_forward_shape():
    phone_dim = 5
    seq_len = 4
    model = PCNStreamingScorer(
        phone_dim=phone_dim,
        seq_len=seq_len,
        embed_dim=16,
        gru_dim=12,
        depth=1,
        num_heads=1,
        utt_pooling_head='gru_visible',
    )
    model.eval()
    cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, word_ids = make_chunk(phone_dim, seq_len)
    out = model(
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len=torch.tensor([seq_len]),
        cumulative_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        new_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        word_ids=word_ids,
    )
    assert out['utt_scores'].shape == (1, 5), 'gru_visible should keep utterance score shape'


def test_concat_vector_gate_forward_shape():
    phone_dim = 5
    seq_len = 4
    model = PCNStreamingScorer(
        phone_dim=phone_dim,
        seq_len=seq_len,
        embed_dim=16,
        gru_dim=12,
        depth=1,
        num_heads=1,
        fusion_mode='concat_vector_gate',
    )
    model.eval()
    cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, word_ids = make_chunk(phone_dim, seq_len)
    out = model(
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len=torch.tensor([seq_len]),
        cumulative_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        new_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        word_ids=word_ids,
    )
    assert out['phone_score'].shape == (1, seq_len, 1), 'concat_vector_gate should keep phone score shape'


def test_detached_stress_branch_forward_shape():
    phone_dim = 5
    seq_len = 4
    model = PCNStreamingScorer(
        phone_dim=phone_dim,
        seq_len=seq_len,
        embed_dim=16,
        gru_dim=12,
        depth=1,
        num_heads=1,
        stress_branch='detached',
    )
    model.eval()
    cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, word_ids = make_chunk(phone_dim, seq_len)
    out = model(
        cn_post,
        cn_stats,
        acoustic_post,
        acoustic_stats,
        prosody,
        visible_len=torch.tensor([seq_len]),
        cumulative_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        new_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        word_ids=word_ids,
    )
    assert out['word_scores'].shape == (1, seq_len, 3), 'detached stress branch should keep word score shape'


def test_task5_ablation_forward_shapes():
    phone_dim = 5
    seq_len = 4
    configs = [
        {'fusion_mode': 'fixed_average'},
        {'use_acoustic': False},
        {'use_prosody': False},
        {'use_uncertainty_stats': False},
        {'utt_pooling_head': 'visible_committed'},
    ]
    cn_post, cn_stats, acoustic_post, acoustic_stats, prosody, word_ids = make_chunk(phone_dim, seq_len)
    for kwargs in configs:
        model = PCNStreamingScorer(
            phone_dim=phone_dim,
            seq_len=seq_len,
            embed_dim=16,
            gru_dim=12,
            depth=1,
            num_heads=1,
            **kwargs,
        )
        model.eval()
        out = model(
            cn_post,
            cn_stats,
            acoustic_post,
            acoustic_stats,
            prosody,
            visible_len=torch.tensor([seq_len]),
            cumulative_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
            new_commit_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
            word_ids=word_ids,
        )
        assert out['phone_score'].shape == (1, seq_len, 1), f'{kwargs} should keep phone score shape'
        assert out['word_scores'].shape == (1, seq_len, 3), f'{kwargs} should keep word score shape'
        assert out['utt_scores'].shape == (1, 5), f'{kwargs} should keep utterance score shape'


def test_relaxed_supervision_weight():
    cn_post = torch.zeros(1, 3, 5)
    acoustic_post = torch.zeros(1, 3, 5)
    cn_post[0, 0, 2] = 0.4
    acoustic_post[0, 0, 2] = 0.5
    chunk = {
        'cn_post': cn_post,
        'acoustic_post': acoustic_post,
        'phone_target': torch.tensor([[[2.0, 3.0], [1.0, 4.0], [-1.0, -1.0]]]),
        'soft_label_weight': torch.zeros(1, 3),
        'cumulative_commit_mask': torch.tensor([[1.0, 1.0, 0.0]]),
        'confidence_loss_mask': torch.zeros(1, 3),
        'asr_correct_target': torch.zeros(1, 3),
    }
    valid_mask = torch.tensor([[1.0, 1.0, 0.0]])
    base_args = {
        'relaxed_min_gt_posterior': 0.05,
        'relaxed_acoustic_scale': 0.3,
        'relaxed_min_weight': 0.05,
        'relaxed_max_weight': 0.5,
    }
    original = effective_supervision_weight(chunk, valid_mask, SimpleNamespace(soft_label_policy='original', **base_args))
    relaxed = effective_supervision_weight(chunk, valid_mask, SimpleNamespace(soft_label_policy='relaxed', **base_args))
    assert original[0, 0].item() == 0.0, 'original policy should preserve zero soft_label_weight'
    assert relaxed[0, 0].item() > 0.0, 'relaxed policy should retain supported non-exact GT phone slots'
    assert not torch.allclose(original, relaxed), 'original and relaxed policies should produce different weights here'


def test_utterance_loader_preserves_chunk_order():
    tmp_dir = Path(tempfile.mkdtemp(prefix='toy_pcn_stateful_'))
    try:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / 'scripts' / 'local' / 'make_toy_pcn_data.py'),
                '--output-dir',
                str(tmp_dir),
                '--overwrite',
                '--num-train',
                '2',
                '--num-val',
                '1',
                '--num-test',
                '1',
                '--chunks-per-utt',
                '4',
            ],
            check=True,
        )
        dataset = PCNUtteranceDataset('train', tmp_dir)
        loader = make_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        chunk_ids = batch['chunk_id'][0, : int(batch['chunk_valid_mask'][0].sum().item())].tolist()
        assert chunk_ids == sorted(chunk_ids), f'chunk order was not preserved: {chunk_ids}'
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_local_confidence_changes_pcn_posterior():
    phn_dict = {'AA': 0}
    high = {
        'words': ['a'],
        'logprob': 0.0,
        'length_normalized_sequence_score': 0.0,
        'phones': ['AA'],
        'phone_ids': [0],
        'phone_to_word': [0],
        'phone_confidences': [0.95],
        'phone_acoustic_supports': [0.95],
        'phone_times': [(0.0, 0.2)],
        'source_counts': {'toy': 1},
    }
    low = dict(high)
    low['phone_confidences'] = [0.10]
    low['phone_acoustic_supports'] = [0.10]
    pcn_high = build_pcn_from_hypotheses([high], phone_mapper=None, phn_dict=phn_dict)
    pcn_low = build_pcn_from_hypotheses([low], phone_mapper=None, phn_dict=phn_dict)
    assert pcn_high['cn_post'][0, 0] > pcn_low['cn_post'][0, 0], 'higher local confidence should increase phone posterior'
    assert pcn_high['cn_post'][0, 1] < pcn_low['cn_post'][0, 1], 'lower local confidence should increase epsilon posterior'


def main():
    test_gru_new_commits_only()
    test_state_reset()
    test_gru_visible_forward_shape()
    test_concat_vector_gate_forward_shape()
    test_detached_stress_branch_forward_shape()
    test_relaxed_supervision_weight()
    test_utterance_loader_preserves_chunk_order()
    test_local_confidence_changes_pcn_posterior()
    print('stateful_pcn_tests_ok')


if __name__ == '__main__':
    main()
