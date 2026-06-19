import argparse
import json
import os
import platform
import re
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), platform.node(), time.asctime()))


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with train_prefix.jsonl / val_prefix.jsonl / test_prefix.jsonl / metadata.json.')
    parser.add_argument('--exp-dir', type=str, default='./exp_streaming_whisper')
    parser.add_argument('--model-name-or-path', type=str, default='openai/whisper-base')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR')
    parser.add_argument('--n-epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max-target-tokens', type=int, default=96)
    parser.add_argument('--eval-generate-max-samples', type=int, default=256, help='Maximum number of final-chunk validation samples used for autoregressive generation/WER.')
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for faster training/inference at the cost of more memory.')
    parser.add_argument('--tf32', action='store_true', help='Enable TF32 matmul/cudnn on Ampere+ GPUs.')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--resume', action='store_true', help='Resume from exp-dir/last_checkpoint.pt if it exists.')
    return parser.parse_args()


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def word_error_rate(references, hypotheses):
    total_words = 0
    total_edits = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = normalize_text(ref).split()
        hyp_words = normalize_text(hyp).split()
        total_words += max(len(ref_words), 1)

        dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            dp[i][0] = i
        for j in range(len(hyp_words) + 1):
            dp[0][j] = j
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        total_edits += dp[-1][-1]
    return float(total_edits) / float(max(total_words, 1))


class PrefixDataset(Dataset):
    def __init__(self, manifest_path, sample_rate):
        self.rows = []
        self.sample_rate = sample_rate
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio_end = max(float(row['audio_end']), 1e-4)
        wav, _ = librosa.load(row['audio_path'], sr=self.sample_rate, mono=True)
        wav = wav[: int(audio_end * self.sample_rate)]
        return {
            'audio': wav.astype(np.float32),
            'target_text': row['target_text'],
            'visible_text': row['visible_text'],
            'utt_id': row['utt_id'],
            'chunk_id': row['chunk_id'],
            'is_final': bool(row['is_final']),
        }


def build_final_subset(dataset, max_samples):
    final_indices = [idx for idx, row in enumerate(dataset.rows) if bool(row.get('is_final', False))]
    if max_samples > 0:
        final_indices = final_indices[:max_samples]
    return Subset(dataset, final_indices)


class WhisperPrefixCollator:
    def __init__(self, processor, language):
        self.processor = processor
        self.language = language

    def __call__(self, batch):
        audio = [item['audio'] for item in batch]
        target_text = [item['target_text'] for item in batch]
        visible_text = [item['visible_text'] for item in batch]
        utt_ids = [item['utt_id'] for item in batch]
        chunk_ids = [item['chunk_id'] for item in batch]
        is_final = [item['is_final'] for item in batch]

        model_inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors='pt',
        )

        label_batch = self.processor.tokenizer(
            target_text,
            padding=True,
            return_tensors='pt',
        )
        labels = label_batch['input_ids']
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            'input_features': model_inputs['input_features'],
            'labels': labels,
            'target_text': target_text,
            'visible_text': visible_text,
            'utt_id': utt_ids,
            'chunk_id': chunk_ids,
            'is_final': is_final,
        }


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval-loss', leave=False):
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_features=input_features, labels=labels)
            batch_size = input_features.shape[0]
            total_loss += float(output.loss.item()) * batch_size
            total_items += batch_size
    return total_loss / max(total_items, 1)


def evaluate_generation(model, loader, processor, device, max_target_tokens):
    model.eval()
    references = []
    hypotheses = []
    gen_model = unwrap_model(model)
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval-gen', leave=False):
            input_features = batch['input_features'].to(device)
            generated = gen_model.generate(
                input_features=input_features,
                max_new_tokens=max_target_tokens,
            )
            pred_text = processor.batch_decode(generated, skip_special_tokens=True)
            references.extend(batch['target_text'])
            hypotheses.extend(pred_text)

    if not references:
        return 1.0, references, hypotheses
    wer = word_error_rate(references, hypotheses)
    return wer, references, hypotheses


def unwrap_model(model):
    model = getattr(model, '_orig_mod', model)
    while isinstance(model, nn.DataParallel):
        model = model.module
        model = getattr(model, '_orig_mod', model)
    return model


def save_last_model(model, processor, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def save_checkpoint(exp_dir, model, optimizer, scheduler, history, epoch, global_step, best_wer):
    checkpoint = {
        'model_state': unwrap_model(model).state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'history': history,
        'epoch': int(epoch),
        'global_step': int(global_step),
        'best_wer': float(best_wer),
    }
    torch.save(checkpoint, exp_dir / 'last_checkpoint.pt')


def load_checkpoint(exp_dir, model, optimizer, scheduler, device):
    checkpoint_path = exp_dir / 'last_checkpoint.pt'
    if not checkpoint_path.exists():
        return 0, 0, 1e9, []

    checkpoint = torch.load(checkpoint_path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    history = checkpoint.get('history', [])
    epoch = int(checkpoint.get('epoch', 0))
    global_step = int(checkpoint.get('global_step', 0))
    best_wer = float(checkpoint.get('best_wer', 1e9))
    return epoch, global_step, best_wer, history


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn, prefetch_factor):
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
    }
    if torch.cuda.is_available():
        kwargs['pin_memory'] = True
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = max(2, int(prefetch_factor))
    return DataLoader(**kwargs)


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.tf32 and device.type == 'cuda':
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language=args.language, task='transcribe')
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.to(device)
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)
    elif args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    train_dataset = PrefixDataset(data_dir / 'train_prefix.jsonl', args.sample_rate)
    val_dataset = PrefixDataset(data_dir / 'val_prefix.jsonl', args.sample_rate)
    test_dataset = PrefixDataset(data_dir / 'test_prefix.jsonl', args.sample_rate)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(
            f'Empty prefix dataset: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}. '
            f'Inspect {data_dir / "metadata.json"} and the train/val/test *_prefix.jsonl files.'
        )
    collator = WhisperPrefixCollator(processor, args.language)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, collator, args.prefetch_factor)
    val_loader = make_loader(val_dataset, args.eval_batch_size, False, args.num_workers, collator, args.prefetch_factor)
    test_loader = make_loader(test_dataset, args.eval_batch_size, False, args.num_workers, collator, args.prefetch_factor)
    val_final_dataset = build_final_subset(val_dataset, args.eval_generate_max_samples)
    test_final_dataset = build_final_subset(test_dataset, args.eval_generate_max_samples)
    if len(val_final_dataset) == 0:
        raise ValueError(
            'No final-chunk validation samples were found for generation eval. '
            f'Inspect {data_dir / "val_prefix.jsonl"} and ensure is_final rows exist.'
        )
    val_final_loader = make_loader(val_final_dataset, args.eval_batch_size, False, args.num_workers, collator, args.prefetch_factor)
    test_final_loader = make_loader(test_final_dataset, args.eval_batch_size, False, args.num_workers, collator, args.prefetch_factor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(1.0 / max(args.warmup_steps, 1), 1e-3),
        total_iters=max(args.warmup_steps, 1),
    )

    best_wer = 1e9
    history = []
    global_step = 0
    start_epoch = 0

    config = {
        'data_dir': str(data_dir),
        'args': vars(args),
    }
    (exp_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.resume:
        start_epoch, global_step, best_wer, history = load_checkpoint(exp_dir, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        running_loss = 0.0
        seen_items = 0
        train_bar = tqdm(train_loader, desc=f'train-{epoch}', leave=False)
        for batch in train_bar:
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_features=input_features, labels=labels)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if global_step < args.warmup_steps:
                scheduler.step()

            batch_size = input_features.shape[0]
            running_loss += float(loss.item()) * batch_size
            seen_items += batch_size
            global_step += 1
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        eval_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
        )
        eval_wer, references, hypotheses = evaluate_generation(
            model=model,
            loader=val_final_loader,
            processor=processor,
            device=device,
            max_target_tokens=args.max_target_tokens,
        )
        train_loss = running_loss / max(seen_items, 1)
        history.append({
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'eval_loss': float(eval_loss),
            'eval_wer': float(eval_wer),
            'eval_generate_samples': int(len(val_final_dataset)),
            'eval_loss_samples': int(len(val_dataset)),
        })
        (exp_dir / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
        save_last_model(model, processor, exp_dir / 'last_model')
        save_checkpoint(exp_dir, model, optimizer, scheduler, history, epoch + 1, global_step, best_wer)

        if eval_wer < best_wer:
            best_wer = eval_wer
            best_dir = exp_dir / 'best_model'
            best_dir.mkdir(parents=True, exist_ok=True)
            unwrap_model(model).save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            sample_preview = [
                {
                    'reference': references[idx],
                    'prediction': hypotheses[idx],
                }
                for idx in range(min(20, len(references)))
            ]
            (best_dir / 'eval_preview.json').write_text(json.dumps(sample_preview, ensure_ascii=False, indent=2), encoding='utf-8')
            save_checkpoint(exp_dir, model, optimizer, scheduler, history, epoch + 1, global_step, best_wer)

    best_dir = exp_dir / 'best_model'
    if best_dir.exists():
        test_model = WhisperForConditionalGeneration.from_pretrained(best_dir).to(device)
        if args.compile and hasattr(torch, 'compile'):
            test_model = torch.compile(test_model)
    else:
        test_model = model

    test_loss = evaluate_loss(
        model=test_model,
        loader=test_loader,
        device=device,
    )
    test_wer, test_references, test_hypotheses = evaluate_generation(
        model=test_model,
        loader=test_final_loader,
        processor=processor,
        device=device,
        max_target_tokens=args.max_target_tokens,
    )
    test_summary = {
        'test_loss': float(test_loss),
        'test_wer': float(test_wer),
        'test_generate_samples': int(len(test_final_dataset)),
        'test_loss_samples': int(len(test_dataset)),
    }
    (exp_dir / 'test_metrics.json').write_text(json.dumps(test_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    test_preview = [
        {
            'reference': test_references[idx],
            'prediction': test_hypotheses[idx],
        }
        for idx in range(min(20, len(test_references)))
    ]
    (exp_dir / 'test_preview.json').write_text(json.dumps(test_preview, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
