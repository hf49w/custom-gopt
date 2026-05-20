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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), platform.node(), time.asctime()))


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with train_prefix.jsonl / test_prefix.jsonl / metadata.json.')
    parser.add_argument('--exp-dir', type=str, default='./exp_streaming_whisper')
    parser.add_argument('--model-name-or-path', type=str, default='openai/whisper-base')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR')
    parser.add_argument('--n-epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max-target-tokens', type=int, default=96)
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


def evaluate(model, loader, processor, device, max_target_tokens):
    model.eval()
    total_loss = 0.0
    total_items = 0
    references = []
    hypotheses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval', leave=False):
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_features=input_features, labels=labels)
            batch_size = input_features.shape[0]
            total_loss += float(output.loss.item()) * batch_size
            total_items += batch_size

            generated = model.generate(
                input_features=input_features,
                max_new_tokens=max_target_tokens,
            )
            pred_text = processor.batch_decode(generated, skip_special_tokens=True)
            references.extend(batch['target_text'])
            hypotheses.extend(pred_text)

    wer = word_error_rate(references, hypotheses)
    return total_loss / max(total_items, 1), wer, references, hypotheses


def save_last_model(model, processor, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def save_checkpoint(exp_dir, model, optimizer, scheduler, history, epoch, global_step, best_wer):
    checkpoint = {
        'model_state': model.state_dict(),
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
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    history = checkpoint.get('history', [])
    epoch = int(checkpoint.get('epoch', 0))
    global_step = int(checkpoint.get('global_step', 0))
    best_wer = float(checkpoint.get('best_wer', 1e9))
    return epoch, global_step, best_wer, history


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language=args.language, task='transcribe')
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.to(device)

    train_dataset = PrefixDataset(data_dir / 'train_prefix.jsonl', args.sample_rate)
    test_dataset = PrefixDataset(data_dir / 'test_prefix.jsonl', args.sample_rate)
    collator = WhisperPrefixCollator(processor, args.language)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

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

        eval_loss, eval_wer, references, hypotheses = evaluate(
            model=model,
            loader=test_loader,
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
        })
        (exp_dir / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
        save_last_model(model, processor, exp_dir / 'last_model')
        save_checkpoint(exp_dir, model, optimizer, scheduler, history, epoch + 1, global_step, best_wer)

        if eval_wer < best_wer:
            best_wer = eval_wer
            best_dir = exp_dir / 'best_model'
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
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


if __name__ == '__main__':
    main()
