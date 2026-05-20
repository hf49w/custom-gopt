import argparse
from pathlib import Path

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def get_args():
    parser = argparse.ArgumentParser(description='Download/save a Whisper checkpoint into a plain local directory for server upload.')
    parser.add_argument('--model-name-or-path', type=str, default='openai/whisper-base')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--language', type=str, default='english')
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    main()
