import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from IPython.display import Audio, display
import os

from hyperparameters import hp
from model import TransformerTTS
from melspecs import Melspecs


def synthesize(text, checkpoint_path=None, save_name='tts_model'):
    device = torch.device(hp.device)

    if checkpoint_path is None:
        checkpoint_path = f'{hp.output_dir}/checkpoints/{save_name}/best.pt'

    if not os.path.exists(checkpoint_path):
        legacy_path = f'{hp.output_dir}/checkpoints/best_model.pt'
        if os.path.exists(legacy_path):
            checkpoint_path = legacy_path
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    model = TransformerTTS(hp).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Enforce lowercase for inference text
    text = str(text).lower()
    print(f'Text: "{text}"')
    seq = torch.LongTensor([hp.text_to_sequence(text)]).to(device)

    with torch.no_grad():
        mel = model.inference(seq)

    print("Generating audio...")
    melspecs = Melspecs(hp).to(device)
    mel_for_audio = mel.squeeze(0).T.unsqueeze(0)  # [1, n_mels, T]
    audio = melspecs.mel_to_wav(mel_for_audio).cpu()

    return audio.squeeze().numpy(), mel.squeeze().cpu().numpy()


def save_audio(audio, text, output_name=None):
    if output_name is None:
        clean_text = ''.join(c for c in text[:30] if c.isalnum() or c.isspace())
        output_name = clean_text.replace(' ', '_')

    output_path = f'{hp.output_dir}/audio/{output_name}.wav'
    torchaudio.save(output_path, torch.tensor(audio).unsqueeze(0), hp.sample_rate)
    print(f"Saved audio to {output_path}")
    return output_path


def visualize_mel(mel, text, save=True):
    plt.figure(figsize=(12, 4))
    plt.imshow(mel.T, aspect='auto', origin='lower', cmap='viridis')
    ttl = text[:50] + ("..." if len(text) > 50 else "")
    plt.title(f'Generated Mel-Spectrogram: "{ttl}"')
    plt.xlabel('Time Step')
    plt.ylabel('Mel Channel')
    plt.colorbar(label='Magnitude (dB)')
    plt.tight_layout()

    if save:
        plot_path = f'{hp.output_dir}/plots/mel_generated.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")

    plt.show()


def tts_demo(texts=None):
    if texts is None:
        texts = [
            "hello world",
            "Astros are going to WIN",
            "hi",
            "%($*#argument8923!",
        ]

    print("\nTTS Demo\n" + "="*50)

    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] processing...")
        audio, mel = synthesize(text)
        audio_path = save_audio(audio, text, f"demo_{i}")

        if hasattr(__builtins__, '__IPYTHON__'):
            display(Audio(audio, rate=hp.sample_rate))

        visualize_mel(mel, text)

        duration = len(audio) / hp.sample_rate
        print(f"Duration: {duration:.2f} seconds")
        print(f"Mel shape: {mel.shape}")
        print("-" * 50)

    print("\nDemo complete.")


def interactive_tts():
    print("\nInteractive TTS")
    print("Enter text to synthesize (or 'quit' to exit):\n")

    while True:
        text = input("Text: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye.")
            break
        if not text:
            print("Please enter some text.")
            continue

        try:
            audio, mel = synthesize(text)
            save_audio(audio, text)
            if hasattr(__builtins__, '__IPYTHON__'):
                display(Audio(audio, rate=hp.sample_rate))
            visualize_mel(mel, text, save=False)
            print(f"Generated {len(audio)/hp.sample_rate:.2f} seconds of speech\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TTS Inference')
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--demo', action='store_true', help='Run demo with examples')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    if args.demo:
        tts_demo()
    elif args.interactive:
        interactive_tts()
    elif args.text:
        audio, mel = synthesize(args.text, args.checkpoint)
        save_audio(audio, args.text)
        visualize_mel(mel, args.text)
        print(f"Generated {len(audio)/hp.sample_rate:.2f} seconds of speech")
    else:
        print("Usage: python inference.py --text 'your text here'")
        print("       python inference.py --demo")
        print("       python inference.py --interactive")
