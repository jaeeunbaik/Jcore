import torch
import torchaudio
from src.models.student_model import StudentModel
from src.utils.audio_processing import preprocess_audio
from src.utils.metrics import calculate_wer
import yaml

def load_model(model_path):
    model = StudentModel.load_from_checkpoint(model_path)
    model.eval()
    return model

def infer(model, audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    processed_audio = preprocess_audio(waveform, sample_rate)
    with torch.no_grad():
        predictions = model(processed_audio)
    return predictions

def main(config_path, audio_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model = load_model(config['model']['checkpoint_path'])
    predictions = infer(model, audio_path)

    # Assuming predictions are in a format that can be converted to text
    predicted_text = predictions.argmax(dim=-1)  # Example for getting predicted indices
    print("Predicted Text:", predicted_text)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file for inference")
    args = parser.parse_args()

    main(args.config, args.audio)