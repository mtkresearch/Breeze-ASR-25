import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutomaticSpeechRecognitionPipeline
import argparse

def transcribe_audio(file_name):
    # 1. Load audio
    waveform, sample_rate = torchaudio.load(file_name)          

    # 2. Preprocess
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)                         
    waveform = waveform.squeeze().numpy()                        

    if sample_rate != 16_000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16_000)
        waveform = resampler(torch.tensor(waveform)).numpy()
        sample_rate = 16_000

    # 3. Load Model
    processor = WhisperProcessor.from_pretrained("MediaTek-Research/Twister")
    model = WhisperForConditionalGeneration.from_pretrained("MediaTek-Research/Twister").to("cuda").eval()

    # 4. Build Pipeline
    asr_pipeline = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=0
    )

    # 5. Inference
    output = asr_pipeline(waveform, return_timestamps=True)  
    print("Result:", output["text"])

# Set up command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument('--file_name', type=str, required=True, help="Path to the input audio file")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments from the command line
    args = parse_args()
    
    # Call the transcription function with the provided file_name
    transcribe_audio(args.file_name)
