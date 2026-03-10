import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pyannote").setLevel(logging.ERROR)

import os
import warnings
import torch
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline

warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# --- 2. Maximize RTX 5080 Speed ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Configuration ---
HF_TOKEN = "hf_XXXXXXXXXXXXXX"    #----------> Replace with you hugging face api key.
AUDIO_FILE = r"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"    #-----> Replace with your audio file path.


def run_diarization():
    print("--- System & GPU Check ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Loading Pyannote Pipeline ---")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        pipeline.to(device)
        print("SUCCESS: Pipeline loaded and moved to GPU.")
    except Exception as e:
        print(f"ERROR loading pipeline: {e}")
        return

    print(f"\n--- Processing Audio: {AUDIO_FILE} ---")
    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: Could not find '{AUDIO_FILE}'.")
        return

    try:
        audio_data, sample_rate = sf.read(AUDIO_FILE, dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = np.expand_dims(audio_data, axis=0)
        waveform = torch.from_numpy(audio_data)

        audio_in_memory = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

    except Exception as e:
        print(f"ERROR loading audio file: {e}")
        return

    print("Running diarization inference on the RTX 5080... (Please wait)")

    # 3. API FIX: Get the raw output from Pyannote 4.x
    diarize_output = pipeline(audio_in_memory)

    # Extract the actual Annotation object from the wrapper
    annotation = diarize_output.speaker_diarization

    print("\n--- Diarization Results ---")
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        print(f"[{turn.start:05.1f}s - {turn.end:05.1f}s] {speaker}")


if __name__ == "__main__":
    run_diarization()
