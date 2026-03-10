# Speaker Diarization

Speaker Diarization is a technology that identifies and separates different speakers in an audio recording. The main objective of this project is to analyze an audio file and determine who spoke and when they spoke during a conversation. 

## 🧠 Theory & Architecture

This script implements a high-performance Speaker Diarization pipeline using state-of-the-art deep learning models. It breaks down an audio stream into distinct segments and clusters them based on speaker identity (e.g., `SPEAKER_00`, `SPEAKER_01`), without necessarily transcribing the actual words spoken.

### Core Concepts & Workflow

**1. The Pyannote Pipeline**
The script leverages `pyannote/speaker-diarization-3.1`, a leading open-source model hosted on Hugging Face. Under the hood, this pipeline performs several sequential tasks:
* **Voice Activity Detection (VAD):** Identifies segments of the audio that contain human speech, filtering out silence and background noise.
* **Speaker Embedding:** Extracts acoustic features from the speech segments, creating a unique "voiceprint" for each speaker.
* **Clustering:** Groups the extracted embeddings to determine the total number of speakers and assigns a specific speaker label to each time segment.

**2. Hardware Acceleration & TF32 Optimization**
To maximize throughput on modern GPUs (like the NVIDIA RTX 5000 series), the script explicitly enables **TensorFloat-32 (TF32)** for PyTorch and cuDNN matrix multiplications. This math mode provides the dynamic range of standard 32-bit floats but the precision of 16-bit floats, drastically accelerating deep learning inference on Tensor Cores with zero noticeable degradation in diarization accuracy.

**3. Audio Preprocessing Strategy**
Deep learning audio models require standardized inputs to function correctly. The script handles this automatically:
* **Stereo-to-Mono Downmixing:** If the input audio file contains multiple channels (stereo), the script averages them along the channel axis. This collapses the audio into a single mono channel, which is required by Pyannote.
* **Tensor Formatting:** The audio is reshaped and converted into a PyTorch tensor in memory, allowing for direct, zero-latency transfer to the GPU for processing.

**4. API Compatibility & Warning Management**
The script includes specific handling for newer Pyannote 4.x ecosystem wrappers. It extracts the base `Annotation` object to safely parse the generated tracks, yielding precise start times, end times, and speaker labels. Additionally, it intelligently filters out bleeding-edge ecosystem noise (like harmless PyTorch pooling or internal TF32 warnings) to keep console outputs clean and production-ready.

---

## ⚡ Hardware Execution: GPU vs. CPU (Boons & Banes)

Because audio models are computationally heavy, the hardware you choose significantly impacts performance. The script automatically detects your hardware and routes the workload accordingly.

### Running WITH a GPU (NVIDIA CUDA)
*Ideal for production, large datasets, or real-time applications.*

**Boons (Pros):**
* **Blazing Fast Speeds:** Can process audio significantly faster than real-time (e.g., processing a 10-minute audio file in seconds).
* **TF32/Mixed Precision Support:** Leverages Tensor Cores (on RTX cards) to maximize throughput without sacrificing accuracy.
* **Unblocks the CPU:** Offloads the heavy matrix math, leaving your CPU free to handle API routing, database writes, or other concurrent tasks.

**Banes (Cons):**
* **Hardware Dependency:** Requires an NVIDIA GPU; AMD or Intel GPUs are generally not supported out-of-the-box for this PyTorch workflow.
* **Complex Setup:** Requires strict version matching between NVIDIA drivers, CUDA Toolkit, cuDNN, and PyTorch.
* **High VRAM Usage:** Diarization models consume a significant amount of Video RAM, which can be a bottleneck if you are running other models (like Whisper or LLMs) on the same GPU simultaneously.

### Running WITHOUT a GPU (CPU Only)
*Ideal for testing, development, and lightweight local scripts.*

**Boons (Pros):**
* **Highly Accessible:** Will run on practically any machine (Windows, Mac, Linux) without needing expensive specialized hardware.
* **Simple Environment Setup:** Avoids the headache of installing CUDA toolkits—just run `pip install` and the code works.
* **No VRAM Limits:** Relies on system RAM, meaning you can process massive audio files as long as you have enough standard memory.

**Banes (Cons):**
* **Extremely Slow:** Processing times can easily take 2x to 5x the actual length of the audio file (e.g., a 10-minute file might take 30+ minutes to process).
* **System Bottlenecking:** Will pin your CPU usage to 100%, making the rest of your computer sluggish while the script runs.
* **Not Production-Ready:** The high latency makes CPU execution unusable for real-time calling agents or high-volume asynchronous APIs.

---

## 🛠️ Setup & Installation

**1. Install Dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pyannote.audio soundfile numpy
```

**2. Run the Script**
Simply update the `AUDIO_FILE` path variable in the script and execute:
```bash
python diarization.py
```

## 🤗 Hugging Face Integration

This project relies on models hosted on the **Hugging Face Hub**. The `pyannote/speaker-diarization-3.1` model is a "gated" model, meaning you cannot download it anonymously. You must explicitly agree to the creator's terms of service before the code will run.

### Authentication Steps:
To run this code successfully, follow these steps:

1. **Create an Account:** Go to [huggingface.co](https://huggingface.co/) and create a free account.
2. **Accept the Terms:** You must visit **both** of the following links while logged in and click the "Agree and access repository" button:
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. **Generate a Token:** Go to your Hugging Face Account Settings -> **Access Tokens** and create a new token (Read permissions are sufficient).
4. **Update the Code:** Copy that token and paste it into the script as the `HF_TOKEN` variable:
   ```python
   HF_TOKEN = "your_generated_hf_token_here"
   ```
