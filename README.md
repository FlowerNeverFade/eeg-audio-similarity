# EEG-Audio Similarity Analysis Toolkit

This repository contains the code for analyzing the similarity between EEG neural responses and audio language model representations using Representational Similarity Analysis (RSA) and related metrics.

## Overview

We investigate how well large audio language models (Audio LLMs) capture neural representations of speech processing by comparing their internal representations with human EEG signals. Our analysis pipeline includes:

- **EEG Preprocessing**: Filtering, artifact removal, bad channel interpolation
- **Sentence Segmentation**: Segmenting EEG/audio by sentence boundaries using alignment tables
- **Audio LLM Inference**: Extracting layer-wise hidden states from various audio models
- **Time Alignment**: Aligning EEG and audio embeddings to common time axis (Ns→Ta)
- **Representational Similarity Analysis (RSA)**: Comparing representational geometries
- **Centered Kernel Alignment (CKA)**: Measuring similarity between representation spaces
- **Multiple Association Metrics**: Distance correlation, HSIC, RV coefficient, Kendall's Tau
- **Permutation Testing**: Statistical significance testing
- **Prosody Analysis**: Acoustic/prosodic feature extraction and TNC analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/eeg-audio-similarity.git
cd eeg-audio-similarity

# Install dependencies
pip install -r requirements.txt

# For Audio LLM inference (optional)
pip install transformers librosa
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12 (with CUDA support recommended)
- NumPy >= 1.20
- SciPy >= 1.7
- Pandas >= 1.3
- Matplotlib >= 3.4
- (Optional) transformers >= 4.30 for Audio LLM inference
- (Optional) MNE >= 1.0.0 for advanced topography plots
- (Optional) OpenSMILE >= 2.4.0 for acoustic feature extraction

---

## Datasets

This toolkit was validated on two EEG datasets:

### Dataset 1: Alice in Wonderland EEG Dataset

**Source**: University of Michigan Deep Blue Data Repository  
**DOI**: https://doi.org/10.7302/Z29C6VNH  
**Citation**: Brennan, J.R. & Hale, J.T. (2019). Hierarchical structure guides rapid linguistic predictions during naturalistic listening. *PLoS ONE* 14(1): e0207741.

| Property | Value |
|----------|-------|
| Subjects | 49 (33 used in main analysis) |
| Channels | 61 active electrodes (+ 2 reference = 63 total in file) |
| Analysis Channels | 62 (C=62 after re-referencing, excluding EOG) |
| Amplifier | Brain Products actiCHamp |
| Sampling Rate | 500 Hz (0.1-200 Hz band) |
| Stimulus | 12.4 min audiobook (Alice in Wonderland Ch.1) |
| Audio Segments | 12 .wav files |
| Words | 2,130 words across 84 sentences |
| Format | MATLAB (.mat) for FieldTrip toolbox |

> **Note on Channel Numbers:**
> - **61 active electrodes**: Original recording (Brain Products actiCHamp)
> - **62 channels (C=62)**: After average re-referencing and excluding EOG channels (used in paper analysis)
> - The raw `.mat` files contain 61 EEG channels; preprocessing scripts apply re-referencing

#### Dataset Files

| File | Description |
|------|-------------|
| `S01.mat` - `S49.mat` | Raw EEG data (FieldTrip format) |
| `AliceChapterOne-EEG.csv` | Word-level alignment with 16 columns |
| `proc.zip` | Preprocessing parameters (42 subjects) |
| `audio/` | 12 audio segments (.wav) |
| `datasets.mat` | Subject inclusion/exclusion flags |
| `comprehension-scores.txt` | Comprehension quiz scores (8 questions) |

#### Alignment Table Columns (AliceChapterOne-EEG.csv)

| Column | Description |
|--------|-------------|
| `Word` | Word token |
| `Segment` | Audio segment ID (1-12) |
| `onset` / `offset` | Word timing relative to segment start (seconds) |
| `Order` | Word order in full stimulus (1-2130) |
| `Sentence` | Sentence ID (1-84) |
| `Position` | Word position within sentence |
| `LogFreq` | Log word frequency (HAL corpus) |
| `SndPower` | Audio power at word onset |
| `Length` | Word duration (seconds) |
| `IsLexical` | Content word (1) vs function word (0) |
| `NGRAM` / `RNN` / `CFG` | Surprisal values from language models |

#### Loading Dataset 1

```python
import scipy.io
import pandas as pd
from paper_code.preprocessing import segment_eeg_by_sentences

# Load EEG data (FieldTrip format)
def load_alice_eeg(mat_path):
    data = scipy.io.loadmat(mat_path)
    raw = data['raw'][0, 0]
    eeg_data = raw['trial'][0, 0]  # (n_channels, n_samples)
    fs = float(raw['fsample'][0, 0])  # 500 Hz
    labels = [str(lbl[0]) for lbl in raw['label']]
    return eeg_data, fs, labels

# Load alignment table
alignment = pd.read_csv('AliceChapterOne-EEG.csv')

# Example: Load subject S01
eeg, fs, channels = load_alice_eeg('S01.mat')
print(f"EEG shape: {eeg.shape}, fs: {fs} Hz, channels: {len(channels)}")

# Segment by sentence
sentence_segments = {}
for sent_id in alignment['Sentence'].unique():
    sent_words = alignment[alignment['Sentence'] == sent_id]
    start_time = sent_words['onset'].min()
    end_time = sent_words['offset'].max()
    segment_id = sent_words['Segment'].iloc[0]
    
    # Convert to samples (adjust for segment offset)
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    
    sentence_segments[sent_id] = eeg[:, start_sample:end_sample]
```

#### Subject Selection

```python
import scipy.io

# Load dataset flags
datasets = scipy.io.loadmat('datasets.mat')
used_subjects = datasets['use'].flatten()      # N=33 main analysis
low_perf = datasets['lowperf'].flatten()       # N=8 excluded (low quiz score)
high_noise = datasets['highnoise'].flatten()   # N=8 excluded (noisy data)

# Get valid subject IDs
valid_subjects = [f'S{i:02d}' for i, use in enumerate(used_subjects, 1) if use]
print(f"Using {len(valid_subjects)} subjects: {valid_subjects[:5]}...")
```

---

### Dataset 2: OpenNeuro ds004408

| Property | Value |
|----------|-------|
| Channels | 128 |
| Sampling Rate | 512 Hz |
| Format | BIDS (continuous recording) |
| Source | https://openneuro.org/datasets/ds004408 |

#### Preparing ds004408 for Analysis

The ds004408 dataset contains continuous EEG recordings during naturalistic speech listening. To use it with this toolkit:

**Step 1: Download from OpenNeuro**
```bash
# Using datalad
datalad install https://github.com/OpenNeuroDatasets/ds004408.git
cd ds004408
datalad get sub-*/eeg/*

# Or direct download from OpenNeuro website
```

**Step 2: Generate Sentence Alignment Table**

The alignment table maps sentence boundaries to EEG sample indices. We provide a script to generate this from the BIDS events:

```python
from paper_code.preprocessing import generate_alignment_from_bids

# Parse BIDS events.tsv to create alignment table
alignment_df = generate_alignment_from_bids(
    bids_root='/path/to/ds004408',
    subject='01',
    task='listening',
    output_path='alignment_sub01.csv'
)

# Output format:
# sentence_id, start_sample, end_sample, start_time, end_time, audio_file
```

**Step 3: Segment Continuous EEG into Sentences**

```python
from paper_code.preprocessing import (
    load_bids_eeg,
    segment_eeg_by_sentences,
    preprocess_eeg
)

# Load continuous EEG from BIDS
eeg_raw = load_bids_eeg('/path/to/ds004408', subject='01', task='listening')

# Preprocess (bandpass 0.1-40Hz, notch 50Hz)
eeg_clean = preprocess_eeg(eeg_raw, fs=512, lowcut=0.1, highcut=40)

# Segment by sentences using alignment table
sentence_segments = segment_eeg_by_sentences(
    eeg_clean,
    alignment_csv='alignment_sub01.csv',
    fs=512
)
# Returns: Dict[sentence_id, np.ndarray of shape (n_channels, n_samples)]
```

**Step 4: Resample to Match Audio Frame Rate**

```python
from paper_code.preprocessing import resample_to_target_length

# ds004408 uses 512Hz; resample each sentence to match audio frames
for sent_id, eeg_segment in sentence_segments.items():
    # If audio has 100 frames for this sentence
    n_audio_frames = audio_embeddings[sent_id].shape[0]
    eeg_aligned = resample_to_target_length(eeg_segment, n_audio_frames)
```

#### Pre-generated Alignment Tables

We provide pre-generated alignment tables for ds004408 (see `data/alignments/`):

| File | Description |
|------|-------------|
| `ds004408_alignment_all.csv` | Sentence boundaries for all subjects |
| `ds004408_sentence_audio_mapping.json` | Sentence ID → audio file mapping |

> **Note**: Due to licensing, we do not redistribute the EEG data itself. Please download directly from OpenNeuro.

#### Dataset Configuration

```python
# Dataset 1: Alice in Wonderland (62ch after preprocessing, 500Hz)
ALICE_CONFIG = {
    'name': 'Alice in Wonderland EEG',
    'n_channels_raw': 61,      # Active electrodes in original recording
    'n_channels_analysis': 62, # C=62 after re-referencing (paper)
    'fs': 500,
    'n_subjects': 49,
    'n_valid_subjects': 33,
    'n_words': 2130,
    'n_sentences': 84,
    'n_segments': 12,
    'duration_min': 12.4,
    'format': 'fieldtrip_mat',
    'doi': 'https://doi.org/10.7302/Z29C6VNH',
}

# Dataset 2: OpenNeuro ds004408 (128ch, 512Hz)
DS004408_CONFIG = {
    'name': 'OpenNeuro ds004408',
    'n_channels': 128,
    'fs': 512,
    'format': 'bids',
    'source': 'https://openneuro.org/datasets/ds004408',
}
```

---

## Project Structure

```
paper_code/
├── __init__.py
├── README.md
├── requirements.txt
│
├── preprocessing/                    # Data preprocessing
│   ├── __init__.py
│   ├── eeg_preprocessing.py         # bandpass_filter, notch_filter, preprocess_eeg
│   ├── sentence_segmentation.py     # load_alignment_table, segment_eeg_by_sentences,
│   │                                # generate_alignment_from_bids, load_bids_eeg
│   └── time_alignment.py            # align_eeg_to_audio, resample_to_target_length
│
├── models/                           # Audio LLM inference (NEW)
│   ├── __init__.py
│   ├── model_registry.py            # SUPPORTED_MODELS, get_model_config
│   └── audio_llm_inference.py       # load_audio_model, extract_hidden_states
│
├── metrics/                          # Similarity metrics
│   ├── __init__.py
│   ├── rdm.py                       # compute_rdm_vec, compute_rdm_full
│   ├── rsa.py                       # rsa_spearman, rsa_pearson, rsa_between_rdms
│   ├── cka.py                       # compute_cka, compute_cka_linear, compute_cka_rbf
│   ├── distance_correlation.py     # compute_distance_correlation
│   ├── hsic.py                      # compute_hsic_rbf, compute_hsic_linear
│   ├── rv_coefficient.py           # compute_rv_coefficient
│   ├── permutation_test.py         # permutation_pvalue
│   ├── mutual_info.py              # compute_mutual_info_gaussian
│   ├── kendall.py                  # compute_kendall_tau_b
│   └── batch_metrics.py            # compute_rdm_vec_batch, rsa_between_rdms_batch
│
├── features/                         # Feature extraction
│   ├── __init__.py
│   ├── eeg_features.py             # extract_eeg_multichannel_aligned
│   └── audio_features.py           # reduce_audio_dimensions
│
├── visualization/                    # Visualization tools
│   ├── __init__.py
│   ├── rdm_visualization.py        # plot_rdm, plot_rdm_pair
│   ├── topography.py               # plot_topography_simple
│   ├── histogram.py                # plot_histogram, plot_rsa_distribution
│   ├── model_comparison.py         # plot_layerwise_comparison
│   └── group_comparison.py         # plot_group_boxplot
│
├── analysis/                         # Analysis pipelines
│   ├── __init__.py
│   ├── prosody_analysis.py         # extract_prosody_features, compute_tnc
│   ├── opensmile_analysis.py       # add_sentiment_scores
│   └── time_window_analysis.py     # n400_analysis, sliding_window_rsa_analysis
│
└── utils/                            # Utilities
    ├── __init__.py
    ├── gpu_utils.py                # setup_gpu, get_device
    └── data_loading.py             # load_eeg_data, load_audio_embeddings
```

**Total: 35 Python files, 126 functions**

---

## End-to-End Pipeline

### Step 1: EEG Preprocessing

```python
from paper_code.preprocessing import preprocess_eeg, load_alignment_table, segment_eeg_by_sentences
import numpy as np

# Load raw EEG (62 channels after preprocessing, 500Hz sampling rate)
# Note: Original recording has 61 active electrodes; after re-referencing = 62 channels
eeg_raw = np.load('subject_01_raw.npy')  # shape: (62, n_samples)

# Preprocess: bandpass (0.5-40Hz), notch (50Hz), re-reference
eeg_clean, bad_channels = preprocess_eeg(eeg_raw, fs=500, 
                                          low_freq=0.5, high_freq=40,
                                          notch_freq=50)
print(f"Detected {len(bad_channels)} bad channels: {bad_channels}")

# Load sentence alignment table
alignment = load_alignment_table('alignment.csv')
# Expected columns: sentence_id, start_time, end_time, text

# Segment EEG by sentences
eeg_segments = segment_eeg_by_sentences(eeg_clean, alignment, fs=500)
print(f"Extracted {len(eeg_segments)} sentence segments")

# Save segments
for sid, segment in eeg_segments.items():
    np.save(f'eeg_segments/sentence_{sid:03d}.npy', segment)
```

### Step 2: Extract Audio LLM Hidden States

```python
from paper_code.models import load_audio_model, extract_all_layers, save_embeddings
import librosa

# Load audio model (supports: qwen2-audio, salmonn, wavlm-base/large, wav2vec2, hubert, whisper)
model, processor, config = load_audio_model('wavlm-large', device='cuda')
print(f"Loaded {config.model_name}: {config.n_layers} layers, {config.hidden_dim}d")

# Extract embeddings for each sentence
for audio_file in audio_files:
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Extract all layer hidden states
    embeddings = extract_all_layers(model, processor, audio, sample_rate=16000)
    # embeddings = {0: (T, 768), 1: (T, 768), ..., 23: (T, 768)}
    
    # Save to npz
    output_path = f'embeddings/{audio_file.stem}_all_layers.npz'
    save_embeddings(embeddings, output_path, metadata={'model': 'wavlm-large'})
```

### Step 3: Time Alignment (EEG ↔ Audio Embeddings)

```python
from paper_code.preprocessing import align_eeg_to_audio, compute_optimal_lag
from paper_code.models import load_embeddings

# Load EEG segment and audio embeddings
eeg = np.load('eeg_segments/sentence_001.npy')  # (62, n_samples) or (61, n_samples)
audio_emb = load_embeddings('embeddings/sentence_001_all_layers.npz', layer_idx=12)
# audio_emb shape: (T_audio, 1024)

# Align to common time axis (resample EEG to match audio time steps)
eeg_aligned, audio_aligned = align_eeg_to_audio(
    eeg, audio_emb,
    eeg_fs=500,               # EEG sampling rate
    audio_frame_shift=0.02,   # Audio frame shift (20ms)
    method='resample_eeg'     # Resample EEG to match audio
)
print(f"Aligned shapes: EEG {eeg_aligned.shape}, Audio {audio_aligned.shape}")
# Both now have same number of time steps!

# Find optimal lag (EEG typically lags audio by 100-300ms)
best_lag, correlation = compute_optimal_lag(eeg_aligned, audio_aligned, 
                                             max_lag_ms=300, fs=50)
print(f"Optimal lag: {best_lag * 20}ms, correlation: {correlation:.4f}")
```

### Step 4: Compute RSA

```python
from paper_code.metrics import compute_rdm_vec, rsa_between_rdms, permutation_pvalue
from paper_code.features import extract_eeg_multichannel_aligned, reduce_audio_dimensions

# Extract features
eeg_features = extract_eeg_multichannel_aligned(eeg_aligned, n_time_steps=len(audio_aligned))
audio_features = reduce_audio_dimensions(audio_aligned, n_components=20)

# Compute RDMs
rdm_eeg = compute_rdm_vec(eeg_features)
rdm_audio = compute_rdm_vec(audio_features)

# Compute RSA
spearman, _, pearson = rsa_between_rdms(rdm_eeg, rdm_audio)
print(f"RSA: Spearman={spearman:.4f}, Pearson={pearson:.4f}")

# Permutation test for significance
pval = permutation_pvalue(eeg_features, audio_features, 
                          observed_rsa=spearman, permutations=500)  # n=500 as in paper
print(f"p-value: {pval:.4f}")
```

### Step 5: Batch Processing for All Electrodes

```python
from paper_code.metrics import compute_rdm_vec_batch, rsa_between_rdms_batch

# Process all 62 electrodes at once (C=62 as in paper)
eeg_batch = torch.randn(62, 100, 50)  # (n_electrodes, time_steps, features)
rdm_batch = compute_rdm_vec_batch(eeg_batch)

# Compare all electrodes with audio RDM
spearman_all, pearson_all = rsa_between_rdms_batch(rdm_batch, rdm_audio)
print(f"Mean RSA: {spearman_all.mean():.4f} ± {spearman_all.std():.4f}")
```

### Step 6: Layer-wise Analysis

```python
from paper_code.models import load_embeddings
from paper_code.visualization import plot_layerwise_comparison

# Compute RSA for each layer
layer_results = []
for layer_idx in range(24):  # WavLM-Large has 24 layers
    audio_emb = load_embeddings(npz_path, layer_idx=layer_idx)
    audio_features = reduce_audio_dimensions(audio_emb, n_components=20)
    rdm_audio = compute_rdm_vec(audio_features)
    rsa, _, _ = rsa_between_rdms(rdm_eeg, rdm_audio)
    layer_results.append({'layer_idx': layer_idx, 'rsa_spearman': rsa})

# Plot layer profile
import pandas as pd
df = pd.DataFrame(layer_results)
plot_layerwise_comparison({'WavLM-Large': df}, out_path='layer_profile.png')
```

### Step 7: Time-Window RSA → Scalp Topography (N400 Analysis)

```python
from paper_code.analysis import (
    n400_analysis, 
    sliding_window_rsa_analysis,
    erp_component_analysis,
    plot_time_window_topography,
    plot_rsa_time_course,
    ERP_WINDOWS
)
from paper_code.visualization import plot_topography_simple

# === N400 Component Analysis (300-500ms) ===
n400_results = n400_analysis(eeg, audio_features, eeg_fs=500, audio_fs=50)
print(f"N400 mean RSA: {n400_results['mean_rsa']:.4f}")
print(f"Best electrode: {n400_results['max_electrode']} (RSA={n400_results['max_rsa']:.4f})")

# Plot N400 scalp topography
plot_topography_simple(
    n400_results['electrode_rsa'],
    out_path='n400_topography.png',
    title='N400 RSA Topography (300-500ms)'
)

# === Multiple ERP Components ===
erp_results = erp_component_analysis(
    eeg, audio_features,
    components=['N100', 'P200', 'N400', 'P600'],
    eeg_fs=500, audio_fs=50
)

for comp_name, data in erp_results.items():
    print(f"{comp_name}: mean RSA = {data['mean_rsa']:.4f}")
    plot_topography_simple(
        data['electrode_rsa'],
        out_path=f'{comp_name.lower()}_topography.png',
        title=f'{comp_name} RSA ({int(data["window"].start_ms)}-{int(data["window"].end_ms)}ms)'
    )
```

### Step 8: Sliding Window RSA Time Course

```python
# === Sliding Window Analysis (250ms windows, 50ms steps) ===
sliding_results = sliding_window_rsa_analysis(
    eeg, audio_features,
    window_size_ms=250,    # 250ms time window
    step_size_ms=50,       # 50ms step
    eeg_fs=500, 
    audio_fs=50
)

print(f"Computed RSA for {len(sliding_results['windows'])} time windows")
print(f"RSA matrix shape: {sliding_results['rsa_matrix'].shape}")  # (n_windows, n_electrodes)

# Plot RSA time course (mean ± std across electrodes)
plot_rsa_time_course(
    sliding_results,
    out_path='rsa_time_course.png',
    title='RSA Time Course (250ms windows)'
)

# Generate topography series for each time window
from paper_code.analysis import plot_sliding_window_topography_series
plot_sliding_window_topography_series(
    sliding_results,
    out_dir='topography_series/',
    step=2,  # Plot every 2nd window
    vmin=-0.1, vmax=0.3
)
```

### Predefined ERP Time Windows

| Component | Time Range | Description |
|-----------|------------|-------------|
| N100 | 80-150ms | Early auditory processing |
| P200 | 150-250ms | Attention, stimulus evaluation |
| N200 | 200-300ms | Cognitive control |
| N400 | 300-500ms | Semantic processing |
| P300 | 250-500ms | Context updating |
| P600 | 500-800ms | Syntactic processing |

```python
# Access predefined windows
from paper_code.analysis import ERP_WINDOWS

n400_window = ERP_WINDOWS['N400']
print(f"N400: {n400_window.start_ms}-{n400_window.end_ms}ms")
```

---

## Supported Audio Models

| Model | Layers | Hidden Dim | Key |
|-------|--------|------------|-----|
| Audio-Flamingo-3 | 32 | 4096 | `audio-flamingo-3` |
| Baichuan-Audio-Base | 32 | 4096 | `baichuan-audio-base` |
| Baichuan-Audio-Instruct | 32 | 4096 | `baichuan-audio-instruct` |
| GLM-4-Voice-9B | 40 | 4096 | `glm-4-voice-9b` |
| Granite-Speech-3.3-8B | 32 | 4096 | `granite-speech-3.3-8b` |
| Llama-3.1-8B-Omni | 32 | 4096 | `llama-3.1-8b-omni` |
| MiniCPM-o-2_6 | 28 | 3584 | `minicpm-o-2_6` |
| Qwen2-Audio-7B | 32 | 4096 | `qwen2-audio` |
| Qwen2-Audio-7B-Instruct | 32 | 4096 | `qwen2-audio-instruct` |
| SpeechGPT-2.0-preview-7B | 32 | 4096 | `speechgpt` |
| Ultravox-v0.5 (Llama-3.1-8B) | 32 | 4096 | `ultravox-llama3.1-8b` |
| Ultravox-v0.5 (Llama-3.2-1B) | 16 | 2048 | `ultravox-llama3.2-1b` |

---

## Metrics Reference

### Representational Dissimilarity Matrix (RDM)

| Function | Description | Output |
|----------|-------------|--------|
| `compute_rdm_vec(X)` | Compute upper-triangular RDM vector (1 - correlation) | 1D tensor |
| `compute_rdm_full(X)` | Compute full RDM matrix | 2D tensor (N×N) |
| `compute_rdm_vec_batch(X_batch)` | Batch RDM computation for multiple electrodes | (B, n_pairs) |

### Representational Similarity Analysis (RSA)

| Function | Description | Range |
|----------|-------------|-------|
| `rsa_spearman(rdm1, rdm2)` | Spearman rank correlation between RDMs | [-1, 1] |
| `rsa_pearson(rdm1, rdm2)` | Pearson correlation between RDMs | [-1, 1] |
| `rsa_between_rdms(rdm1, rdm2)` | Returns (spearman, spearman_p, pearson) | - |
| `rsa_between_rdms_batch(rdm_batch, rdm_ref)` | Batch RSA for multiple electrodes | (B,) |
| `gpu_rankdata_average(x)` | GPU-accelerated rank computation | tensor |

### Centered Kernel Alignment (CKA)

| Function | Description | Range |
|----------|-------------|-------|
| `compute_cka(X, Y, kernel='linear')` | Unified CKA function | [0, 1] |
| `compute_cka_linear(X, Y)` | Linear kernel CKA | [0, 1] |
| `compute_cka_rbf(X, Y, sigma=None)` | RBF/Gaussian kernel CKA | [0, 1] |

### Distance Correlation

| Function | Description | Range |
|----------|-------------|-------|
| `compute_distance_correlation(X, Y)` | Distance correlation (dCor) | [0, 1] |

### Hilbert-Schmidt Independence Criterion (HSIC)

| Function | Description | Range |
|----------|-------------|-------|
| `compute_hsic_linear(X, Y)` | HSIC with linear kernel | [0, ∞) |
| `compute_hsic_rbf(X, Y, sigma=None)` | HSIC with RBF kernel | [0, ∞) |

### RV Coefficient

| Function | Description | Range |
|----------|-------------|-------|
| `compute_rv_coefficient(X, Y)` | Escoufier's RV coefficient | [0, 1] |

### Mutual Information

| Function | Description | Range |
|----------|-------------|-------|
| `compute_mutual_info_gaussian(X, Y)` | MI via Gaussian approximation | [0, ∞) |

### Kendall's Tau

| Function | Description | Range |
|----------|-------------|-------|
| `compute_kendall_tau_b(x, y)` | Kendall's Tau-b (handles ties) | [-1, 1] |
| `compute_kendall_tau_exact(x, y)` | Exact Tau computation | [-1, 1] |

### Permutation Testing

| Function | Description | Output |
|----------|-------------|--------|
| `permutation_pvalue(X, Y, observed, n_perm)` | Permutation test p-value | [0, 1] |
| `permutation_test_rsa(rdm1, rdm2, n_perm)` | RSA-specific permutation test | (rsa, p-value) |

### Quick Reference Table

| Category | Metric | Function | Range | GPU |
|----------|--------|----------|-------|-----|
| **RDM** | Correlation Distance | `compute_rdm_vec()` | [0, 2] | ✅ |
| **RSA** | Spearman | `rsa_spearman()` | [-1, 1] | ✅ |
| **RSA** | Pearson | `rsa_pearson()` | [-1, 1] | ✅ |
| **CKA** | Linear | `compute_cka_linear()` | [0, 1] | ✅ |
| **CKA** | RBF | `compute_cka_rbf()` | [0, 1] | ✅ |
| **Dependence** | Distance Correlation | `compute_distance_correlation()` | [0, 1] | ✅ |
| **Dependence** | HSIC (Linear) | `compute_hsic_linear()` | [0, ∞) | ✅ |
| **Dependence** | HSIC (RBF) | `compute_hsic_rbf()` | [0, ∞) | ✅ |
| **Similarity** | RV Coefficient | `compute_rv_coefficient()` | [0, 1] | ✅ |
| **Information** | Mutual Info | `compute_mutual_info_gaussian()` | [0, ∞) | ✅ |
| **Correlation** | Kendall Tau-b | `compute_kendall_tau_b()` | [-1, 1] | ✅ |
| **Statistics** | Permutation Test | `permutation_pvalue()` | [0, 1] | ✅ |

---

## GPU Acceleration

All metrics are GPU-accelerated using PyTorch:

```python
from paper_code.utils import setup_gpu

# Auto-select GPU with most free memory
device = setup_gpu()

# Or specify GPU index
device = setup_gpu(gpu_index=0)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{anonymous2024eegaudio,
  title={Acoustic-EEG Similarity Analysis with Audio Language Models},
  author={Anonymous},
  booktitle={Proceedings of ACL},
  year={2024}
}
```

## License

This project is licensed under the MIT License.
