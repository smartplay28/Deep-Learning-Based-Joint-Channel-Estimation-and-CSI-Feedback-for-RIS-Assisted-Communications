# Deep Learning-Based Joint Channel Estimation and CSI Feedback for RIS-Assisted Communications

> **Course Project**: Machine Learning for Wireless Communication  
> **Authors**: Akshat Mittal, Snehal Sharma, Hrushikesh Sawant  
> **Institution**: International Institute of Information Technology Bangalore  
> **Base Paper**: Feng et al., "Deep Learning-Based Joint Channel Estimation and CSI Feedback for RIS-Assisted Communications," _IEEE Communications Letters_, vol. 28, no. 8, pp. 1860--1864, Aug. 2024.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Background and Motivation](#background-and-motivation)  
3. [System Model](#system-model)  
4. [Base Paper: JDCNet Architecture](#base-paper-jdcnet-architecture)  
5. [Our Contributions and Novelty](#our-contributions-and-novelty)  
6. [Repository Structure](#repository-structure)  
7. [Dataset Generation](#dataset-generation)  
8. [Implementation Details](#implementation-details)  
9. [Results and Evaluation](#results-and-evaluation)  
10. [How to Run](#how-to-run)  
11. [References](#references)

---

## Project Overview

This project implements and extends **JDCNet** (Joint Deep Learning Channel Estimation and CSI Feedback Network), a unified encoder--decoder deep learning framework for Reconfigurable Intelligent Surface (RIS)-assisted wireless communication systems. The original paper proposes jointly optimizing channel estimation and CSI (Channel State Information) feedback within a single end-to-end network, thereby eliminating the cumulative errors that arise from treating these two tasks independently.

We faithfully reproduce the base paper results and introduce **three novel extensions** that go beyond the original work:

1. **Multi-User Extension (MU-JDCNet)** -- a shared-encoder, per-user-decoder architecture that scales JDCNet to K=4 simultaneous users.  
2. **SNR-Adaptive Dynamic Quantization** -- an adaptive bit-allocation scheme that adjusts feedback precision based on channel quality.  
3. **SNR-Conditioned Decoding with FiLM Layers and Curriculum Training** -- decoder-side SNR conditioning via Feature-wise Linear Modulation (FiLM) combined with a structured curriculum training strategy.

---

## Background and Motivation

### Reconfigurable Intelligent Surfaces (RIS)

RIS technology introduces large, programmable metasurfaces comprising N passive reflecting elements that can independently control the phase shift of incident signals. By intelligently configuring these phase shifts, an RIS creates favorable propagation conditions between a base station (BS) and user equipment (UE), enabling coverage extension, interference mitigation, and enhanced spectral efficiency -- all without additional power amplification.

### The CSI Acquisition Challenge

Effective RIS beamforming demands accurate CSI at the BS. In RIS-assisted systems, the cascaded channel (BS -> RIS -> UE) has dimensionality M x N (BS antennas x RIS elements), which can be extremely large. For example, with M=16 antennas and N=64 RIS elements, 1024 complex channel parameters must be estimated. This creates three compounding challenges:

- **Pilot Overhead**: Traditional estimation requires a number of pilot symbols proportional to M x N, consuming valuable transmission time.
- **Feedback Bandwidth**: The UE must compress and transmit the estimated CSI back to the BS over a limited-bandwidth uplink feedback channel.
- **Error Propagation**: When channel estimation and CSI feedback are optimized independently (as in prior art), errors from the estimation stage propagate into and compound with errors from the feedback compression stage.

### Limitations of Prior Work

| Approach | Method | Limitation |
|---|---|---|
| Channel Estimation Only | SRNet, CNN-based, EDSR | Does not address feedback overhead; errors propagate to downstream feedback |
| CSI Feedback Only | CsiNet, CLNet, DCRNet | Assumes perfect CSI input; separate optimization from estimation |
| Compressive Sensing | TVAL3, BM3D-PRGAMP | Relies on channel sparsity assumptions that may not hold in RIS scenarios |
| Separate Optimization | Estimation + Feedback combined | Each module optimized independently; cumulative errors degrade performance |

**Key Gap**: No prior method jointly optimizes channel estimation and CSI feedback within a single, end-to-end trainable framework for RIS-assisted systems.

---

## System Model

The system comprises:

- **Base Station (BS)**: Equipped with M=16 antennas (ULA configuration)
- **Reconfigurable Intelligent Surface (RIS)**: N=64 passive reflecting elements
- **User Equipment (UE)**: Single antenna (Nr=1)
- **Operating Frequency**: 28 GHz (mmWave)
- **Environment**: Indoor InH Office

### Channel Model

Communication occurs through the RIS-reflected path. The cascaded channel is formulated as:

```
H = diag(h_rd) * G
```

where:
- `G` is the N x M BS-to-RIS channel matrix
- `h_rd` is the N x 1 RIS-to-UE channel vector
- `H` is the N x M effective cascaded channel

The received signal at the UE is:

```
y(t) = phi(t)^T * H * w(t) * s(t) + z(t)
```

where `phi` is the RIS phase-shift vector, `w` is the beamforming vector, `s` is the transmitted symbol, and `z` is AWGN noise.

### Dimension Reduction via RIS Grouping

To reduce the dimensionality of the estimation problem, adjacent RIS elements are grouped into clusters of size N0. This reduces the number of unknowns from M x N to M x N', where N' = N/N0. Two configurations are studied:

| Parameter | N0 = 2 | N0 = 4 |
|---|---|---|
| Grouped Elements | 32 | 16 |
| Channel Parameters | 512 | 256 |
| Pilot Reduction | 2x | 4x |

### LS Channel Estimation

The initial Least Squares (LS) estimate of the grouped cascaded channel is obtained as:

```
vec(H_LS) = (W^H * W)^{-1} * W^H * y
```

where W = V (x) Phi (Kronecker product of DFT beamforming and RIS phase-shift matrices).

---

## Base Paper: JDCNet Architecture

JDCNet is an encoder--decoder convolutional neural network that jointly performs CSI compression at the UE and channel reconstruction at the BS.

### Encoder (UE Side)

The encoder operates on the noisy, grouped LS channel estimate and compresses it for feedback:

1. **Input**: Real and imaginary parts of the LS estimate are separated into 2 channels, forming a tensor of shape (2, M, N')
2. **Min-Max Normalization**: Values are scaled to [0, 1]
3. **Two Convolutional Layers**: Each with 4x4 kernels, stride 2, padding 1, followed by ReLU activation
4. **Flattening**: Output is flattened into a feature vector v of dimension C * M * N' / 16
5. **Uniform Quantization**: The feature vector is quantized to q bits per element using Straight-Through Estimator (STE) for gradient flow

The compression ratio is defined as:

```
gamma = C / (32 * N0)
```

With gamma = 1/16, this achieves 16x compression of the original CSI.

### Decoder (BS Side)

The decoder reconstructs the full (ungrouped) channel matrix from the quantized feedback:

1. **Dequantization and Reshape**: Bit stream is converted back to a feature map
2. **Two Transposed Convolution Layers**: Upsample the feature maps
3. **B=4 Residual Blocks**: Each containing two 3x3 convolutions with ReLU and skip connections for refined feature extraction
4. **Final Upsampling**: Transposed convolution to expand from grouped to full RIS dimension
5. **Output Convolution**: 3x3 convolution layer producing the reconstructed channel matrix (2, M, N)

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Training Samples | 40,000 |
| Validation Samples | 4,000 |
| Test Samples | 4,000 |
| Batch Size | 256 |
| Epochs | 200 |
| Optimizer | Adam (default parameters) |
| Learning Rate Schedule | Cosine annealing with warm-up (Tw=30) |
| Maximum LR | 2e-3 |
| Minimum LR | 5e-5 |
| Loss Function | Mean Squared Error (MSE) |
| Quantization Bits | q = 8 |

### Performance Metric

Normalized Mean Square Error (NMSE) in dB:

```
NMSE = E[||H_hat - H||^2] / E[||H||^2]
```

---

## Our Contributions and Novelty

Beyond the faithful reproduction of the base paper, this project introduces three significant extensions that address limitations not covered in the original work. Each extension was developed iteratively, with systematic ablation and version tracking.

---

### [Contribution 1] Multi-User JDCNet (MU-JDCNet)

**Motivation**: The original JDCNet serves only a single user. Practical RIS-assisted systems must simultaneously support multiple users sharing the same RIS infrastructure.

**Architecture**: MU-JDCNet employs a shared-encoder, per-user-decoder design:

- **Shared Encoder**: A single encoder processes the LS estimates from all K=4 users. The encoded latent representations are **aggregated via summation** at the bottleneck layer, creating a compact, shared representation that captures cross-user channel correlations.
- **Per-User Decoders**: K=4 independent decoder branches reconstruct the individual cascaded channel for each user. Each decoder has its own set of transposed convolution layers, residual blocks, and final reconstruction layers.
- **Quantization Annealing**: Quantization is disabled for the first 50 epochs to allow the network to first learn meaningful latent representations, then enabled for the remaining 250 epochs.

**Key Design Choices**:

| Feature | Detail |
|---|---|
| Users (K) | 4 |
| Decoder Channels | 32 (increased from 16) |
| ResBlocks per Decoder | B=6 (increased from 4) |
| Training Epochs | 300 (extended from 200) |
| Gradient Clipping | max_norm = 1.0 |
| BatchNorm in ResBlocks | Added for multi-user training stability |
| Quantization Annealing | OFF for epochs 0--49, ON for epochs 50--299 |
| Training SNR Range | [0, 35] dB (widened to cover the full test range) |

**Dataset**: A dedicated multi-user dataset was generated using MATLAB with 4 UE positions, producing independent RIS-to-UE channels (G_1 through G_4) while sharing the BS-to-RIS channel (H).

---

### [Contribution 2] SNR-Adaptive Dynamic Quantization

**Motivation**: The base paper uses a fixed number of quantization bits (q=8) regardless of channel conditions. However, at low SNR, the channel estimate is dominated by noise -- high quantization precision wastes feedback bits encoding noise. At high SNR, the channel estimate is cleaner and benefits from finer quantization to preserve signal detail.

**Approach**: A dynamic bit-allocation function maps the operating SNR to an appropriate quantization bit-depth:

```
q(SNR) = round(q_min + (q_max - q_min) * (SNR - SNR_min) / (SNR_max - SNR_min))
```

**Evolution**:

| Version | Bit Range | SNR Range | Notes |
|---|---|---|---|
| v1 (Initial) | 2--8 bits | 0--35 dB | Linear mapping; no architecture changes; marginal gains |
| v2 (Final) | 3--10 bits | 0--35 dB | Extended precision range; higher max bits for high-SNR regime |

**Rationale**: This adaptive scheme reduces the average feedback bandwidth (fewer bits at low SNR) while preserving reconstruction accuracy at high SNR (more bits when they matter). The Straight-Through Estimator (STE) enables gradient-based training through the non-differentiable quantization operation with variable bit-widths.

---

### [Contribution 3] SNR-Conditioned Decoding via FiLM Layers and Curriculum Training

**Motivation**: The base paper's decoder processes compressed CSI without any knowledge of the noise level at which the channel was estimated. This forces the decoder to learn a single denoising strategy averaged across all SNR levels, leading to suboptimal performance at both extremes of the SNR range.

**Approach -- FiLM-Based SNR Conditioning**: A Feature-wise Linear Modulation (FiLM) layer is injected into the decoder after the residual block stack. An MLP-based SNR conditioner takes the normalized SNR as input and produces channel-wise scale and bias parameters that modulate the decoder features:

```
y_out = y * (1 + alpha * scale) + bias
```

where `scale` and `bias` are generated by the MLP, and `alpha` is a residual scaling factor (0.3) to prevent extreme modulations.

**SNR Conditioner Evolution**:

| Version | MLP Size | Output Activation | Notes |
|---|---|---|---|
| v3 (Initial) | 1 -> 32 -> 2*ch | Linear | FiLM: y * (scale + 1) + bias |
| v4 (Final) | 1 -> 64 -> 64 -> 2*ch | Tanh (bounded) | Residual connection; bounded modulation for stability |

**Approach -- Curriculum Training**: A structured curriculum schedules the SNR distribution during training:

- **Phase 1 (Warm-up, epochs 0--40%)**: Training begins with high-SNR (easy) examples only. The minimum training SNR progressively decreases from `snr_ceil` to `snr_floor`.
- **Phase 2 (Full range, epochs 40--100%)**: The full SNR range [0, 35] dB is used. In the final curriculum version, 70% of samples in the latter half of Phase 2 are drawn from the high-SNR regime (> 20 dB) to address the observed high-SNR performance plateau.

**Additional Training Improvements**:

- **Gradient Clipping**: max_norm = 1.0 for training stability
- **6 ResBlocks**: Increased from B=4 to provide more decoder capacity
- **Extended SNR Training Range**: Expanded to [0, 35] dB to match the full test evaluation range

---

### Summary of Iterative Development

The following table summarizes the progression from the base paper reproduction to the final extended architectures:

| Version | Architecture | Key Changes | Status |
|---|---|---|---|
| Mid-Progress (n0-2, n0-4) | JDCNet_Paper | Faithful reproduction with N0=2 and N0=4 | Baseline established |
| v1 | JDCNet_Paper + dynamic q-bits | SNR-to-bits mapping (2--8 bits); no architecture changes | Marginal improvement |
| v2 | JDCNet_V3 | FiLM-based SNR conditioning; B=6 ResBlocks; curriculum training | Gains at low SNR |
| v3 (Final Single-User) | JDCNet_V4 | Larger MLP conditioner; 3--10 bit range; high-SNR curriculum focus | Balanced gains across SNR |
| Multi-User | MU-JDCNet | Shared encoder + K=4 per-user decoders; quantization annealing; BatchNorm | New capability |

---

## Repository Structure

```
.
|-- README.md
|-- Presentation.pdf                         # Project presentation slides
|-- CE4_Deep_Learning-Based_Joint_...pdf     # Original IEEE research paper
|
|-- Dataset Generation/
|   |-- Multiuser/
|       |-- datagen.m                        # MATLAB script for multi-user channel generation
|                                            # (SimRIS_v18 channel simulator embedded)
|
|-- Mid Progress Implementation/
|   |-- n0-2-final.ipynb                     # Base paper reproduction: N0=2 grouping
|   |-- n0-4-final.ipynb                     # Base paper reproduction: N0=4 grouping
|
|-- Post Mid Progress Implementation/
    |-- v1_initial_dyanamic_quantisation.ipynb   # Dynamic quantization (v1)
    |-- v2_part1.ipynb                           # SNR conditioning + curriculum (v3 arch)
    |-- v2_part2.ipynb                           # Continued training and evaluation
    |-- v3_arch_with_all_changes.ipynb           # Final JDCNet_V4 with all improvements
    |-- n0-2-multiuser.ipynb                     # Multi-user JDCNet (K=4, N0=2)
    |-- n0-4-multiuser.ipynb                     # Multi-user JDCNet (K=4, N0=4)
```

---

## Dataset Generation

### Single-User Dataset

Channel data is generated using the **SimRIS Channel Simulator** (v18, Koc University CoreLab) embedded within our MATLAB scripts. The simulator models the 3GPP Indoor Hotspot (InH) office environment at 28 GHz with both LoS and NLoS propagation paths.

**Parameters**:

| Parameter | Value |
|---|---|
| Environment | Indoor InH Office |
| Frequency | 28 GHz |
| BS Antennas (M) | 16 (ULA) |
| RIS Elements (N) | 64 |
| UE Antennas (Nr) | 1 |
| BS Position | (0, 25, 2) m |
| RIS Position | (40, 50, 2) m |
| Total Samples | 48,000 |

### Multi-User Dataset

For the multi-user extension, the same SimRIS simulator generates independent RIS-to-UE channels for K=4 users at distinct spatial locations, while the BS-to-RIS channel (H) is shared across all users.

**User Positions**:

| User | Position (x, y, z) | Distance to RIS |
|---|---|---|
| UE1 | (45, 45, 1) m | 7.1 m |
| UE2 | (38, 43, 1) m | 7.4 m |
| UE3 | (44, 42, 1) m | 9.0 m |
| UE4 | (36, 47, 1) m | 5.1 m |

**Output**: `RIS_Channels_MU_K4.mat` containing:
- `H`: BS-to-RIS channel (64 x 16 x 48000)
- `G_1` through `G_4`: RIS-to-UE channels per user (1 x 64 x 48000)
- `D`: Direct BS-to-UE channel (1 x 16 x 48000)

The cascaded channel for each user k is computed as:

```
H_cascade_k = diag(G_k^T) * H
```

---

## Implementation Details

### Framework and Hardware

- **Framework**: PyTorch 1.11+
- **Training Hardware**: NVIDIA A6000 / T4 GPUs (Kaggle)
- **Dataset Format**: MATLAB `.mat` files (v7 for single-user, v7.3 HDF5 for multi-user)

### Key Implementation Components

**Straight-Through Estimator (STE) Quantizer**: Enables gradient-based training through the non-differentiable uniform quantization operation. During the forward pass, quantization is applied; during the backward pass, gradients are passed through unchanged.

**LS Pilot Generation**: Implemented in PyTorch to generate noisy grouped LS channel estimates on-the-fly during training. The SNR is sampled from a configurable range to ensure the model generalizes across operating conditions.

**Min-Max Normalization**: Per the paper specification, global train-set min/max values are computed and used to normalize all channel data to [0, 1] before encoder input.

**DataParallel Compatibility**: A flexible state-dict loader handles the `module.` prefix mismatch that arises when switching between single-GPU and multi-GPU (DataParallel) configurations.

---

## Results and Evaluation

All methods are evaluated using NMSE (dB) across an SNR range of [0, 35] dB with compression ratio gamma = 1/16 and quantization bits q = 8 (unless dynamic).

### Base Paper Reproduction

The faithful JDCNet reproduction achieves NMSE performance consistent with the published results for both N0=2 and N0=4 grouping configurations, confirming the validity of our implementation.

### Novel Extension Results

- **JDCNet_V4** (SNR conditioning + curriculum + dynamic bits): Demonstrates improved or comparable NMSE across the full SNR spectrum versus the base JDCNet, with particular gains at low SNR due to FiLM-conditioned denoising and maintained high-SNR accuracy through curriculum focusing.
- **MU-JDCNet**: Successfully extends the single-user framework to K=4 users with per-user NMSE tracking, achieving channel reconstruction for all users through a shared bottleneck representation.
- **Dynamic Quantization**: Provides rate-distortion flexibility by reducing average feedback bits while preserving reconstruction quality.

---

## How to Run

### Prerequisites

```
Python >= 3.8
PyTorch >= 1.11
NumPy
SciPy
h5py (for multi-user HDF5 datasets)
matplotlib
```

### Step 1: Dataset Generation

Generate single-user or multi-user channel data using MATLAB:

```matlab
% In MATLAB, run:
cd 'Dataset Generation/Multiuser'
datagen  % Generates RIS_Channels_MU_K4.mat
```

Ensure the SimRIS simulator parameters match the configuration specified above. The script is self-contained with the SimRIS_v18 function embedded.

### Step 2: Training

All training notebooks are designed to run on **Kaggle** with GPU acceleration. Upload the generated `.mat` dataset and execute the notebooks in order:

1. **Base Paper Reproduction**: Run `Mid Progress Implementation/n0-2-final.ipynb` or `n0-4-final.ipynb`
2. **Dynamic Quantization**: Run `Post Mid Progress Implementation/v1_initial_dyanamic_quantisation.ipynb`
3. **SNR Conditioning + Curriculum**: Run `Post Mid Progress Implementation/v2_part1.ipynb` followed by `v2_part2.ipynb`
4. **Final Architecture (V4)**: Run `Post Mid Progress Implementation/v3_arch_with_all_changes.ipynb`
5. **Multi-User**: Run `Post Mid Progress Implementation/n0-2-multiuser.ipynb` or `n0-4-multiuser.ipynb`

Each notebook contains the complete pipeline: data loading, model definition, training loop, and SNR-wise NMSE evaluation with visualization.

### Step 3: Evaluation

Each notebook evaluates the trained model over the SNR grid [0, 5, 10, 15, 20, 25, 30, 35] dB and produces NMSE plots comparing against the paper benchmarks.

---

## References

1. Feng et al., "Deep Learning-Based Joint Channel Estimation and CSI Feedback for RIS-Assisted Communications," _IEEE Communications Letters_, vol. 28, no. 8, pp. 1860--1864, Aug. 2024.
---

**Note**: This project was developed as part of an academic course on Machine Learning for Wireless Communications. The base paper (JDCNet) is the work of Feng et al.; all extensions described under "Our Contributions and Novelty" are original contributions by the authors of this project.
