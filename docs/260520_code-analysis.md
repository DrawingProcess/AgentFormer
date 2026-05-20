# AgentFormer Code Analysis

**Date**: May 20, 2026
**Paper**: AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting
**Authors**: Ye Yuan, Xinshuo Weng, Yanglan Ou, Kris Kitani
**Conference**: ICCV 2021

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Core Architecture](#core-architecture)
4. [Key Components](#key-components)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Data Processing](#data-processing)
8. [Loss Functions](#loss-functions)
9. [Configuration System](#configuration-system)
10. [Code Quality & Design Patterns](#code-quality--design-patterns)

---

## Overview

AgentFormer is a trajectory forecasting model designed for multi-agent scenarios, focusing on socio-temporal interactions. The model predicts future trajectories of multiple agents (pedestrians, vehicles, etc.) while considering their interactions with each other.

### Key Features

- **Agent-Aware Attention Mechanism**: Custom attention that models both inter-agent and intra-agent (self) attention separately
- **Transformer-Based Architecture**: Encoder-decoder structure using transformers
- **VAE Framework**: Variational Autoencoder for stochastic trajectory prediction
- **Two-Stage Training**:
  1. VAE model (AgentFormer)
  2. DLow trajectory sampler for diverse predictions

### Datasets Supported

- **ETH/UCY**: Pedestrian trajectory datasets (ETH, Hotel, Univ, Zara1, Zara2)
- **nuScenes**: Autonomous driving dataset with vehicle trajectories

### Important Note

The codebase includes a fix for a [normalization bug](https://github.com/Khrylx/AgentFormer/issues/5) mentioned in the README. Updated results are available in the arXiv version of the paper.

---

## Project Structure

```
AgentFormer/
├── cfg/                          # Configuration files
│   ├── eth_ucy/                  # ETH/UCY dataset configs
│   │   ├── eth/
│   │   ├── hotel/
│   │   ├── univ/
│   │   ├── zara1/
│   │   └── zara2/
│   └── nuscenes/                 # nuScenes dataset configs
│       ├── 5_sample/
│       └── 10_sample/
├── data/                         # Data loading and preprocessing
│   ├── dataloader.py             # Main data generator
│   ├── preprocessor.py           # Data preprocessing
│   ├── ethucy_split.py           # ETH/UCY dataset splits
│   ├── nuscenes_pred_split.py    # nuScenes dataset splits
│   ├── process_nuscenes.py       # nuScenes data processing
│   ├── homography_warper.py      # Coordinate transformations
│   ├── map.py                    # Map handling
│   └── convert_ethucy.py         # ETH/UCY format conversion
├── datasets/                     # Dataset files
│   ├── eth_ucy/                  # ETH/UCY trajectory data
│   └── nuscenes_pred/            # nuScenes trajectory data
├── model/                        # Model implementations
│   ├── agentformer.py            # Main AgentFormer model
│   ├── agentformer_lib.py        # Agent-aware attention implementation
│   ├── agentformer_loss.py       # Loss functions for AgentFormer
│   ├── dlow.py                   # DLow trajectory sampler
│   ├── map_encoder.py            # Map encoding module
│   ├── map_cnn.py                # CNN for map features
│   ├── model_lib.py              # Model registry
│   └── common/                   # Common utilities
│       ├── mlp.py                # Multi-layer perceptron
│       ├── dist.py               # Probability distributions
│       └── resnet.py             # ResNet blocks
├── utils/                        # Utility functions
│   ├── config.py                 # Configuration management
│   ├── torch.py                  # PyTorch utilities
│   └── utils.py                  # General utilities
├── results/                      # Saved models and results
├── train.py                      # Training script
├── test.py                       # Testing script
├── eval.py                       # Evaluation metrics
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

---

## Core Architecture

### High-Level Architecture

AgentFormer follows an encoder-decoder architecture with three main components:

1. **Context Encoder**: Encodes past trajectories of all agents
2. **Future Encoder**: Encodes ground truth future trajectories (training only) to learn latent distribution
3. **Future Decoder**: Decodes latent codes to predict future trajectories

```
Input: Past Trajectories (T_past × N_agents × 2)
                    ↓
        [Context Encoder]
                    ↓
        Context Encoding (T_past × N_agents × D_model)
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
[Future Encoder]        [Future Decoder]
(Training only)              ↓
        ↓               Predicted Trajectories
   Latent z            (T_future × N_agents × 2)
        ↓
  KL Divergence Loss
```

### Agent-Aware Attention Mechanism

The core innovation is the **Agent-Aware Attention** mechanism (`model/agentformer_lib.py:27-341`), which computes attention in two separate streams:

1. **Inter-Agent Attention**: Models interactions between different agents
2. **Intra-Agent Attention** (Self): Models temporal dependencies within the same agent

**Implementation** (`model/agentformer_lib.py:289-300`):

```python
# Separate attention weights for inter-agent and intra-agent (self)
attn_output_weights_inter = attn_output_weights  # Standard cross-agent attention
attn_output_weights_self = torch.bmm(q_self, k_self.transpose(1, 2))  # Self attention

# Create mask to separate self and inter-agent attention
attn_weight_self_mask = torch.eye(num_agent)  # Identity matrix for self-attention

# Combine: use self attention for diagonal, inter-agent for off-diagonal
attn_output_weights = (attn_output_weights_inter * (1 - attn_weight_self_mask) +
                       attn_output_weights_self * attn_weight_self_mask)
```

This allows the model to:
- Learn different attention patterns for self-dynamics vs. social interactions
- Use separate learnable parameters for inter-agent and intra-agent attention
- Better capture the nuanced differences between temporal and social reasoning

### Model Components in Detail

#### 1. Context Encoder (`model/agentformer.py:103-168`)

**Purpose**: Encode observed past trajectories into context representations.

**Key Features**:
- Takes past trajectories (position, velocity, or normalized coordinates)
- Uses AgentFormer encoder layers with agent-aware attention
- Supports multiple input types: position, velocity, normalized coordinates, scene-normalized coordinates, heading vectors, map features
- Outputs per-agent context via pooling (mean or max)

**Architecture**:
```
Input Trajectories → Linear Projection → Positional Encoding
→ AgentFormer Encoder Layers (N layers) → Agent Context
```

#### 2. Future Encoder (`model/agentformer.py:171-256`)

**Purpose**: Learn a posterior distribution q(z|past, future) over latent codes during training.

**Key Features**:
- Only used during training (not inference)
- Encodes future ground truth trajectories
- Uses AgentFormer decoder layers to attend to context
- Produces parameters for latent distribution (Gaussian or Categorical)
- Enables stochastic prediction via VAE framework

**Architecture**:
```
Future Trajectories → Linear Projection → Positional Encoding
→ AgentFormer Decoder Layers → MLP → q(z) parameters
```

#### 3. Future Decoder (`model/agentformer.py:259-428`)

**Purpose**: Generate future trajectory predictions from latent codes and context.

**Key Features**:
- Autoregressive decoding: predicts one timestep at a time
- Learns prior distribution p(z) for inference
- Supports different prediction types: velocity, position, scene-normalized
- Can detach gradients during autoregression for stability

**Architecture**:
```
Initial State + Latent z + Context → Transformer Decoder
→ Autoregressive Loop (T_future steps) → Predicted Trajectories
```

**Autoregressive Decoding** (`model/agentformer.py:306-391`):
- Starts with last observed position/velocity
- At each timestep:
  1. Concatenate current state with latent z and optional features (heading, map)
  2. Pass through transformer decoder with agent-aware attention
  3. Predict offset/velocity for next timestep
  4. Update current state (with optional gradient detachment)
  5. Repeat for all future timesteps

---

## Key Components

### 1. Positional and Agent Encoding (`model/agentformer.py:32-100`)

**PositionalAgentEncoding** combines:

- **Temporal Positional Encoding**: Standard sinusoidal encoding for timesteps
- **Agent Encoding**: Optional encoding to distinguish different agents (can be learned or fixed)

```python
# Build sinusoidal positional encoding
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

# Combine position and agent encodings
if self.use_agent_enc:
    x += pos_enc + agent_enc  # Element-wise addition
```

Supports two modes:
- **Concatenation**: `[features, pos_enc, agent_enc]` → Linear projection
- **Addition**: `features + pos_enc + agent_enc`

### 2. Masking Mechanisms

#### Agent Mask (`model/agentformer.py:571-583`)

Controls which agents can attend to each other based on spatial proximity:

```python
conn_dist = cfg.get('conn_dist', 100000.0)  # Connection distance threshold
if conn_dist < 1000.0:
    # Compute pairwise distances
    pdist = F.pdist(cur_motion)
    # Mask out agent pairs beyond threshold
    mask[D > threshold] = float('-inf')
```

#### Attention Masks

- **Source Mask**: For encoder self-attention (`model/agentformer.py:23-29`)
- **Target Mask**: For decoder self-attention with autoregressive masking (`model/agentformer.py:15-23`)
- **Memory Mask**: For decoder cross-attention to encoder outputs (`model/agentformer.py:26-29`)

### 3. Map Encoding

**Map Encoder** (`model/map_encoder.py`): Encodes scene context from bird's-eye view maps.

Features:
- Crops local map patches around each agent
- Supports rotation alignment (global or agent-heading-based)
- Uses CNN (typically ResNet) to extract features
- Concatenated with trajectory features

### 4. Latent Variable Framework

**Two Latent Distributions**:

1. **Posterior q(z|past, future)**: Learned by Future Encoder during training
2. **Prior p(z)**:
   - **Fixed**: Standard Gaussian (default)
   - **Learned**: Predicted from context (optional)

**Latent Types** (`model/agentformer.py:448`):
- **Gaussian**: Continuous latent space with reparameterization trick
- **Categorical/Discrete**: Discrete latent codes with Gumbel-Softmax

**Sampling**:
- Training: Sample from posterior q(z)
- Inference: Sample from prior p(z) multiple times for diverse predictions

### 5. DLow Trajectory Sampler (`model/dlow.py`)

**Purpose**: Second-stage model that learns to sample diverse latent codes.

**Architecture**:
- Loads pretrained AgentFormer VAE (frozen)
- Learns Q-network to map context to diverse latent codes
- Uses affine transformation: `z = A * ε + b` where ε ~ N(0, I)

**Q-Network**:
```python
qnet_h = MLP(agent_context)  # [512, 256] hidden layers
A = Linear(qnet_h, nk * nz)  # Scale parameters
b = Linear(qnet_h, nk * nz)  # Bias parameters
z = A * eps + b
```

**Training Objectives**:
1. **KL Divergence**: q(z) should match p(z)
2. **Diversity Loss**: Encourage diverse predictions
3. **Reconstruction Loss**: Predictions should match ground truth

---

## Training Pipeline

### Two-Stage Training Process

#### Stage 1: AgentFormer VAE (`train.py`)

**Objective**: Train the encoder-decoder VAE model.

```bash
python train.py --cfg user_eth_agentformer_pre --gpu 0
```

**Training Loop** (`train.py:30-63`):

1. Load batch of trajectories
2. Forward pass through model:
   - Context encoder processes past
   - Future encoder learns q(z|past, future)
   - Future decoder predicts trajectories
3. Compute losses:
   - **MSE Loss**: Reconstruction error
   - **KL Divergence**: q(z) vs. p(z)
   - **Sample Loss**: Best-of-K prediction error (optional)
4. Backpropagate and update weights
5. Step learning rate scheduler
6. Step KL annealer (for discrete z)

**Key Configuration**:
- Learning rate: 1e-4
- Past frames: 8
- Future frames: 12
- Latent dimension: 32
- Transformer: 6 layers, 8 heads, 512 model dim, 2048 FFN dim

#### Stage 2: DLow Sampler (`train.py` with DLow config)

**Objective**: Train the trajectory sampler for diverse predictions.

```bash
python train.py --cfg user_eth_agentformer --gpu 0
```

**Training Loop**:

1. Load pretrained AgentFormer VAE (frozen)
2. Forward pass through DLow:
   - Context encoding from frozen AgentFormer
   - Q-network generates K diverse latent codes
   - Decoder produces K trajectory samples
3. Compute losses:
   - **KL Divergence**: Q-network output vs. prior
   - **Diversity Loss**: Encourages sample diversity
   - **Reconstruction Loss**: Best sample should match GT
4. Update only Q-network parameters

**Key Configuration**:
- Sample K: 20 diverse predictions
- Q-network MLP: [512, 256]
- Share epsilon across agents (optional)
- Train with mean (optional)

### Data Augmentation

**Random Scene Rotation** (`model/agentformer.py:531-543`):

```python
if self.rand_rot_scene and self.training:
    if self.discrete_rot:
        theta = torch.randint(high=24, size=(1,)) * (np.pi / 12)  # 15° increments
    else:
        theta = torch.rand(1) * np.pi * 2  # Random angle
    # Rotate all trajectories and headings
    for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
        self.data[f'{key}'], self.data[f'{key}_scene_norm'] = \
            rotation_2d_torch(self.data[key], theta, self.data['scene_orig'])
```

**Agent Shuffling** (`model/agentformer.py:566-569`):

Randomizes agent order to prevent the model from learning position-dependent biases.

### Optimization

**Optimizer**: Adam with configurable learning rate (default: 1e-4)

**Learning Rate Schedulers**:
1. **Linear**: Fixed LR for N epochs, then linear decay
2. **Step**: Multiply LR by gamma every N epochs

**Gradient Clipping**: Not explicitly used, but backpropagation is standard.

**Regularization**:
- Dropout in transformer layers (default: 0.1)
- KL divergence with minimum clipping to prevent posterior collapse
- Optional gradient detachment in autoregressive decoder

---

## Evaluation Pipeline

### Testing Script (`test.py`)

**Functionality**:
1. Load trained model checkpoint
2. Generate predictions for test set
3. Save predictions to disk
4. Compute evaluation metrics

**Usage**:
```bash
python test.py --cfg eth_agentformer --gpu 0 --data_eval test
```

**Prediction Generation** (`test.py:16-21`):

```python
def get_model_prediction(data, sample_k):
    model.set_data(data)
    # Reconstruction: single best prediction
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    # Stochastic sampling: K diverse predictions
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k)
    return recon_motion_3D, sample_motion_3D
```

**Modes**:
- **recon**: Single deterministic prediction using posterior mode
- **infer**: K stochastic predictions by sampling from prior

**Output Structure**:
```
results/epoch_XXXX/test/
├── gt/                    # Ground truth trajectories
├── recon/                 # Deterministic predictions
└── samples/               # Stochastic predictions
    ├── sample_000/
    ├── sample_001/
    └── ...
```

### Evaluation Metrics (`eval.py`)

**Metrics Computed**:

1. **ADE (Average Displacement Error)** (`eval.py:11-19`):
   - Average L2 distance across all timesteps
   - Computed for best of K samples: `min_k(mean_t(||pred_k - gt||))`

2. **FDE (Final Displacement Error)** (`eval.py:22-30`):
   - L2 distance at final timestep only
   - Computed for best of K samples: `min_k(||pred_k[-1] - gt[-1]||)`

**Evaluation Process**:
```python
for each scene:
    for each agent:
        # Load predictions (K samples x T frames x 2)
        # Load ground truth (T frames x 2)
        # Align predictions and GT by frame IDs
        # Compute ADE: min over K samples of mean error across T frames
        # Compute FDE: min over K samples of final frame error
    # Average over all agents in scene
```

**Results Format**:

For ETH/UCY:
| Dataset | ADE  | FDE  |
|---------|------|------|
| ETH     | 0.45 | 0.75 |
| Hotel   | 0.14 | 0.22 |
| Univ    | 0.25 | 0.45 |
| Zara1   | 0.18 | 0.30 |
| Zara2   | 0.14 | 0.24 |
| Avg     | 0.23 | 0.39 |

For nuScenes:
| Metric | ADE_5 | FDE_5 | ADE_10 | FDE_10 |
|--------|-------|-------|--------|--------|
| Value  | 1.856 | 3.889 | 1.452  | 2.856  |

---

## Data Processing

### Data Generator (`data/dataloader.py`)

**Purpose**: Provides batched trajectory data for training/testing.

**Key Features**:
- Lazy loading: loads data on-demand per frame
- Supports train/val/test splits
- Handles multiple sequences
- Frame skipping support for temporal downsampling

**Architecture**:
```python
class data_generator:
    def __init__(self, parser, log, split='train', phase='training'):
        # Load dataset splits
        # Initialize preprocessors for each sequence
        # Count total samples

    def __call__(self):
        # Get next sample
        # Return preprocessed data dict
```

**Data Flow**:
1. Select sample index from shuffled list
2. Map to (sequence, frame) pair
3. Call sequence preprocessor with frame index
4. Return dict with past/future trajectories, masks, metadata

### Preprocessor (`data/preprocessor.py`)

**Purpose**: Loads and preprocesses raw trajectory data for a sequence.

**Key Responsibilities**:
- Load raw trajectory files
- Extract past and future windows for each frame
- Handle missing data and agent filtering
- Normalize coordinates (scene-level and agent-level)
- Load map data if available
- Create data masks for valid trajectories

**Data Dictionary**:
```python
{
    'pre_motion_3D': List[Tensor],      # Past trajectories per agent
    'fut_motion_3D': List[Tensor],      # Future trajectories per agent
    'fut_motion_mask': List[Tensor],    # Valid frame masks
    'pre_motion_mask': List[Tensor],    # Valid frame masks
    'heading': List[float],             # Agent headings
    'seq': str,                         # Sequence name
    'frame': int,                       # Frame number
    'scene_map': Map,                   # Map object (if available)
    'traj_scale': float,                # Coordinate scaling factor
}
```

### Dataset Formats

#### ETH/UCY Format (`datasets/eth_ucy/`)

Text files with space-separated values:
```
frame_id agent_id x y [additional columns...]
```

**Preprocessing** (`data/convert_ethucy.py`):
- Converts from original ETH/UCY format
- Splits into train/val/test files
- Standardizes coordinate systems

#### nuScenes Format (`datasets/nuscenes_pred/`)

Processed from nuScenes prediction challenge format:
- JSON files with scene metadata
- Trajectory annotations in global coordinates
- HD maps for scene context

**Preprocessing** (`data/process_nuscenes.py`):
```bash
python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>
```

### Coordinate Systems and Normalization

**Three Normalization Schemes**:

1. **Scene Normalization** (`model/agentformer.py:522-526`):
   ```python
   scene_orig = pre_motion[-1].mean(dim=0)  # Center of all agents at last timestep
   motion_scene_norm = motion - scene_orig
   ```

2. **Agent Normalization** (`model/agentformer.py:548-549`):
   ```python
   cur_motion = pre_motion[-1]  # Last observed position per agent
   motion_norm = motion - cur_motion
   ```

3. **Velocity** (`model/agentformer.py:545-546`):
   ```python
   pre_vel = pre_motion[1:] - pre_motion[:-1]
   fut_vel = fut_motion - cat([pre_motion[-1], fut_motion[:-1]])
   ```

**Heading Alignment** (`model/agentformer.py:138, 550-551`):

Optional rotation of velocities to agent-centric frame:
```python
heading_vec = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
vel_aligned = rotation_2d_torch(vel, -heading)
```

---

## Loss Functions

### AgentFormer VAE Losses (`model/agentformer_loss.py`)

#### 1. MSE Loss (Reconstruction)

**Purpose**: Ensure predictions match ground truth trajectories.

```python
def compute_motion_mse(data, cfg):
    diff = data['fut_motion_orig'] - data['train_dec_motion']
    if cfg.get('mask', True):
        mask = data['fut_mask']
        diff *= mask.unsqueeze(2)  # Apply valid frame mask
    loss_unweighted = diff.pow(2).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= diff.shape[0]  # Normalize by number of frames
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted
```

**Configuration**:
```yaml
mse:
  weight: 1.0
  mask: true      # Mask invalid frames
  normalize: true # Normalize by frame count
```

#### 2. KL Divergence Loss

**Purpose**: Regularize latent distribution to match prior.

```python
def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)  # Prevent collapse
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted
```

**Configuration**:
```yaml
kld:
  weight: 1.0
  min_clip: 2.0   # Minimum KL value to prevent posterior collapse
  normalize: true
```

**KL Annealing** (for discrete z):
- Temperature parameter τ annealed from high to low
- Helps training stability for categorical latent variables

#### 3. Sample Loss (Best-of-K)

**Purpose**: Ensure at least one sample is close to ground truth.

```python
def compute_sample_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)  # Sum over time and coordinates
    loss_unweighted = dist.min(dim=1)[0]  # Best sample per agent
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted
```

**Configuration**:
```yaml
sample:
  weight: 1.0
  k: 20           # Number of samples
  mask: true
  normalize: true
```

### DLow Losses (`model/dlow.py:11-50`)

#### 1. KL Divergence (Q-network)

Similar to AgentFormer but between Q-network output and prior:
```python
loss = q_z_dist_dlow.kl(p_z_dist_infer).sum()
```

#### 2. Diversity Loss

**Purpose**: Encourage diverse predictions across K samples.

```python
def diversity_loss(data, cfg):
    loss_unweighted = 0
    fut_motions = data['infer_dec_motion'].view(*shape[:2], -1)
    for motion in fut_motions:  # Per agent
        dist = F.pdist(motion, 2) ** 2  # Pairwise distances between K samples
        loss_unweighted += (-dist / cfg['d_scale']).exp().mean()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted
```

Penalizes samples that are too similar (small pairwise distances).

**Configuration**:
```yaml
diverse:
  weight: 20.0
  d_scale: 10.0   # Scale factor for distance
```

#### 3. Reconstruction Loss

Best-of-K matching to ground truth (same as AgentFormer sample loss).

**Configuration**:
```yaml
recon:
  weight: 5.0
```

---

## Configuration System

### Configuration Management (`utils/config.py`)

**Purpose**: Load and manage YAML configuration files.

**Key Features**:
- Loads base config from YAML file
- Creates result directories automatically
- Tracks experiment metadata
- Supports temporary configs (for testing)

**Usage**:
```python
cfg = Config('eth_agentformer', tmp=False, create_dirs=True)
```

**Directory Structure Created**:
```
results/<cfg_name>_<timestamp>/
├── models/          # Saved checkpoints
├── tf_logs/         # TensorBoard logs
├── log.txt          # Training log
├── log_eval.txt     # Evaluation log
└── <cfg_name>.yml   # Copy of config
```

### Configuration File Structure

**Example**: `cfg/eth_ucy/eth/eth_agentformer_pre.yml`

```yaml
# General
description: AgentFormer (VAE)
seed: 1
dataset: eth

# Data paths
data_root_ethucy: datasets/eth_ucy
data_root_nuscenes_pred: datasets/nuscenes_pred

# Temporal windows
past_frames: 8
future_frames: 12
min_past_frames: 8
min_future_frames: 12

# Preprocessing
traj_scale: 2              # Coordinate scaling
motion_dim: 2              # Trajectory dimensionality (x, y)
forecast_dim: 2

# Model architecture
model_id: agentformer
tf_model_dim: 512          # Transformer hidden dimension
tf_ff_dim: 2048            # Feedforward dimension
tf_nhead: 8                # Number of attention heads
tf_dropout: 0.1
nz: 32                     # Latent dimension
z_type: gaussian           # gaussian or discrete

# Input/output types
input_type: [pos, vel, norm]
fut_input_type: [pos, vel, norm]
dec_input_type: [heading]
pred_type: vel

# Agent-aware attention
tf_cfg:
  gaussian_kernel: true
  sep_attn: true           # Separate inter/intra-agent attention

# Positional encoding
pos_concat: false
use_agent_enc: true
agent_enc_learn: true
agent_enc_shuffle: true

# Training
lr: 1.e-4
num_epochs: 30
lr_fix_epochs: 10
lr_scheduler: linear
print_freq: 20
model_save_freq: 5

# Loss weights
loss_cfg:
  mse:
    weight: 1.0
  kld:
    weight: 1.0
    min_clip: 2.0
```

### Key Configuration Parameters

**Transformer Configuration** (`tf_cfg`):
- `gaussian_kernel`: Use Gaussian kernel for attention (distance-based)
- `sep_attn`: Enable agent-aware attention (separate self/inter)

**Encoder/Decoder Layers**:
- `context_encoder.nlayer`: Number of encoder layers (default: 6)
- `future_encoder.nlayer`: Number of future encoder layers (default: 6)
- `future_decoder.nlayer`: Number of decoder layers (default: 6)

**Prediction Configuration**:
- `pred_type`: Output type (vel, pos, scene_norm)
- `ar_detach`: Detach gradients in autoregressive loop
- `ar_train`: Use autoregressive decoding during training

**Map Configuration** (if `use_map: true`):
```yaml
use_map: true
map_encoder:
  model_id: map_cnn
  normalize: true
  dropout: 0.0
```

---

## Code Quality & Design Patterns

### Strengths

1. **Modular Architecture**:
   - Clear separation: data loading, model, training, evaluation
   - Reusable components (MLPs, attention layers, distributions)
   - Model registry pattern for easy experimentation

2. **Flexible Configuration System**:
   - YAML-based configs for all hyperparameters
   - Easy to create new experiments
   - Automatic result directory management

3. **Clean Abstraction**:
   - Data generator abstraction works for multiple datasets
   - Unified preprocessing interface
   - Consistent model interface (set_data, forward, inference)

4. **Research-Friendly**:
   - Easy to swap components (attention types, loss functions)
   - Extensive configuration options
   - Clear separation of training stages

5. **Documentation**:
   - Detailed README with setup instructions
   - Code comments for complex sections
   - Example configurations provided

### Areas for Improvement

1. **Type Hints**:
   - Limited use of Python type hints
   - Could improve code clarity and catch errors

2. **Testing**:
   - No unit tests or integration tests
   - Could benefit from test coverage

3. **Error Handling**:
   - Limited validation of configuration parameters
   - Some error messages could be more descriptive

4. **Code Duplication**:
   - Some repeated patterns in encoder/decoder initialization
   - Could use more inheritance or composition

5. **Magic Numbers**:
   - Some hardcoded values (e.g., coordinate indices [13, 15])
   - Could be extracted to constants

### Design Patterns Used

1. **Factory Pattern**: Model registry in `model_lib.py`
   ```python
   model_dict = {
       'agentformer': AgentFormer,
       'dlow': DLow
   }
   model = model_dict[model_id](cfg)
   ```

2. **Strategy Pattern**: Multiple loss functions registered in dict
   ```python
   loss_func = {
       'mse': compute_motion_mse,
       'kld': compute_z_kld,
       'sample': compute_sample_loss
   }
   ```

3. **Builder Pattern**: Configuration system builds models with many options

4. **Template Method**: Preprocessor base class with dataset-specific implementations

5. **Iterator Pattern**: Data generator with `__call__` method

### Code Organization Best Practices

1. **Single Responsibility**: Each module has clear purpose
2. **DRY Principle**: Common utilities extracted to utils/
3. **Configuration over Code**: Behavior controlled by configs
4. **Separation of Concerns**: Model logic separate from training/data

---

## Technical Implementation Details

### Memory Management

**Agent Limiting** (`model/agentformer.py:506-512`):
- Randomly sample `max_train_agent` agents if scene has too many
- Prevents OOM errors on crowded scenes
- Default: 100 agents max

### Numerical Stability

1. **KL Clipping** (`model/agentformer_loss.py:19`):
   ```python
   loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
   ```
   Prevents posterior collapse in VAE training.

2. **Gradient Detachment** (`model/agentformer.py:356-359`):
   ```python
   if self.ar_detach:
       out_in = seq_out[-agent_num:].clone().detach()
   ```
   Optional detachment in autoregressive loop for stability.

3. **Log-Variance Parameterization**:
   ```python
   logvar = (A ** 2 + 1e-8).log()  # Add epsilon for numerical stability
   ```

### Coordinate Transformations

**Rotation** (`utils/torch.py` - assumed):
```python
def rotation_2d_torch(x, theta, origin=None):
    # Rotate 2D coordinates by angle theta around origin
    # Used for data augmentation and heading alignment
```

**Homography Warping** (`data/homography_warper.py`):
- Transforms coordinates between camera and world space
- Used for ETH/UCY dataset preprocessing

---

## Performance Considerations

### Computational Complexity

**Attention Complexity**: O(T² × N² × D) per layer
- T: sequence length
- N: number of agents
- D: model dimension

**Optimizations**:
- Agent masking reduces N² to only connected agents
- Batch processing across independent scenes
- Optional connection distance threshold

### Training Time

- **Stage 1 (AgentFormer VAE)**: ~30 epochs, several hours on single GPU
- **Stage 2 (DLow)**: ~50 epochs, similar time (most parameters frozen)

### Inference Speed

- Single scene: ~50-100ms on GPU
- Bottleneck: Autoregressive decoding (T_future sequential steps)
- Batch inference speeds up evaluation

---

## Summary

AgentFormer is a well-structured implementation of a transformer-based multi-agent trajectory forecasting model. The codebase demonstrates:

**Key Innovations**:
- Agent-aware attention mechanism for modeling social interactions
- Two-stage training with VAE and DLow for diverse predictions
- Support for multiple datasets and modalities (trajectories + maps)

**Implementation Quality**:
- Modular and extensible architecture
- Comprehensive configuration system
- Clean separation of concerns
- Research-friendly design

**Use Cases**:
- Pedestrian trajectory prediction
- Vehicle motion forecasting
- Multi-agent simulation
- Social robotics

The code is production-ready for research purposes and provides a solid foundation for extensions and improvements in multi-agent forecasting tasks.

---

## References

1. [AgentFormer Paper (ICCV 2021)](https://arxiv.org/abs/2103.14023)
2. [Project Website](https://www.ye-yuan.com/agentformer)
3. [GitHub Repository](https://github.com/Khrylx/AgentFormer)
4. [ETH/UCY Datasets](https://paperswithcode.com/dataset/eth)
5. [nuScenes Dataset](https://www.nuscenes.org/)