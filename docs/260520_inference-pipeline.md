# AgentFormer Inference Pipeline Analysis

**Date**: May 20, 2026
**Document**: Comprehensive analysis of the test.py inference pipeline and result file organization

---

## Table of Contents

1. [Overview](#overview)
2. [Inference Pipeline Flow](#inference-pipeline-flow)
3. [Prediction Generation Process](#prediction-generation-process)
4. [Result File Organization](#result-file-organization)
5. [Data Structures](#data-structures)
6. [Model Inference Modes](#model-inference-modes)
7. [Coordinate System and Scaling](#coordinate-system-and-scaling)
8. [Complete Pipeline Walkthrough](#complete-pipeline-walkthrough)
9. [Result File Formats](#result-file-formats)
10. [Evaluation Integration](#evaluation-integration)

---

## Overview

The `test.py` script implements the inference pipeline for AgentFormer, generating trajectory predictions for test datasets. The pipeline supports:

- **Two inference modes**: Reconstruction (deterministic) and Sampling (stochastic)
- **Multiple predictions per scene**: K diverse trajectory samples
- **Multiple agents per frame**: Joint multi-agent prediction
- **Organized output structure**: Separate directories for GT, reconstruction, and samples

**Key Features**:
- No gradient computation (inference only)
- Batch processing across all test frames
- Automatic evaluation after prediction generation
- Optional cleanup mode to save disk space

---

## Inference Pipeline Flow

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

1. Setup
   ├── Load configuration
   ├── Initialize model
   ├── Load checkpoint
   └── Set device and disable gradients

2. Data Loading
   ├── Create data generator for test split
   └── Iterate over all frames in dataset

3. For Each Frame:
   ├── Load past trajectories (8 frames)
   ├── Load future trajectories (12 frames) - GT only
   ├── Identify valid agents
   └── Prepare data dictionary

4. Model Inference
   ├── Set data in model
   ├── Run reconstruction mode (1 sample)
   ├── Run inference mode (K samples)
   └── Scale predictions back to original coordinates

5. Save Predictions
   ├── Save ground truth trajectories
   ├── Save reconstruction (deterministic)
   └── Save K stochastic samples

6. Evaluation
   ├── Run eval.py on saved predictions
   └── Compute ADE/FDE metrics

7. Cleanup (Optional)
   └── Remove prediction files to save space
```

### Code Flow in test.py

**Main Script** (`test.py:91-146`):

```python
if __name__ == '__main__':
    # 1. Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)           # Config file
    parser.add_argument('--data_eval', default='test')   # train/val/test
    parser.add_argument('--epochs', default=None)        # Epoch(s) to evaluate
    parser.add_argument('--gpu', type=int, default=0)    # GPU device
    parser.add_argument('--cached', action='store_true') # Skip inference
    parser.add_argument('--cleanup', action='store_true')# Remove files after eval

    # 2. Setup
    cfg = Config(args.cfg)
    epochs = [cfg.get_last_epoch()] if args.epochs is None else [int(x) for x in args.epochs.split(',')]
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 else torch.device('cpu')
    torch.set_grad_enabled(False)  # IMPORTANT: Disable gradients

    # 3. Loop over epochs
    for epoch in epochs:
        # 4. Load model
        model = model_dict[model_id](cfg)
        model.set_device(device)
        model.eval()
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'], strict=False)

        # 5. Run inference
        for split in data_splits:
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'
            test_model(generator, save_dir, cfg)

            # 6. Evaluate
            eval_dir = f'{save_dir}/samples'
            subprocess.run(f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split}")

            # 7. Cleanup
            if args.cleanup:
                shutil.rmtree(save_dir)
```

---

## Prediction Generation Process

### Step 1: Get Model Predictions (`test.py:16-21`)

```python
def get_model_prediction(data, sample_k):
    """
    Generate predictions for a single frame

    Args:
        data: Dictionary containing past trajectories and metadata
        sample_k: Number of stochastic samples to generate

    Returns:
        recon_motion_3D: [N_agents, T_future, 2] - Deterministic prediction
        sample_motion_3D: [K, N_agents, T_future, 2] - K stochastic predictions
    """
    model.set_data(data)

    # Reconstruction mode: uses posterior q(z|past,future) mode (during training)
    # At test time, uses learned prior p(z) or mean of distribution
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

    # Inference mode: samples from prior p(z) multiple times for diversity
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)

    # Transpose to [K, N_agents, T_future, 2]
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

    return recon_motion_3D, sample_motion_3D
```

**Key Points**:
- `recon`: Single deterministic prediction (best-guess)
- `infer`: K diverse stochastic predictions by sampling latent codes
- Predictions are in normalized coordinates (scaled by `traj_scale`)

### Step 2: Model Inference Methods

#### AgentFormer.inference() (`model/agentformer.py:599-608`)

```python
def inference(self, mode='infer', sample_num=20, need_weights=False):
    """
    Run inference through the model

    Args:
        mode: 'recon' for reconstruction, 'infer' for sampling
        sample_num: Number of samples (K)
        need_weights: Whether to return attention weights

    Returns:
        predictions: [N_agents, K, T_future, 2] if mode='infer'
                    [N_agents, T_future, 2] if mode='recon'
        data: Full data dictionary with intermediate computations
    """
    # Encode map features if needed
    if self.use_map and self.data['map_enc'] is None:
        self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])

    # Encode past trajectories to context
    if self.data['context_enc'] is None:
        self.context_encoder(self.data)

    # For reconstruction: use future encoder to get latent mode
    if mode == 'recon':
        sample_num = 1
        self.future_encoder(self.data)  # Learn q(z|past,future) - not available at test time

    # Decode future trajectories
    self.future_decoder(self.data, mode=mode, sample_num=sample_num,
                       autoregress=True, need_weights=need_weights)

    return self.data[f'{mode}_dec_motion'], self.data
```

#### DLow.inference() (`model/dlow.py:122-127`)

```python
def inference(self, mode, sample_num, need_weights=False):
    """
    DLow inference: uses Q-network to generate diverse latent codes

    Args:
        mode: 'recon' or 'infer'
        sample_num: Number of samples (K)

    Returns:
        predictions: [N_agents, K, T_future, 2] or [N_agents, T_future, 2]
    """
    # Use mean of Q-network distribution (deterministic for evaluation)
    self.main(mean=True, need_weights=need_weights)

    # Extract predictions
    res = self.data[f'infer_dec_motion']

    # For reconstruction mode, take only first sample
    if mode == 'recon':
        res = res[:, 0]

    return res, self.data
```

**Difference between AgentFormer and DLow**:
- **AgentFormer**: Samples latent z from learned prior p(z)
- **DLow**: Uses Q-network to generate K diverse latent codes deterministically

### Step 3: Main Test Loop (`test.py:56-89`)

```python
def test_model(generator, save_dir, cfg):
    """
    Run inference on entire test set

    Args:
        generator: Data generator for test split
        save_dir: Root directory to save results
        cfg: Configuration object
    """
    total_num_pred = 0

    # Iterate over all frames in test set
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue

        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d\r' % (seq_name, frame))

        # Prepare ground truth (for saving only, not used in prediction)
        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale

        # Generate predictions (no gradients)
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)

        # Scale back to original coordinates
        recon_motion_3D = recon_motion_3D * cfg.traj_scale
        sample_motion_3D = sample_motion_3D * cfg.traj_scale

        # Create output directories
        recon_dir = os.path.join(save_dir, 'recon')
        sample_dir = os.path.join(save_dir, 'samples')
        gt_dir = os.path.join(save_dir, 'gt')

        # Save all predictions
        for i in range(sample_motion_3D.shape[0]):  # K samples
            save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
        save_prediction(recon_motion_3D, data, '', recon_dir)
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir)

        total_num_pred += num_pred

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)

    # Sanity check for nuScenes
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {'train': 32186, 'val': 8560, 'test': 9041}
        assert total_num_pred == scene_num[generator.split]
```

### Step 4: Save Predictions (`test.py:23-54`)

```python
def save_prediction(pred, data, suffix, save_dir):
    """
    Save predictions for a single frame to disk

    Args:
        pred: [N_agents, T_future, 2] - Predicted trajectories
        data: Data dictionary with metadata
        suffix: File suffix (e.g., '/sample_000' or '')
        save_dir: Base directory to save results

    Returns:
        pred_num: Number of agents with predictions saved
    """
    pred_num = 0
    pred_arr = []

    # Extract metadata
    fut_data = data['fut_data']      # List of future frame data
    seq_name = data['seq']           # Sequence name (e.g., 'biwi_eth')
    frame = data['frame']            # Current frame number
    valid_id = data['valid_id']      # List of valid agent IDs
    pred_mask = data['pred_mask']    # Mask for agents to predict (nuScenes only)

    # Process each agent
    for i in range(len(valid_id)):
        identity = valid_id[i]

        # Skip if not in prediction mask (nuScenes only)
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        # Build trajectory for this agent across future frames
        for j in range(cfg.future_frames):  # T_future timesteps
            # Get original data for this frame (contains full state info)
            cur_data = fut_data[j]

            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                # Agent exists in original data, use that as template
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                # Agent missing in original data, use most recent
                data = most_recent_data.copy()
                data[0] = frame + j + 1  # Update frame number

            # Replace position with prediction
            data[[13, 15]] = pred[i, j].cpu().numpy()  # Indices [13, 15] = x, z position
            most_recent_data = data.copy()
            pred_arr.append(data)

        pred_num += 1

    # Save to file
    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        # Extract only: frame_id, agent_id, x, z
        indices = [0, 1, 13, 15]
        pred_arr = pred_arr[:, indices]

        # Construct filename
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")

    return pred_num
```

**Important Details**:

1. **Column Indices** (`test.py:41, 48`):
   - Index `[13, 15]`: x and z positions in the original data format
   - Index `0`: Frame ID
   - Index `1`: Agent ID
   - These correspond to the nuScenes/ETH-UCY data format

2. **Template Mechanism**:
   - Uses original future data as template to preserve metadata
   - Only replaces position coordinates with predictions
   - Handles missing agents by copying most recent data

3. **File Structure**:
   - GT and Recon: Single file per frame
   - Samples: Directory per frame with K files

---

## Result File Organization

### Directory Structure

```
results/
└── <config_name>/                    # e.g., eth_agentformer
    ├── log/
    │   ├── log_test.txt              # Test execution log
    │   └── log_eval.txt              # Evaluation results
    ├── models/
    │   ├── model_0005.p              # Model checkpoint epoch 5
    │   ├── model_0010.p              # Model checkpoint epoch 10
    │   └── ...
    └── results/
        └── epoch_XXXX/               # e.g., epoch_0005
            └── <split>/              # train/val/test
                ├── gt/               # Ground truth trajectories
                │   └── <seq_name>/   # e.g., biwi_eth
                │       ├── frame_000087.txt
                │       ├── frame_000088.txt
                │       └── ...
                ├── recon/            # Deterministic predictions
                │   └── <seq_name>/
                │       ├── frame_000087.txt
                │       ├── frame_000088.txt
                │       └── ...
                └── samples/          # Stochastic predictions
                    └── <seq_name>/
                        ├── frame_000087/
                        │   ├── sample_000.txt
                        │   ├── sample_001.txt
                        │   ├── ...
                        │   └── sample_019.txt  # K=20 samples
                        ├── frame_000088/
                        │   ├── sample_000.txt
                        │   └── ...
                        └── ...
```

### Example Paths

**For ETH dataset, epoch 5, test split, frame 868**:

- **GT**: `results/eth_agentformer/results/epoch_0005/test/gt/biwi_eth/frame_000868.txt`
- **Recon**: `results/eth_agentformer/results/epoch_0005/test/recon/biwi_eth/frame_000868.txt`
- **Samples**: `results/eth_agentformer/results/epoch_0005/test/samples/biwi_eth/frame_000868/sample_XXX.txt`

### File Naming Convention

| Type | File Pattern | Description |
|------|-------------|-------------|
| GT | `frame_{frame_id:06d}.txt` | Ground truth, single file per frame |
| Recon | `frame_{frame_id:06d}.txt` | Reconstruction, single file per frame |
| Samples | `frame_{frame_id:06d}/sample_{k:03d}.txt` | K samples in subdirectory |

**Examples**:
- `frame_000087.txt` - Frame 87
- `frame_001234.txt` - Frame 1234
- `sample_000.txt` - Sample 0 (out of K=20)
- `sample_019.txt` - Sample 19 (last sample)

---

## Data Structures

### Input Data Dictionary (from preprocessor)

**Generated by** `data/preprocessor.py:176-192`

```python
data = {
    # Trajectories
    'pre_motion_3D': List[Tensor],          # List of [T_past, 2] per agent
    'fut_motion_3D': List[Tensor],          # List of [T_future, 2] per agent
    'fut_motion_mask': List[Tensor],        # List of [T_future] masks per agent
    'pre_motion_mask': List[Tensor],        # List of [T_past] masks per agent

    # Raw data (for save_prediction)
    'pre_data': List[ndarray],              # List of past frame data
    'fut_data': List[ndarray],              # List of future frame data

    # Metadata
    'valid_id': List[float],                # Agent IDs that have complete trajectories
    'heading': List[float] or None,         # Agent headings (nuScenes only)
    'pred_mask': ndarray or None,           # Prediction mask (nuScenes only)
    'seq': str,                             # Sequence name (e.g., 'biwi_eth')
    'frame': int,                           # Current frame number
    'traj_scale': float,                    # Coordinate scaling factor
    'scene_map': GeometricMap or None,      # Scene map object
}
```

### Model Prediction Outputs

**Tensor Shapes**:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `recon_motion_3D` | `[N_agents, T_future, 2]` | Deterministic prediction |
| `sample_motion_3D` | `[K, N_agents, T_future, 2]` | K stochastic samples |
| `gt_motion_3D` | `[N_agents, T_future, 2]` | Ground truth |

**Coordinate System**:
- Predictions initially in normalized space (÷ `traj_scale`)
- Scaled back to original coordinates before saving (× `traj_scale`)
- Saved in meters (ETH/UCY) or dataset-specific units

### Preprocessor Raw Data Format

**Original data columns** (from nuScenes/ETH-UCY files):

```
Index   Column          Description
0       frame_id        Frame number
1       agent_id        Agent/pedestrian ID
2       class_id        Object class (1=Pedestrian, 2=Car, etc.)
...     ...             Additional metadata
13      x               X position (horizontal)
14      y               Y position (vertical/height)
15      z               Z position (depth)
16      heading         Agent heading angle (radians)
...     ...             Additional state information
```

**Used Indices**:
- `[0, 1, 13, 15]`: frame_id, agent_id, x, z (saved to result files)
- `[16]`: heading (for nuScenes, used in prediction)

---

## Model Inference Modes

### Mode 1: Reconstruction (`mode='recon'`)

**Purpose**: Generate a single deterministic "best guess" prediction.

**Process**:

```
Past Trajectories → Context Encoder → Context Encoding
                                            ↓
                            Future Encoder (if available)
                                            ↓
                                    Latent z (mode/mean)
                                            ↓
                            Future Decoder → Reconstruction
```

**For AgentFormer** (`model/agentformer.py:604-606`):
1. Encode past trajectories to context
2. Run future encoder to get posterior q(z|past, future)
   - **Note**: At test time, future is not available, so this uses learned prior
3. Use mode/mean of distribution (not sampling)
4. Decode to single trajectory

**For DLow** (`model/dlow.py:125-126`):
1. Encode past trajectories to context
2. Generate K latent codes from Q-network (mean mode)
3. Take first sample as reconstruction

**Output Shape**: `[N_agents, T_future, 2]`

**Use Case**: Best single prediction, comparison baseline

### Mode 2: Inference (`mode='infer'`)

**Purpose**: Generate K diverse stochastic predictions.

**Process**:

```
Past Trajectories → Context Encoder → Context Encoding
                                            ↓
                                    Sample z ~ p(z)  (K times)
                                            ↓
                            Future Decoder → K Predictions
```

**For AgentFormer** (`model/agentformer.py:419-420`):
1. Encode past trajectories to context
2. Sample K different latent codes from prior p(z)
3. Decode each latent code to a trajectory
4. Return K diverse predictions

**For DLow** (`model/dlow.py:122-124`):
1. Encode past trajectories to context
2. Q-network generates K latent codes (affine transformation of noise)
3. Decode all K latent codes
4. Return K diverse predictions

**Output Shape**: `[N_agents, K, T_future, 2]`

**Use Case**: Capture multimodal futures, uncertainty quantification

### Comparison: AgentFormer vs. DLow

| Aspect | AgentFormer | DLow |
|--------|-------------|------|
| Latent Sampling | Random sampling from p(z) | Q-network maps context to diverse z |
| Training | Direct VAE training | Two-stage: VAE then Q-network |
| Diversity | Inherent from prior p(z) | Explicitly optimized via diversity loss |
| Determinism | Stochastic (different samples each run) | Deterministic given context |
| Inference Speed | Fast (direct sampling) | Fast (forward pass through Q-net) |

---

## Coordinate System and Scaling

### Trajectory Scaling

**Purpose**: Normalize coordinates for better neural network training.

**Configuration** (e.g., `cfg/eth_ucy/eth/eth_agentformer.yml`):
```yaml
traj_scale: 2
```

**Usage**:

1. **During Preprocessing** (`data/preprocessor.py:124, 144`):
   ```python
   # Divide by traj_scale when loading
   pos = original_pos / self.traj_scale
   ```

2. **During Inference** (`test.py:67, 70`):
   ```python
   # Ground truth
   gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale

   # Predictions
   recon_motion_3D = recon_motion_3D * cfg.traj_scale
   sample_motion_3D = sample_motion_3D * cfg.traj_scale
   ```

**Effect**:
- Model works in normalized space (÷ traj_scale)
- Predictions scaled back to original coordinates before saving
- Loss computed in normalized space (smaller gradients)

### Coordinate Frames

**Original Coordinates** (from dataset):
- ETH/UCY: Meters in world frame
- nuScenes: Meters in global frame

**Model Coordinates** (internal):
- Normalized by `traj_scale`
- Optional scene normalization (subtract scene center)
- Optional rotation augmentation during training

**Output Coordinates** (saved files):
- Same as original coordinates
- Directly comparable to ground truth
- Used for evaluation metrics

---

## Complete Pipeline Walkthrough

### Example: Testing frame 868 from ETH dataset

**Command**:
```bash
python test.py --cfg eth_agentformer --gpu 0 --data_eval test
```

**Step-by-Step Execution**:

#### 1. Setup Phase

```python
# Load configuration
cfg = Config('eth_agentformer')  # Loads cfg/eth_ucy/eth/eth_agentformer.yml

# Configuration values:
cfg.past_frames = 8
cfg.future_frames = 12
cfg.sample_k = 20
cfg.traj_scale = 2
cfg.dataset = 'eth'
cfg.model_id = 'dlow' or 'agentformer'

# Initialize model
model = DLow(cfg) if cfg.model_id == 'dlow' else AgentFormer(cfg)
model.set_device(torch.device('cuda', 0))
model.eval()

# Load checkpoint
checkpoint = torch.load('results/eth_agentformer/models/model_0050.p')
model.load_state_dict(checkpoint['model_dict'])

# Disable gradients
torch.set_grad_enabled(False)
```

#### 2. Data Loading

```python
# Create data generator
generator = data_generator(cfg, log, split='test', phase='testing')

# Generator loads test sequences: biwi_eth
# Total frames to process: 750 frames

# For frame 868:
data = generator()  # Calls preprocessor for frame 868
```

#### 3. Preprocessor Extracts Data

```python
# preprocessor.py processes frame 868

# Load past frames: 860-867 (8 frames)
pre_data = [
    gt[gt[:, 0] == 868],  # Frame 868 (current)
    gt[gt[:, 0] == 867],  # Frame 867
    gt[gt[:, 0] == 866],  # Frame 866
    ...
    gt[gt[:, 0] == 861],  # Frame 861
]

# Load future frames: 869-880 (12 frames)
fut_data = [
    gt[gt[:, 0] == 869],  # Frame 869
    gt[gt[:, 0] == 870],  # Frame 870
    ...
    gt[gt[:, 0] == 880],  # Frame 880
]

# Identify valid agents
valid_id = [171.0]  # Agent 171 has complete past and future

# Extract trajectories
pre_motion_3D = [Tensor([8, 2])]  # Past 8 positions for agent 171
fut_motion_3D = [Tensor([12, 2])] # Future 12 positions for agent 171

# Return data dictionary
data = {
    'pre_motion_3D': [Tensor([8, 2])],
    'fut_motion_3D': [Tensor([12, 2])],
    'valid_id': [171.0],
    'seq': 'biwi_eth',
    'frame': 868,
    'traj_scale': 2.0,
    'fut_data': [array(...) for each future frame],
    ...
}
```

#### 4. Model Prediction

```python
# test.py:69 - Get predictions
recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)

# Inside get_model_prediction:
model.set_data(data)  # Preprocess and set data in model

# Reconstruction mode
recon_motion_3D, _ = model.inference(mode='recon', sample_num=20)
# Shape: [1, 12, 2] - 1 agent, 12 frames, 2D position

# Inference mode
sample_motion_3D, _ = model.inference(mode='infer', sample_num=20)
# Shape: [1, 20, 12, 2] before transpose
sample_motion_3D = sample_motion_3D.transpose(0, 1)
# Shape: [20, 1, 12, 2] after transpose - 20 samples, 1 agent, 12 frames, 2D
```

#### 5. Model Inference Details

**AgentFormer.inference('recon')**:
```python
# 1. Encode past trajectories
self.context_encoder(self.data)
# Creates context_enc: [8, 1, 512] - 8 past frames, 1 agent, 512 dimensions

# 2. Future encoder (uses ground truth during training, but at test uses prior)
self.future_encoder(self.data)
# Computes q_z_dist (posterior) - but at test time, uses learned prior

# 3. Decode using latent mode (not sampling)
self.future_decoder(self.data, mode='recon', sample_num=1)
# Autoregressively generates 12 future frames
# Output: [1, 12, 2]
```

**AgentFormer.inference('infer')**:
```python
# 1. Encode past (same as recon)
self.context_encoder(self.data)

# 2. Skip future encoder (not available at test time)

# 3. Decode with K samples
self.future_decoder(self.data, mode='infer', sample_num=20)
# Samples 20 different z ~ p(z)
# Decodes each to get 20 diverse trajectories
# Output: [1, 20, 12, 2]
```

#### 6. Scale Predictions

```python
# test.py:70 - Scale back to original coordinates
gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0) * 2.0
recon_motion_3D = recon_motion_3D * 2.0
sample_motion_3D = sample_motion_3D * 2.0

# Now in meters (original coordinates)
```

#### 7. Save Predictions

**Ground Truth** (`test.py:79`):
```python
save_prediction(gt_motion_3D, data, '', gt_dir)
# Creates: results/eth_agentformer/results/epoch_0050/test/gt/biwi_eth/frame_000868.txt
# Content: 12 lines (12 future frames for agent 171)
```

**Reconstruction** (`test.py:78`):
```python
save_prediction(recon_motion_3D, data, '', recon_dir)
# Creates: results/eth_agentformer/results/epoch_0050/test/recon/biwi_eth/frame_000868.txt
# Content: 12 lines (12 future frames for agent 171)
```

**Samples** (`test.py:76-77`):
```python
for i in range(20):  # K=20 samples
    save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
# Creates: results/eth_agentformer/results/epoch_0050/test/samples/biwi_eth/frame_000868/
#   ├── sample_000.txt (12 lines)
#   ├── sample_001.txt (12 lines)
#   ├── ...
#   └── sample_019.txt (12 lines)
```

#### 8. File Content Example

**GT File**: `gt/biwi_eth/frame_000868.txt`
```
869.000 171.000 2.600 8.010
870.000 171.000 3.040 7.990
871.000 171.000 3.500 8.060
...
880.000 171.000 6.960 8.400
```

**Recon File**: `recon/biwi_eth/frame_000868.txt`
```
869.000 171.000 2.559 7.931
870.000 171.000 2.893 7.967
871.000 171.000 3.233 8.029
...
880.000 171.000 5.803 8.527
```

**Sample File**: `samples/biwi_eth/frame_000868/sample_000.txt`
```
869.000 171.000 2.559 7.931
870.000 171.000 2.893 7.967
871.000 171.000 3.233 8.029
...
880.000 171.000 5.803 8.527
```

**Format**: `frame_id agent_id x z`
- frame_id: Future frame number (869-880)
- agent_id: Agent identifier (171.0)
- x, z: 2D position in meters

---

## Result File Formats

### Ground Truth Files

**Path**: `results/<config>/results/epoch_XXXX/<split>/gt/<seq_name>/frame_NNNNNN.txt`

**Format**: Space-separated values
```
<frame_id> <agent_id> <x> <z>
```

**Properties**:
- One line per (frame, agent) pair
- N_agents × T_future total lines per file
- Only includes agents with complete past and future trajectories

**Example** (3 agents, 12 future frames):
```
869.000 171.000 2.600 8.010
870.000 171.000 3.040 7.990
...
880.000 171.000 6.960 8.400
869.000 172.000 1.500 6.200
870.000 172.000 1.520 6.180
...
880.000 172.000 1.850 5.900
869.000 173.000 4.200 9.100
...
880.000 173.000 5.100 9.800
```
Total: 36 lines (3 agents × 12 frames)

### Reconstruction Files

**Path**: `results/<config>/results/epoch_XXXX/<split>/recon/<seq_name>/frame_NNNNNN.txt`

**Format**: Identical to ground truth files
```
<frame_id> <agent_id> <x> <z>
```

**Properties**:
- Single deterministic prediction per agent
- Same number of lines as ground truth file
- Directly comparable to GT for evaluation

### Sample Files

**Path**: `results/<config>/results/epoch_XXXX/<split>/samples/<seq_name>/frame_NNNNNN/sample_KKK.txt`

**Format**: Identical to ground truth files
```
<frame_id> <agent_id> <x> <z>
```

**Properties**:
- K files per frame (typically K=20)
- Each file contains one stochastic sample
- All K files have identical structure but different predictions
- Evaluation selects best of K for ADE/FDE

**File Organization**:
```
frame_000868/
├── sample_000.txt  <- First sample
├── sample_001.txt  <- Second sample
├── ...
└── sample_019.txt  <- 20th sample (K=20)
```

### Multi-Agent Frame Example

**Example**: Frame 293 with 3 agents

**GT File** (`gt/biwi_eth/frame_000293.txt`):
```
294.000 150.000 1.200 5.600
295.000 150.000 1.210 5.620
...
305.000 150.000 1.450 6.100
294.000 151.000 3.400 7.800
295.000 151.000 3.420 7.850
...
305.000 151.000 3.780 8.500
294.000 152.000 2.100 6.200
295.000 152.000 2.150 6.250
...
305.000 152.000 2.850 7.100
```
Total: 36 lines (3 agents × 12 frames)

**Sample File** (`samples/biwi_eth/frame_000293/sample_000.txt`):
- Same structure as GT
- Different x, z values (predictions)
- 36 lines total

### File Size Considerations

**Disk Space per Frame**:
- GT: ~0.5-2 KB (depends on number of agents)
- Recon: ~0.5-2 KB (same as GT)
- Samples: ~10-40 KB (K × size of GT)

**Example** (1 agent, 12 frames):
- GT: ~0.5 KB
- Recon: ~0.5 KB
- Samples: 20 × 0.5 KB = ~10 KB

**Total per Frame**: ~11 KB

**Full Test Set** (e.g., 750 frames):
- Total: ~8-10 MB

**Cleanup Mode** (`test.py:144-145`):
```python
if args.cleanup:
    shutil.rmtree(save_dir)  # Remove all predictions after evaluation
```
Saves disk space by keeping only evaluation metrics.

---

## Evaluation Integration

### Automatic Evaluation

**After generating all predictions** (`test.py:139-141`):

```python
log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
subprocess.run(cmd.split(' '))
```

**Evaluation Script**: `eval.py`
- Reads all sample files from `{save_dir}/samples/`
- Compares against ground truth files
- Computes ADE and FDE metrics
- Logs results to `log_eval.txt`

### Evaluation Process (eval.py)

**For each frame**:
1. Load ground truth: `gt/<seq>/<frame>.txt`
2. Load all samples: `samples/<seq>/<frame>/sample_*.txt`
3. For each agent:
   - Align predictions with GT by frame IDs
   - Compute error for each of K samples
   - Take minimum error across K (best-of-K)
4. Average across all agents and frames

**Metrics Computed**:

1. **ADE (Average Displacement Error)**:
   ```
   ADE = (1/N) Σ min_k mean_t ||pred_k(t) - gt(t)||_2
   ```
   - Average over all timesteps
   - Best of K samples per agent

2. **FDE (Final Displacement Error)**:
   ```
   FDE = (1/N) Σ min_k ||pred_k(T) - gt(T)||_2
   ```
   - Error at final timestep only
   - Best of K samples per agent

### Result Logging

**Log File**: `results/<config>/log/log_eval.txt`

**Example Output**:
```
-------------------------- STATS --------------------------
750 ADE: 0.4532
750 FDE: 0.7845
-----------------------------------------------------------
```

**Interpretation**:
- 750 frames evaluated
- ADE: 0.45 meters average error
- FDE: 0.78 meters final error

### Expected Performance (from README)

**ETH/UCY**:
| Dataset | ADE  | FDE  |
|---------|------|------|
| ETH     | 0.45 | 0.75 |
| Hotel   | 0.14 | 0.22 |
| Univ    | 0.25 | 0.45 |
| Zara1   | 0.18 | 0.30 |
| Zara2   | 0.14 | 0.24 |

**nuScenes**:
| Metric | Value |
|--------|-------|
| ADE_5  | 1.856 |
| FDE_5  | 3.889 |
| ADE_10 | 1.452 |
| FDE_10 | 2.856 |

---

## Command Line Options

### Basic Usage

```bash
python test.py --cfg <config_name> --gpu <gpu_id> --data_eval <split>
```

### All Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cfg` | str | None | Configuration file name (e.g., 'eth_agentformer') |
| `--data_eval` | str | 'test' | Data split to evaluate: train/val/test |
| `--epochs` | str | None | Comma-separated epoch numbers (e.g., '10,20,30') |
| `--gpu` | int | 0 | GPU device ID (-1 for CPU) |
| `--cached` | flag | False | Skip inference, only run evaluation on existing files |
| `--cleanup` | flag | False | Remove prediction files after evaluation |

### Examples

**Test latest epoch**:
```bash
python test.py --cfg eth_agentformer --gpu 0
```

**Test multiple epochs**:
```bash
python test.py --cfg eth_agentformer --epochs 10,20,30,40,50 --gpu 0
```

**Test on validation set**:
```bash
python test.py --cfg eth_agentformer --data_eval val --gpu 0
```

**Re-evaluate existing predictions**:
```bash
python test.py --cfg eth_agentformer --cached --gpu 0
```

**Test and cleanup (save disk space)**:
```bash
python test.py --cfg eth_agentformer --cleanup --gpu 0
```

**CPU inference**:
```bash
python test.py --cfg eth_agentformer --gpu -1
```

---

## Summary

The AgentFormer inference pipeline is a comprehensive system for generating and evaluating multi-agent trajectory predictions. Key characteristics:

### Pipeline Strengths

1. **Organized Output Structure**: Clear separation of GT, reconstruction, and samples
2. **Multi-Modal Predictions**: Generates K diverse samples per scene
3. **Automatic Evaluation**: Integrated metrics computation
4. **Memory Efficient**: Optional cleanup mode for disk space management
5. **Flexible Testing**: Supports multiple epochs, splits, and configurations

### Data Flow Summary

```
Dataset → Preprocessor → Data Generator → Model → Predictions → Files → Evaluation → Metrics
  ↓           ↓              ↓              ↓          ↓           ↓         ↓          ↓
Raw      Extract        Batch         Encode     Generate    Save      Compute    Report
Files    Trajectories   Frames        Context    Futures     Results   Errors     Numbers
```

### Result File Organization

- **GT**: Single file per frame with ground truth
- **Recon**: Single file per frame with deterministic prediction
- **Samples**: K files per frame (subdirectory) with stochastic predictions
- **Format**: `frame_id agent_id x z` (space-separated)
- **Evaluation**: Best-of-K ADE/FDE computed automatically

### Typical Inference Time

- **Single Frame**: ~50-100ms on GPU
- **Full Test Set (750 frames)**: ~1-2 minutes
- **With Evaluation**: Additional ~30 seconds

The pipeline is designed for research use, balancing flexibility, clarity, and efficiency for trajectory forecasting experiments.

---

## Appendix: Debugging Tips

### Verify Predictions Are Generated

```bash
# Check if files exist
ls results/eth_agentformer/results/epoch_0050/test/samples/

# Count frames processed
find results/eth_agentformer/results/epoch_0050/test/gt/ -name "*.txt" | wc -l

# Verify sample count
ls results/eth_agentformer/results/epoch_0050/test/samples/biwi_eth/frame_000868/ | wc -l
# Should output: 20 (for K=20)
```

### Check File Format

```bash
# View first few predictions
head results/eth_agentformer/results/epoch_0050/test/samples/biwi_eth/frame_000868/sample_000.txt

# Check line count (should be N_agents × T_future)
wc -l results/eth_agentformer/results/epoch_0050/test/gt/biwi_eth/frame_000868.txt
```

### Common Issues

1. **Missing predictions**: Check that model checkpoint exists and is loaded
2. **Wrong file count**: Verify cfg.sample_k matches expected number
3. **Coordinate mismatch**: Check traj_scale is correctly applied
4. **NaN values**: Check model training converged properly
