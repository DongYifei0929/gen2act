# Gen2Act Codebase Structure

This repository is small, so the implementation is split into a light package
instead of a large framework.

## Chosen Backbone

- Vision encoder: `ViT-B/16`-style backbone implemented with PyTorch modules
- Transformer stack: `torch.nn.TransformerEncoder` and `TransformerEncoderLayer`

Why this choice:

- It matches a widely used ViT configuration from the Torchvision ecosystem.
- It keeps the implementation dependency-light and easy to inspect.
- The encoder stack is standard PyTorch, so training, debugging, and export are straightforward.

References:

- Torchvision `vit_b_16`: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html
- PyTorch `TransformerEncoderLayer`: https://docs.pytorch.org/docs/main/generated/torch.nn.modules.transformer.TransformerEncoderLayer.html

## Directory Layout

```text
architecture.py
gen2act/
  __init__.py
  data/
    __init__.py
    hdf5_policy_dataset.py
  modeling/
    __init__.py
    vit.py
    resampler.py
    transformer.py
    track.py
    policy.py
configs/
  gen2act_policy.toml
scripts/
  train_policy.py
  infer_policy.py
```

## Module Responsibilities

- `gen2act/modeling/vit.py`
  - Patch embedding
  - ViT-B/16-style positional encoding
  - Transformer encoder blocks
  - Outputs patch tokens `[B, P, D]`

- `gen2act/modeling/resampler.py`
  - Perceiver-style compression
  - Converts variable-length frame tokens into fixed latent tokens `[B, K, D]`

- `gen2act/modeling/transformer.py`
  - Generic `TransformerEncoder` wrapper for fusion and other sequence processing

- `gen2act/modeling/track.py`
  - Training-only auxiliary trajectory predictor
  - Takes conditioning tokens plus point tracks and predicts future coordinates

- `gen2act/modeling/policy.py`
  - Composes the full policy
  - Encodes human video and robot history
  - Fuses both streams
  - Predicts discretized actions, terminate, and gripper logits
  - Exposes `build_default_policy()`

- `architecture.py`
  - Compatibility wrapper that re-exports the main classes for older imports

## Runtime Entry Points

- `configs/gen2act_policy.toml`
  - Central model, data, train, and inference defaults
- `scripts/train_policy.py`
  - HDF5-based behavior cloning training loop for the Gen2Act policy
- `scripts/infer_policy.py`
  - Loads a checkpoint and runs one-step policy inference on a demo window
- `gen2act/data/hdf5_policy_dataset.py`
  - Dataset adapter that turns Isaac Lab HDF5 demos into policy training windows

## Data Flow

1. A generated human video is encoded frame-by-frame by the ViT backbone.
2. A separate robot history clip is encoded with the same ViT.
3. Each token sequence is compressed by its own Perceiver resampler.
4. The compressed tokens are concatenated and processed by a transformer encoder.
5. The pooled context is mapped to:
   - action logits per dimension
   - terminate logits
   - gripper logits
6. During training, the auxiliary track predictor consumes the same latent tokens and predicts point trajectories.

## Tensor Shapes

- Human video input: `[B, 16/24, 3, 224, 224]`
- Robot history input: `[B, 8, 3, 224, 224]`
- ViT output per frame: `[B*T, P, D]`
- Resampler output: `[B, K, D]`
- Fused token sequence: `[B, K_h + K_r, D]`
- Pooled context: `[B, D]`
- Action logits: `[B, A, 256]`
- Terminate logits: `[B, 2]`
- Gripper logits: `[B, 2]`

## Notes

- The code is written to be easy to swap into a more complete robotics training loop.
- The video generator is intentionally not part of this package; it is treated as an external, frozen dependency.
- The auxiliary track predictor is optional at inference time.

### architecture graph:

```mermaid
graph TD
    %% 样式定义
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px;
    classDef moduleStyle fill:#bbf,stroke:#333,stroke-width:2px;
    classDef varStyle fill:#fff,stroke:#333,stroke-dasharray: 5 5;
    classDef outputStyle fill:#bfb,stroke:#333,stroke-width:2px;

    %% --- 1. 输入数据输入端 ---
    subgraph INPUT [输入端 Input]
        I0["场景初始图像 I₀"]:::inputStyle
        G["语言指令 G<br/>'Drag the Chair...'"]:::inputStyle
        VM["预训练视频生成模型<br/>(Video Model)"]:::moduleStyle
        Vg["生成的人类视频 V_g"]:::varStyle
        Ir["机器人历史观测 I_{t-k:t}"]:::inputStyle
    end

    I0 --> VM
    G --> VM
    VM --> Vg

    %% --- 2. 视觉特征提取 (模块1) ---
    subgraph M1 [1. ViT encoder]
        ViT_g["ViT Feature Extractor<br/>(人类视频分支)"]:::moduleStyle
        ViT_r["ViT Feature Extractor<br/>(机器人分支)"]:::moduleStyle
    end
    Vg --> ViT_g
    Ir --> ViT_r

    %% --- 3. 潜在时空特征处理 (模块2) ---
    subgraph M2 [2. transformer encoder / perceiver-resampler]
        TE_g["Transformer Encoder<br/>(含 Perceiver-Resampler 压缩)"]:::moduleStyle
        TE_r["Transformer Encoder<br/>(含 Perceiver-Resampler 压缩)"]:::moduleStyle
    end
    ViT_g -- "高维时空 Token" --> TE_g
    ViT_r -- "高维时空 Token" --> TE_r

    TE_g -- "人类视频潜在特征" --> X_Attn
    TE_r -- "机器人观测潜在特征" --> X_Attn

    %% --- 4. 真实轨迹提取 (模块3) ---
    subgraph M3 [3. point tracker / co-tracker module]
        CT_g["Co-Tracker<br/>(提取人类视频轨迹)"]:::moduleStyle
        CT_r["Co-Tracker<br/>(提取机器人观测轨迹)"]:::moduleStyle
    end
    Vg --> CT_g
    Ir --> CT_r

    %% 真实标签变量
    tau_g["真实人类点轨迹 τ⁹"]:::varStyle
    tau_r["真实机器人点轨迹 τ^r_{t-k:t}"]:::varStyle
    CT_g --> tau_g
    CT_r --> tau_r

    %% --- 5. 辅助任务：轨迹预测 (模块4) ---
    subgraph M4 [4. track-prediction transformer]
        psi_g["轨迹预测头 ψ_g<br/>(Track-Prediction)"]:::moduleStyle
        psi_r["轨迹预测头 ψ_r<br/>(Track-Prediction)"]:::moduleStyle
    end
    TE_g -.->|仅训练期| psi_g
    TE_r -.->|仅训练期| psi_r

    tau_g_hat["预测人类点轨迹 τ̂⁹"]:::varStyle
    tau_r_hat["预测机器人点轨迹 τ̂^r"]:::varStyle
    psi_g --> tau_r_hat
    psi_r --> tau_g_hat

    %% 轨迹损失计算
    Loss_track_g{"Track Pred. Loss, use MSE loss"}:::moduleStyle
    Loss_track_r{"Track Pred. Loss, use MSE loss"}:::moduleStyle
    tau_g --> Loss_track_g
    tau_g_hat --> Loss_track_g
    tau_r --> Loss_track_r
    tau_r_hat --> Loss_track_r

    %% --- 6. 核心动作交互与输出 (模块5) ---
    subgraph M5 [5. X-Attention block]
        X_Attn["X Attention Blocks<br/>(Cross-Attention 跨模态融合)"]:::moduleStyle
    end

    %% 最终动作输出
    a_pred["预测动作 â_{t:t+h}<br/>(离散化为 256 Bins, 输出为one-hot vector)"]:::outputStyle
    X_Attn --> a_pred

    Loss_BC{"BC Loss<br/>(行为克隆损失，use cross-entropy loss)"}:::moduleStyle
    a_pred --> Loss_BC

    class I0,G,Ir inputStyle;
    class VM,ViT_g,ViT_r,TE_g,TE_r,CT_g,CT_r,psi_g,psi_r,X_Attn,Loss_track_g,Loss_track_r,Loss_BC moduleStyle;
    class Vg,tau_g,tau_r,tau_g_hat,tau_r_hat varStyle;
    class a_pred outputStyle;
```