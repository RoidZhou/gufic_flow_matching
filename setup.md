# train

# infer
cd ~/autolab/GUFIC_mujoco-main/gufic_env/flow_matching
export PYTHONPATH=/home/zhou/autolab/GUFIC_mujoco-main:$PYTHONPATH
python -m infer_fm

# Sphere task
## train log
### time = 2026.4.24
```python
train No.1
1. train cfm with p,R,Fe condition for hist_len = 16
2. [Epoch 016] train_loss=0.001778 val_loss=0.001111

train No.2
1. train cfm with p,R,Fe condition for fe_hist_len = 16, x_hist_len = 1

```
### time = 2026.4.24
```python
train No.1
1. train cfm with p,R,Fe condition for fe_hist_len = 16, x_hist_len = 1
2. noemalize condition

iner No.1
checkpoint: [Epoch 003] train_loss=0.003499 val_loss=0.002204

velocity MSE   : 0.000003
velocity MAE   : 0.000480
mean ||error|| : 0.001550
max  ||error|| : 0.134573

generated traj len : 9999
velocity MSE   : 0.000001
velocity MAE   : 0.000238
mean ||error|| : 0.000759
max  ||error|| : 0.096872
```

## time = 2026.4.28
```python

infer No.1
checkpoint: 37, 0.0024359383802604628, 0.0018670117064048723, 9.966596702259576e-05, 0.001627044685229758
cfm_transformer_random_start_best2

velocity MSE   : 0.000657
velocity MAE   : 0.002479
mean ||error|| : 0.009845
max  ||error|| : 1.063980
```

## time = 2026.5.4
```python
    """
    世界坐标系下点云：xw​=Rxe​+p
    转到机器人末端坐标系下点云：xe​=RT(xw​−p)
    不进行缩放
    """
    R_ec, t_ec, _ = get_hand_eye_from_xml(robot_model, robot_task)
    pc_world = pointcloud_cam_to_world_batch(cond_pc, p_raw, R_raw, R_ec, t_ec)
    pc_ee = np.einsum("tji,tpj->tpi", R_raw, pc_world[..., :3] - p_raw[:, None, :])  # R^T (x_w - p)
    pc_ee = pc_ee / 0.1
    cond_pc = np.stack(
        [uniform_sample_one_frame(pc_t, 2048, use_xyz_only=True) for pc_t in pc_ee],
        axis=0
    ).astype(np.float32)
    cond_pc = (cond_pc / 0.1).astype(np.float32)
```

```bash
infer_fm.py cfm_transformer_vis2pose_{type}_best15.pt

generated traj len : 9999
velocity MSE   : 0.000202
velocity MAE   : 0.001435
mean ||error|| : 0.005195
max  ||error|| : 0.623699
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_vis_pRFe_random_start
```

## time = 2026.5.5 v0.1
```python
    """
    世界坐标系下点云：xw​=Rxe​+p
    转到机器人末端坐标系下点云：xe​=RT(xw​−p)
    进行两次缩放 pc/0.1/0.1
    """
    R_ec, t_ec, _ = get_hand_eye_from_xml(robot_model, robot_task)
    pc_world = pointcloud_cam_to_world_batch(cond_pc, p_raw, R_raw, R_ec, t_ec)
    pc_ee = np.einsum("tji,tpj->tpi", R_raw, pc_world[..., :3] - p_raw[:, None, :])  # R^T (x_w - p)
    cond_pc = np.stack(
        [uniform_sample_one_frame(pc_t, 2048, use_xyz_only=True) for pc_t in pc_ee],
        axis=0
    ).astype(np.float32)
```
### 结果
```bash
infer_fm.py cfm_transformer_vis2pose_{type}_best14.pt

generated traj len : 9999
velocity MSE   : 0.000023
velocity MAE   : 0.000991
mean ||error|| : 0.003038
max  ||error|| : 0.169115
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_vis_pRFe_random_star
```


## time = 2026.5.5 v0.2
### 上一版
```python
    """
    世界坐标系下点云：xw​=Rxe​+p
    转到机器人末端坐标系下点云：xe​=RT(xw​−p)
    进行两次缩放 pc/0.1/0.1
    """
    R_ec, t_ec, _ = get_hand_eye_from_xml(robot_model, robot_task)
    pc_world = pointcloud_cam_to_world_batch(cond_pc, p_raw, R_raw, R_ec, t_ec)
    pc_ee = np.einsum("tji,tpj->tpi", R_raw, pc_world[..., :3] - p_raw[:, None, :])  # R^T (x_w - p)
    cond_pc = np.stack(
        [uniform_sample_one_frame(pc_t, 2048, use_xyz_only=True) for pc_t in pc_ee],
        axis=0
    ).astype(np.float32)
```
### 新增
```python
    embed_dim: int = 128
    cond_dim: int = 9 * x_hist_len + 6 * force_hist_len # K=16 步 6 维 力历史，1 步 9 维状态
# guide_feat FiLM形式注入
class VelocityFMTransformer(nn.Module):
    def __init__(
        self,
        x_dim=6,
        cond_dim=6,          # 这里只放 cond_main 维度，不再包含 guide_dim
        guide_dim=16,        # 新增
        time_dim=64,
        hidden_dim=256,
        num_layers=4,
        use_cond=True,
        nhead=8,
        dropout=0.1,
        max_seq_len=12000,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.use_cond = use_cond
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # x_t + t_emb 先单独投影
        self.input_proj = nn.Linear(x_dim + time_dim, hidden_dim)

        # 主条件分支：p/R/Fe 历史
        if self.use_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.cond_proj = None

        # guide_feat 单独控制 hidden
        self.guide_scale = nn.Sequential(
            nn.Linear(guide_dim, hidden_dim),
            nn.Tanh(),   # 控制 scale 幅度
        )
        self.guide_shift = nn.Linear(guide_dim, hidden_dim)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, t, cond_main=None, guide=None):
        """
        x_t: [B,T,6] or [B,6]
        t:   [B,1] or [B,T,1]
        cond_main: [B,cond_dim] or [B,T,cond_dim]
        guide:     [B,guide_dim] or [B,T,guide_dim]
        """
        squeeze_back = False

        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
            squeeze_back = True

        B, T, _ = x_t.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}")

        # t -> [B,T,1]
        if t.dim() == 2:
            t = t[:, None, :]
            t = t.expand(B, T, 1)
        elif t.dim() == 3 and t.shape[1] == 1:
            t = t.expand(B, T, 1)

        t_emb = self.time_emb(t)                       # [B,T,time_dim]
        h = self.input_proj(torch.cat([x_t, t_emb], dim=-1))   # [B,T,H]

        # 主条件注入
        if self.use_cond:
            if cond_main is None:
                raise ValueError("use_cond=True 时，cond_main 不能为 None")
            if cond_main.dim() == 2:
                cond_main = cond_main.unsqueeze(1).expand(B, T, cond_main.shape[-1])
            elif cond_main.dim() == 3 and cond_main.shape[1] == 1:
                cond_main = cond_main.expand(B, T, cond_main.shape[-1])

            h = h + self.cond_proj(cond_main)

        # guide_feat 直接做 FiLM
        if guide is not None:
            if guide.dim() == 2:
                guide = guide.unsqueeze(1).expand(B, T, guide.shape[-1])
            elif guide.dim() == 3 and guide.shape[1] == 1:
                guide = guide.expand(B, T, guide.shape[-1])

            scale = self.guide_scale(guide)   # [B,T,H]
            shift = self.guide_shift(guide)   # [B,T,H]
            h = h * (1.0 + scale) + shift

        h = h + self.pos_embed[:, :T, :]
        h = self.transformer(h)
        out = self.output_proj(h)

        if squeeze_back:
            out = out.squeeze(1)

        return out



```
### 结果
```bash
infer_fm.py cfm_transformer_vis2pose_{type}_best16.pt

generated traj len : 9999
velocity MSE   : 0.000106
velocity MAE   : 0.001303
mean ||error|| : 0.004913
max  ||error|| : 0.252811
```

## time = 2026.5.5
### 上一版 v0.2
```python
guide_feat FiLM形式注入
embed_dim: int = 128
```
### 新增
```python
    lambda_delta: int = 0.5
```
### 结果
```bash
infer_fm.py cfm_transformer_vis2pose_{type}_best17.pt

generated traj len : 9999
velocity MSE   : 0.000544
velocity MAE   : 0.002544
mean ||error|| : 0.010422
max  ||error|| : 0.722253
```

## time = 2026.5.6
### 上一版 v0.2
```python
guide_feat FiLM形式注入
embed_dim: int = 128
lambda_delta: int = 0.5

obs_encoder.eval()
# 固定 val 随机性
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
```
### 新增
```python
lambda_delta：int = 0.1
embed_dim: int = 64
```
### 结果
```bash
infer_fm.py cfm_transformer_vis2pose_{type}_best19.pt

generated traj len : 9999
velocity MSE   : 0.000095
velocity MAE   : 0.001162
mean ||error|| : 0.004320
max  ||error|| : 0.378553
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_vis_pRFe_random_start
```


# Insertion task
## train log
### time = 2026.5.7 v0.1
### 新增
```python
guide_feat FiLM形式注入
embed_dim: int = 64
lambda_delta: int = 0.1
x_hist_len: int = 1

```
### 结果
```bash
infer_fm.py cfm_transformer_{type}_best4.pt

generated traj len : 5999
velocity MSE   : 0.000832
velocity MAE   : 0.017865
mean ||error|| : 0.056392
max  ||error|| : 0.128510
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_peg_vis_pRFe_random_start
    insertion2026.5.7_v0.1.png

```

### time = 2026.5.8 v0.1
### 新增
```python
x_hist_len: int = 16

```
### 结果
```bash
infer_fm.py cfm_transformer_{type}_best7.pt

generated traj len : 5999
velocity MSE   : 0.000349
velocity MAE   : 0.008611
mean ||error|| : 0.035357
max  ||error|| : 0.185639
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_peg_vis_pRFe_random_start
    insertion2026.5.8_v0.1.png
```

### time = 2026.5.8 v0.2
### 新增
```python
x_hist_len: int = 4

```
### 结果
```bash
infer_fm.py cfm_transformer_{type}_best8.pt

generated traj len : 5999
velocity MSE   : 0.000284
velocity MAE   : 0.008355
mean ||error|| : 0.035939
max  ||error|| : 0.169925
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_peg_vis_pRFe_random_start
    insertion2026.5.8_v0.2.png
```

# Boltnut task
## train log
### time = 2026.5.21 v0.1
### 新增
```python
misc_fun.py:
    elif task == 'bolt':
        # --- 1) 基准位姿（轴系） ---
        T1 = 1.0          # 从当前位置到孔口上方
        T2 = 1.0          # 从孔口上方下落到孔口
        T3 = max_time - T1 - T2   # 沿孔轴插入
        h = -0.005         # 孔口上方 4 cm
collect_dataset.py:
    if self.randomized_start:
        rand_xy = 2 * (np.random.rand(2,) - 0.5) * 0.01

    for episode in range(154, 200):
```

### time = 2026.5.23 v0.1
### 新增
```python
collect_dataset.py:
    elif self.task == 'bolt':
        # self.p_init = np.array([0.50, 0.0, 0.225])
        self.p_init = np.array([0.50, 0.0, 0.25])
        Rd_default = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])

Indy7_nutbolt.xml
<body name="eye_in_hand_body" pos="0.00 0.04 0.15" euler="3.1415 0 0">

misc_fun.py:
    def __post_init__(self):
        if self.task == "bolt":
            self.pred_horizon = 100
            self.stride = 2
            self.delta_p = 5.0
            self.delta_R = 1.0
```

### 结果
```bash
infer_fm.py cfm_transformer_{type}_best9.pt

generated traj len : 15999
velocity MSE   : 0.012578
velocity MAE   : 0.039516
mean ||error|| : 0.208123
max  ||error|| : 3.135781
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_boltnut_vis_pRFe_random_start
    insertion2026.5.23_v0.2.png
```

### time = 2026.5.23 v0.2
### 新增
```python

misc_fun.py:
    def __post_init__(self):
        if self.task == "bolt":
            self.pred_horizon = 100
            self.stride = 2
            self.delta_p = 2.0
            self.delta_R = 1.0
```
### 结果
```bash
infer_fm.py cfm_transformer_{type}_best10.pt

generated traj len : 15999
velocity MSE   : 0.022677
velocity MAE   : 0.073333
mean ||error|| : 0.286423
max  ||error|| : 2.343052
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_boltnut_vis_pRFe_random_start

```

### time = 2026.5.24 v0.1
### 新增
```python
Indy7_wiping_sphere.xml
相机位置改成与boltnut一致，解决在加载数据集时计算相机外参的bug.

misc_fun.py:
    def __post_init__(self):
        if self.task == "bolt":
            self.pred_horizon = 100
            self.stride = 1
            self.delta_p = 2.0
            self.delta_R = 1.0
```
### 结果
```bash
infer_fm.py cfm_transformer_{type}_best13.pt

generated traj len : 15999
velocity MSE   : 0.001498
velocity MAE   : 0.004930
mean ||error|| : 0.018226
max  ||error|| : 2.992507
Saved to: /home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_boltnut_vis_pRFe_random_start

```

