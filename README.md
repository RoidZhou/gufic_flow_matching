


## 云服务器训练修改

### 1. dataset.py
```python
def load_xml(robot_name, task):
    dir = os.getcwd() + '/'
    if robot_name == 'ur5e':
        raise NotImplementedError
    elif robot_name == 'indy7':
        if task == "sphere":
            model_path = dir + "../mujoco_models/Indy7_wiping_sphere.xml"
        elif task == "insertion":
            model_path = dir + "../mujoco_models/Indy7_insertion.xml"
        elif task == "bolt":
                model_path = dir + "../mujoco_models/Indy7_nutbolt.xml"
        else:
            model_path = dir + "../mujoco_models/Indy7_wiping.xml"
    elif robot_name == 'panda':
        raise NotImplementedError
    else:
        raise NotImplementedError

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data
```

### 2. model.py
```python
import sys
sys.path.append(r"/root/vla/gufic_flow_matching")
```

### 3. train_fm.py
```python
    cfg = TrainConfig(
        train_demo_dir="/root/autodl-tmp/boltnut3_demos_vis_random_start/boltnut3_demos_vis_random_start_train",
        val_demo_dir="/root/autodl-tmp/boltnut3_demos_vis_random_start/boltnut3_demos_vis_random_start_val",
        type=type,
        epochs=1000,
        batch_size=8,
        save_dir=f"/root/vla/gufic_flow_matching/gufic_env/flow_matching/checkpoints_cfm_transformer_boltnut3_vis_pRFe_{type}"
    )
```

python merge_lerobot_datasets.py   --filter-root /root/autodl-tmp/ur5_rg2_real_smolvla_dataset_force_boltnut_merged   --drop-existing-episodes 6,7,18,5
2,66   --in-place

```bash
export SMOLVLA_PRETRAINED_PATH=/root/autodl-tmp/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205
export SMOLVLA_VLM_MODEL_NAME=/root/autodl-tmp/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
```

 python  train_vla.py --config_path /root/autodl-tmp/checkpoints_smolvla_wo_force_vqvae_boltnut_speedup_total/checkpoints/020000//pretrained_model/train_config.json