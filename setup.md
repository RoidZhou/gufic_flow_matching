# train

# infer
cd ~/autolab/GUFIC_mujoco-main/gufic_env/flow_matching
export PYTHONPATH=/home/zhou/autolab/GUFIC_mujoco-main:$PYTHONPATH
python -m infer_fm

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