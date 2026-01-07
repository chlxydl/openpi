
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
/opt/conda/envs/ptca/bin/python scripts/train.py \
pi0_xxl_vla \
--exp-name=pi0_xxl_1223_delta_joint \
--data.repo-ids \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_3/2025.12.8 \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_3/2025.12.14 \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_3/2025.12.15 \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_2/2025.12.08 \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_2/2025.12.09 \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_2/2025.12.11 \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_2/2025.12.12 \
                /work/outputs/xxl_ds_1203/xxl_ds_1216/scenario_2/2025.12.14 \
                /work/outputs/xxl_ds_1203/xxl_ds_1223/scenario_1 \
                /work/outputs/xxl_ds_1203/xxl_ds_1223/2025.12.22/scenario_4 \
                /work/outputs/xxl_ds_1203/xxl_ds_1223/2025.12.20/scenario_4 \
                /work/outputs/xxl_ds_1203/xxl_ds_1223/2025.12.16/scenario_3 \
--data.assets.asset-id=/work/outputs/xxl_ds_1203/norm_delta_joints_1223 \
--checkpoint_base_dir=/work/outputs/debug \
--batch_size=32 \
--num_workers=16 \
--num_train_steps=60_000 \
--fsdp_devices=1 \
--wandb_enabled \
--overwrite \
--lr-schedule.warmup-steps=1_000 \
--lr-schedule.peak-lr=2.5e-5 \
--lr-schedule.decay-steps=3_000 \
--lr-schedule.decay-lr=2.5e-6
