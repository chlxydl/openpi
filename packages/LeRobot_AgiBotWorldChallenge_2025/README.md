# Install
## Virtual env
```
# Option 1:
conda create -y -n lerobot_env python=3.10
conda activate lerobot_env
```

```
# Option 2:
python3.10 -m venv lerobot_env
source lerobot_env/bin/activate
```

## LeRobot
官方指南：https://github.com/huggingface/lerobot?tab=readme-ov-file#installation
```bash
# ffmpeg for miniconda
sudo apt update
sudo apt install ffmpeg

# Additional dependencies
sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

cd lerobot
pip install -e .
```
```bash
#For test
pip install -e ".[aloha, pusht]"

python -m lerobot.scripts.eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```
这个测试用直接测试了一个简单的push T任务。如果一切正常，可以看到文件夹中有实验结果，命名类似于outputs/eval/2025-07-03/23-08-06_pusht_diffusion/videos/eval_episode_9.mp4。

正常输出是一个球把T形字母和灰色位置对齐，可以下载到本地看是否正常

# Train
可能存在问题，如果只用单卡训练，建议把
python -m lerobot.scripts.train 替代python -m lerobot.scripts.train_distributed_gyb


Single GPU：
```bash
#公开数据集
python -m lerobot.scripts.train    --policy.type=diffusion     --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human  --output_dir=outputs/train/diffusion__all_task  --policy.repo_id=dummy 

python -m lerobot.scripts.train     --policy.type=diffusion     --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human  --output_dir=outputs/train/diffusion_aloha_transfer  --policy.repo_id=dummy

#自制数据集 更改repo_id为实际存储路径

python -m lerobot.scripts.train     --policy.type=diffusion     --dataset.repo_id=/data/v-yuboguo/Manipulation-SimData_LeRobot_all  --output_dir=outputs/train/diffusion_all_task  --policy.repo_id=dummy --batch_size=64

```

Multiple GPUs on local machine：
- Set the ```num_process``` in ```ds_config.yaml``` as the number of used GPUs. 
- Adjust ```--save_freq```, ```--policy.optimizer_lr```, ```--policy.scheduler_decay_steps```,  ```--policy.scheduler_decay_lr``` when using different numbers of GPUs.
```bash
CUDA_VISIBLE_DEVICES=<GPU_ids> accelerate launch --config_file ds_config.yaml -m lerobot.scripts.train_distribute --policy.path=lerobot/pi0 --dataset.repo_id=<path_of_data> --output_dir=<path_of_output> --policy.repo_id=dummy --policy.train_expert_only=true --policy.chunk_size=30 --policy.n_action_steps=30 --steps=2000000 --wandb.enable=true --wandb.project=pi0_lerobot --wandb.disable_artifact=True --policy.optimizer_lr=5e-5 --policy.scheduler_decay_steps=80000 --policy.scheduler_decay_lr=5e-6  
```


