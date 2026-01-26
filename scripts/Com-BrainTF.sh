cd ./models/Com-BrainTF
export WANDB_MODE=offline

python -m source --multirun datasz=100p model=comtf dataset=ABIDE repeat_time=5 preprocess=non_mixup