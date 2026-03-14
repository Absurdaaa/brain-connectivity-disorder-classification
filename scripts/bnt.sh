cd ./models/BrainNetworkTransformer
export WANDB_MODE=offline
python -m source --multirun datasz=100p model=bnt,fbnetgen,brainnetcnn,transformer dataset=ABIDE,ABCD repeat_time=5 preprocess=mixup
