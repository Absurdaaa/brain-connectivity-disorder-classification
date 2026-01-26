cd ./models/ASDFormer
export WANDB_MODE=offline
python -m source --multirun datasz=100p model=ASDFormer dataset=ABIDE repeat_time=5 preprocess=non_mixup

# * `datasz=100p` → use 100% of the dataset
# * `model=ASDFormer` → specify the model
# * `dataset=ABIDE` → select dataset
# * `repeat_time=5` → number of experiment repetitions
# * `preprocess=non_mixup` → preprocessing strategy

# test acc mean:69.39999893188477  std: 3.4409293529528546
# test auc mean:0.7798568149889402  std: 0.03587693268218858
# test sensitivity mean:0.7978881420963788  std: 0.052839090895082486
# test specficity mean:0.600939469828171  std: 0.0718545413218672