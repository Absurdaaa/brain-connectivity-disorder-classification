# 大创项目

# 配置环境
```
  conda env create -f unified_env.yml                                                                    
  conda activate unified 
```

# 训练模型
```
  bash scripts/train_vote_models.sh
```

# 评测模型
```
  cd models/unified/
  python vote_inference.py --dataset ABIDE
```

