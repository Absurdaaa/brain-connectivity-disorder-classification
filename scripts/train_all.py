#!/usr/bin/env python3
"""
批量训练脚本 - Python 版本
支持串行/并行训练，自动保存权重和日志

用法:
    python scripts/train_all.py --repeat 5 --parallel
    python scripts/train_all.py --models ASDFormer ComBrainTF --datasets ABIDE
"""

import argparse
import subprocess
import os
from datetime import datetime
from pathlib import Path
import json


def run_training(model, dataset, repeat_time, base_dir):
    """运行单个训练任务"""
    print(f"\n{'='*60}")
    print(f"开始训练: {model} on {dataset}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    os.chdir(base_dir / "models/unified")
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    cmd = [
        "python", "-m", "source",
        f"datasz=100p",
        f"model={model}",
        f"dataset={dataset}",
        f"repeat_time={repeat_time}",
        f"preprocess=non_mixup",
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✓ {model} on {dataset} 训练完成")
        return {
            "model": model,
            "dataset": dataset,
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }
    except subprocess.CalledProcessError as e:
        print(f"✗ {model} on {dataset} 训练失败")
        print(f"错误信息: {e.stderr[-500:]}")
        return {
            "model": model,
            "dataset": dataset,
            "status": "failed",
            "error": e.stderr[-500:] if len(e.stderr) > 500 else e.stderr,
        }


def main():
    parser = argparse.ArgumentParser(description="批量训练脚本")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ASDFormer", "ComBrainTF", "BrainNetworkTransformer"],
        help="要训练的模型列表",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ABIDE", "MDD"],
        help="要使用的数据集列表",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="每个任务的重复次数",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="是否并行训练（需要多GPU）",
    )
    args = parser.parse_args()

    base_dir = Path("/Users/linshangjin/Desktop/dachuang")
    results = []

    print("="*60)
    print("批量训练任务")
    print(f"重复次数: {args.repeat}")
    print(f"模型: {args.models}")
    print(f"数据集: {args.datasets}")
    print(f"并行模式: {'是' if args.parallel else '否'}")
    print("="*60)

    if args.parallel:
        # 并行训练（需要多GPU或使用 multiprocessing）
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for model in args.models:
                for dataset in args.datasets:
                    future = executor.submit(
                        run_training, model, dataset, args.repeat, base_dir
                    )
                    futures.append(future)

            for future in futures:
                results.append(future.result())
    else:
        # 串行训练
        for model in args.models:
            for dataset in args.datasets:
                result = run_training(model, dataset, args.repeat, base_dir)
                results.append(result)

    # 保存训练日志
    log_file = base_dir / "models/unified/training_log.json"
    with open(log_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "results": results,
        }, f, indent=2)

    print("\n" + "="*60)
    print("所有训练任务完成")
    print(f"权重保存位置: models/unified/result/")
    print(f"训练日志: {log_file}")
    print("="*60)

    # 打印汇总
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n成功: {success}/{len(results)}")
    print(f"失败: {failed}/{len(results)}")


if __name__ == "__main__":
    main()
