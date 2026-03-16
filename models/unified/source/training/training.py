from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model

        # 自动检测设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.logger.info(f'#model params: {count_params(self.model)}')
        print(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.model.name / str(cfg.unique_id)
        self.save_learnable_graph = cfg.get('save_learnable_graph', False)
        self.save_attn_weights = cfg.get('save_attn_weights', False)

        # 模型行为 flags（通过 model config 控制）
        self.use_partial = cfg.model.get('use_partial', False)
        self.use_pool_loss = cfg.model.get('use_pool_loss', False)
        self.use_regularization = cfg.model.get('use_regularization', False)

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, \
            self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def _forward(self, time_series, node_feature, partial_node_feature):
        if self.use_partial:
            output = self.model(time_series, node_feature, partial_node_feature)
        else:
            output = self.model(time_series, node_feature)

        if self.use_pool_loss:
            predict, aux_loss = output
            if aux_loss is None:
                aux_loss = 0
        else:
            predict, aux_loss = output, 0

        return predict, aux_loss

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, partial_node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series = time_series.to(self.device)
            node_feature = node_feature.to(self.device)
            partial_node_feature = partial_node_feature.to(self.device)
            label = label.to(self.device)

            if self.config.preprocess.continus:
                time_series, node_feature, partial_node_feature, label = continus_mixup_data(
                    time_series, node_feature, partial_node_feature, y=label)

            predict, aux_loss = self._forward(time_series, node_feature, partial_node_feature)
            loss = self.loss_fn(predict, label) + aux_loss

            if self.use_regularization:
                gates = self.model.get_gates().squeeze(-1)
                eps = 1e-6
                importance = gates.sum(dim=0)
                cv_squared = (importance.std(unbiased=False) / (importance.mean() + eps)) ** 2
                loss = loss + self.config.model.regularization_co * cv_squared

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels, result = [], []
        self.model.eval()

        for time_series, node_feature, partial_node_feature, label in dataloader:
            time_series = time_series.to(self.device)
            node_feature = node_feature.to(self.device)
            partial_node_feature = partial_node_feature.to(self.device)
            label = label.to(self.device)

            predict, _ = self._forward(time_series, node_feature, partial_node_feature)
            label = label.float()

            loss = self.loss_fn(predict, label)
            loss_meter.update_with_weight(loss.item(), label.shape[0])
            top1 = accuracy(predict, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(predict, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(labels, result, average='micro')
        report = classification_report(labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def save_attention_weights(self):
        """根据模型类型自动选择保存方式"""
        self.model.eval()
        self.save_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, 'get_all_attn_softmax'):
            # ASDFormer 风格
            attns_list, gates_list, topk_list = [], [], []
            for time_series, node_feature, partial_node_feature, label in self.test_dataloader:
                time_series = time_series.cuda()
                node_feature = node_feature.cuda()
                partial_node_feature = partial_node_feature.cuda()
                self._forward(time_series, node_feature, partial_node_feature)
                attns = torch.mean(torch.stack(self.model.get_all_attn_softmax()), dim=0)
                attns_list.append(attns.detach().cpu())
                gates_list.append(self.model.get_gates().detach().cpu())
                topk_list.append(self.model.get_topk_info().detach().cpu())
            if attns_list:
                np.save(self.save_path / "attns.npy", torch.cat(attns_list, dim=0).numpy())
                np.save(self.save_path / "gates.npy", torch.cat(gates_list, dim=0).numpy())
                np.save(self.save_path / "topk_info.npy", torch.cat(topk_list, dim=0).numpy())

        elif hasattr(self.model, 'get_assign_mat'):
            # Com-BrainTF 风格
            attn_weights, assign_matrices, labels_list = [], [], []
            for dl in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
                for time_series, node_feature, partial_node_feature, label in dl:
                    time_series = time_series.cuda()
                    node_feature = node_feature.cuda()
                    partial_node_feature = partial_node_feature.cuda()
                    self._forward(time_series, node_feature, partial_node_feature)
                    assign_matrices.append(self.model.get_assign_mat().detach().cpu().numpy())
                    attn_weights.append(self.model.get_attention_weights()[0].detach().cpu().numpy())
                    labels_list.append(label.numpy())
            np.save(self.save_path / "attnWeights.npy", np.array(attn_weights, dtype=object), allow_pickle=True)
            np.save(self.save_path / "assign_matrices.npy", np.array(assign_matrices, dtype=object), allow_pickle=True)
            np.save(self.save_path / "labels.npy", np.array(labels_list, dtype=object), allow_pickle=True)

    def save_result(self, results):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy", results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_path / "model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        best_val_AUC = 0
        best_test_acc = best_test_AUC = best_test_sen = best_test_spec = 0

        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)
            test_result = self.test_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.5f}',
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                "Val Loss": self.val_loss.avg,
                "Val Accuracy": self.val_accuracy.avg,
                "Val AUC": val_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
            })

            if val_result[0] > best_val_AUC:
                best_val_AUC = val_result[0]
                best_test_acc = self.test_accuracy.avg
                best_test_AUC = test_result[0]
                best_test_sen = test_result[-1]
                best_test_spec = test_result[-2]
                wandb.run.summary["Best Test Accuracy"] = best_test_acc
                wandb.run.summary["Best Test AUC"] = best_test_AUC
                wandb.run.summary["Best Val AUC"] = best_val_AUC
                wandb.run.summary["Best Test Sensitivity"] = best_test_sen
                wandb.run.summary["Best Test Specificity"] = best_test_spec
                if self.save_attn_weights:
                    self.save_attention_weights()

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

        self.save_result(training_process)
        return [best_test_acc, best_test_AUC, best_test_sen, best_test_spec]
