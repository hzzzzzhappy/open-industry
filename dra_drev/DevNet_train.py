import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import sys
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import random
from dataloaders.dataloader import initDataloader
from model.DevNet import DevNet
from model.loss.deviation_loss import DeviationLoss
from model.loss import build_criterion
import math
class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = initDataloader.build(args)

        self.model = DevNet(args)
        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-5
        )

        warmup_steps = 50
        cosine_steps = args.epochs
        min_lr = args.lr / 10
        base_lr = args.lr

        def lr_lambda(current_step: int):
            """Epoch index; call scheduler.step() once per epoch after training."""
            if current_step < warmup_steps:
                return 1.0

            if current_step < warmup_steps + cosine_steps:
                progress = (current_step - warmup_steps) / float(cosine_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return ((base_lr - min_lr) / base_lr) * cosine_decay + (min_lr / base_lr)

            return min_lr / base_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )

        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()

    def train(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            features = sample['pcd_features']
            target = sample['label']
            if self.args.cuda:
                features = features.cuda()
                target = target.cuda()

            output = self.model(features)
            
            if isinstance(output, list):
                output = output[0]

            loss = self.criterion(output, target.unsqueeze(1).float())
            
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            train_loss += loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
        self.scheduler.step()

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        total_pred = []
        total_target = []

        for i, sample in enumerate(tbar):
            features = sample['pcd_features']
            target = sample['label']
            mask_keep = target != 1
            features = features[mask_keep]
            target   = target[mask_keep]
            target[target == 2] = 1

            if len(target) == 0:
                continue
            if self.args.cuda:
                features = features.cuda()
                target = target.cuda()
            

            with torch.no_grad():
                output = self.model(features)
            
            if isinstance(output, list):
                output = output[0]
            
            loss = self.criterion(output, target.unsqueeze(1).float())
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            total_pred.append(output.data.cpu().numpy())
            total_target.append(target.cpu().numpy())
            
        total_pred = np.concatenate(total_pred)
        total_target = np.concatenate(total_target)
        roc = roc_auc_score(total_target, total_pred)
        pr = average_precision_score(total_target, total_pred)
        
        print(f'AUC-ROC: {roc:.4f}, AUC-PR: {pr:.4f}')
        return roc, pr

    def save_weights(self, filename):
        if not os.path.exists(self.args.experiment_dir):
            os.makedirs(self.args.experiment_dir)
        torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training args
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='devnet_model.pkl', help="the name of model weight")
    parser.add_argument('--experiment_dir', type=str, default='./experiment_devnet', help="experiment dir root")
    parser.add_argument('--xyz_backbone', type=str, default='Point_MAE', help="Backbone: Point_MAE or Point_BERT")
    # Dataset args
    parser.add_argument('--dataset', type=str, default='open_industry', help="dataset name: open_industry, anomaly_shapenet")
    parser.add_argument('--dataset_root', type=str, required=True, help="dataset root path")
    parser.add_argument('--classname', type=str, default='fangxiedianpian', help="the subclass of the datasets")
    parser.add_argument('--know_class', nargs='+', default=['Bump', 'Deformation'], help="known anomaly classes")
    parser.add_argument('--cont_rate', type=float, default=0.1, help="contamination rate")
    parser.add_argument('--test_threshold', type=int, default=0, help="test threshold")
    parser.add_argument('--test_rate', type=float, default=0.0, help="test rate")
    parser.add_argument('--nAnomaly', type=int, default=5, help="number of anomaly data in training set")
    parser.add_argument('--outlier_root', type=str, default=None, help="outlier root")
    
    # Model args
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    
    parser.add_argument('--criterion', type=str, default='deviation', help='loss type')

    # Device
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--use_pseudo_anomaly', type=int, default=0, help="whether to use pseudo anomaly data (1=True, 0=False)")
    args = parser.parse_args()

    SEED = int(args.ramdn_seed)
    print(f"[Seed] Using seed: {SEED}")
    # Make hash-based operations deterministic
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print("batch_size:", args.batch_size)
    print("init_loss",args.lr)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        
    trainer = Trainer(args)
    torch.manual_seed(args.ramdn_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    argsDict = args.__dict__
    with open(os.path.join(args.experiment_dir, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print('Total Epoches:', trainer.args.epochs)
    
    # best_roc = 0
    result_file = os.path.join(args.experiment_dir, "result2.txt")
    with open(result_file, "w") as f:
        f.write("==== Training Log Started ====\n")

    TOP_K = 5
    best_checkpoints = []
    for epoch in range(0, trainer.args.epochs):
        if epoch % 5 == 0 and epoch > 0:
            
            unseen_roc, unseen_pr = trainer.eval()

            log_line = (
                f"Epoch {epoch:03d} | "
                f"Unseen ROC={unseen_roc:.4f}, PR={unseen_pr:.4f}\n"
            )

            with open(result_file, "a") as f:
                f.write(log_line)

            print(f"[Eval] {log_line.strip()}  --> wrote {result_file}")

            score = unseen_roc

            ckpt_name = f"epoch_{epoch:03d}_roc_{score:.4f}.pkl"
            ckpt_path = os.path.join(args.experiment_dir, ckpt_name)
            trainer.save_weights(ckpt_name)
            print(f"[Save] checkpoint saved: {ckpt_path}")

            best_checkpoints.append((score, ckpt_path, epoch))
            best_checkpoints.sort(key=lambda x: x[0], reverse=True)

            if len(best_checkpoints) > TOP_K:
                worst_score, worst_path, worst_epoch = best_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    print(f"[Clean] removed checkpoint: epoch={worst_epoch}, score={worst_score:.4f}, path={worst_path}")

            print("[Best Models]")
            for rank, (s, p, e) in enumerate(best_checkpoints, start=1):
                print(f"  Top{rank}: epoch={e}, score={s:.4f}, path={p}")

        trainer.train(epoch)

    # trainer.eval()
    # trainer.save_weights(args.savename)


