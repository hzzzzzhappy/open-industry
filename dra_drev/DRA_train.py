import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import random
import math
import datetime
import subprocess
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from dataloaders.dataloader import initDataloader
from model.DRA import DRA
from model.loss import build_criterion

WEIGHT_DIR = './weights'


def _torch_load_checkpoint(path, map_location=None):
    try:
        if map_location is not None:
            return torch.load(path, map_location=map_location, weights_only=False)
        return torch.load(path, weights_only=False)
    except TypeError:
        if map_location is not None:
            return torch.load(path, map_location=map_location)
        return torch.load(path)


class Trainer(object):

    def __init__(self, args):
        self.args = args
        kwargs = {'num_workers': args.workers}
        print("Preparing dataloader...")
        self.train_loader, self.test_loader= initDataloader.build(args, **kwargs)
        if self.args.total_heads == 4:
            print('')
            print("Preparing reference dataloader...")
            temp_args = args
            #     temp_args.batch_size = 1
            temp_args.batch_size = self.args.nRef
            temp_args.nAnomaly = 0
            self.ref_loader, _ = initDataloader.build(temp_args, **kwargs)
            assert len(self.ref_loader) % args.nRef == 0, \
        f"[ERROR] The size of ref_loader ({len(self.ref_loader)}) is not divisible by args.nRef ({args.nRef})."
            self.ref = iter(self.ref_loader)
        # args.device should be set in main before Trainer created
        assert args.device is not None, "args.device must be set (e.g. cuda:0)"
        if hasattr(args, 'device'):
            self.device = args.device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DRA(args)

        if self.args.pretrain_dir != None:
            # load pretrained weights and map them to the current device to avoid
            # tensors ending up on a different GPU (which causes cross-device errors)
            state = _torch_load_checkpoint(self.args.pretrain_dir, map_location=self.device)
            self.model.load_state_dict(state)
            print('Load pretrain weight from: ' + self.args.pretrain_dir)

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
            """Epoch index as current_step; call scheduler.step() once per epoch."""
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


        logdir = os.path.join(self.args.experiment_dir, "tensorboard")
        os.makedirs(logdir, exist_ok=True)
        run_name = datetime.datetime.now().strftime(f"{args.classname}_%Y%m%d_%H-%M")
        self.tb_writer = SummaryWriter(log_dir=os.path.join(logdir, run_name), flush_secs=60)
    def generate_target(self, target, eval=False):
        targets = list()
        if eval:
            targets.append(target==0)
            targets.append(target)
            targets.append(target)
            targets.append(target)
            return targets
        else:
            temp_t = target != 0
            targets.append(target == 0)
            targets.append(temp_t[target != 2])
            targets.append(temp_t[target != 1])
            targets.append(target != 0)
        return targets

    def training(self, epoch):
        
        train_loss = 0.0
        class_loss = list()
        for j in range(self.args.total_heads):
            class_loss.append(0.0)
        self.model.train()
        self.scheduler.step()
        tbar = tqdm(self.train_loader)
        max_tries = 10
        tries = 0
        self.train_iter = iter(self.train_loader)
            
        for idx, sample in enumerate(tbar):
            
            pcd_feature, target = sample['pcd_features'], sample['label']

            target_np = target.cpu().numpy()
            while 0 not in target_np and tries < max_tries:
                print(f"[Warn] batch has no normal (label 0); resampling ({target_np})")
                tries += 1
                tbar.set_description(f"Retry {tries}/{max_tries} | No label=0")
                try:
                    sample = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    sample = next(self.train_iter)

                pcd_feature, target = sample['pcd_features'], sample['label']

            if 0 not in target:
                print(f"[Warn] Skip batch idx={idx}: still no label=0 after {tries} retries.")
                continue

            tries = 0

            pcd_feature = pcd_feature.cpu().numpy()
            target = target.cpu().numpy()

            if self.args.cuda:
                pcd_feature = torch.from_numpy(pcd_feature).float().to(self.device)
                target = torch.from_numpy(target).long().to(self.device)

            if self.args.total_heads == 4:
                pcd_features = pcd_feature
                try:
                    ref_pcd_feature = next(self.ref)['pcd_features']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_pcd_feature = next(self.ref)['pcd_features']

                # move ref and batch to correct device
                if isinstance(ref_pcd_feature, np.ndarray):
                    ref_pcd_feature = torch.from_numpy(ref_pcd_feature).float().to(self.device)
                    ref_pcd_feature = ref_pcd_feature.unsqueeze(0)
                else:
                    ref_pcd_feature = ref_pcd_feature.to(self.device)

                if isinstance(pcd_features, np.ndarray):
                    pcd_features = torch.from_numpy(pcd_features).float().to(self.device)
                    pcd_features = pcd_features.unsqueeze(0)
                    
                assert ref_pcd_feature.size(0) == args.nRef, \
                f"[ERROR] ref_pcd_feature batch mismatch: got {ref_pcd_feature.size(0)}, expected {args.nRef}"
                pcd_features = torch.cat([ref_pcd_feature, pcd_features], dim=0)

            
            outputs = self.model(pcd_features, target)

            targets = self.generate_target(target)

            losses = []
            for i in range(self.args.total_heads):
                if self.args.criterion == 'CE':
                    prob = F.softmax(outputs[i], dim=1)
                    head_loss = self.criterion(prob, targets[i].long())
                else:
                    head_loss = self.criterion(outputs[i], targets[i].float())

                head_loss = head_loss.mean()
                losses.append(head_loss)

            loss = torch.stack(losses).sum()

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            batch_total_loss = loss.item()
            train_loss += batch_total_loss

            for i in range(self.args.total_heads):
                class_loss[i] += losses[i].item()

            avg_total_loss = train_loss / (idx + 1)
            avg_head_losses = [class_loss[i] / (idx + 1) for i in range(self.args.total_heads)]

            tbar.set_description(f'Epoch:{epoch}, Total: {avg_total_loss:.3f}')
            tbar.set_postfix({f'h{i}': f'{avg_head_losses[i]:.3f}' for i in range(self.args.total_heads)})

            global_step = epoch * len(self.train_loader) + idx

            self.tb_writer.add_scalar("train/loss_total_batch", batch_total_loss, global_step)
            for i in range(self.args.total_heads):
                self.tb_writer.add_scalar(f"train/loss_head_{i}_batch",losses[i].item(),global_step)
            self.tb_writer.add_scalar("train/loss_total_avg_running", avg_total_loss, global_step)
            for i in range(self.args.total_heads):
                self.tb_writer.add_scalar(f"train/loss_head_{i}_avg_running",avg_head_losses[i],global_step)

    def normalization(self, data):
        data = np.asarray(data, dtype=np.float32)
        dmin = data.min()
        dmax = data.max()

        if dmax - dmin < 1e-12:
            return np.zeros_like(data, dtype=np.float32)

        return (data - dmin) / (dmax - dmin)

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        class_pred = list()
        for i in range(self.args.total_heads):
            class_pred.append(np.array([]))
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            pcd_features, target = sample['pcd_features'], sample['label']
            
            if self.args.cuda:
                pcd_features, target = pcd_features.to(self.device), target.to(self.device)

            if self.args.total_heads == 4:
                try:
                    ref_pcd_feature = next(self.ref)['pcd_features'].to(self.device)
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_pcd_feature = next(self.ref)['pcd_features'].to(self.device)
                pcd_features = torch.cat([ref_pcd_feature, pcd_features], dim=0)
            with torch.no_grad():
                outputs = self.model(pcd_features, target)
                targets = self.generate_target(target, eval=True) 

                losses = list()
                for i in range(self.args.total_heads):
                    if self.args.criterion == 'CE':
                        prob = F.softmax(outputs[i], dim=1)
                        losses.append(self.criterion(prob, targets[i].long()))
                    else:
                        losses.append(self.criterion(outputs[i], targets[i].float()))

                loss = losses[0]
                for i in range(1, self.args.total_heads):
                    loss += losses[i]

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            for i in range(self.args.total_heads):
                if i == 0:
                    data = -1 * outputs[i].data.cpu().numpy()
                else:
                    data = outputs[i].data.cpu().numpy()
                class_pred[i] = np.append(class_pred[i], data)
            total_target = np.append(total_target, target.cpu().numpy())

        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args.total_heads):
            total_pred = total_pred + self.normalization(class_pred[i])
        total_target = total_target.astype(int)


        overall_binary = (total_target != 0).astype(int)
        total_roc, total_pr = aucPerformance(total_pred, overall_binary)

        seen_mask = (total_target == 0) | (total_target == 1)
        seen_targets = total_target[seen_mask]
        seen_pred = total_pred[seen_mask]
        seen_binary = (seen_targets == 1).astype(int)

        if np.any(seen_binary == 1) and np.any(seen_binary == 0):
            seen_roc, seen_pr = aucPerformance(seen_pred, seen_binary)
        else:
            seen_roc, seen_pr = float('nan'), float('nan')

        unseen_mask = (total_target == 0) | (total_target == 2)
        unseen_targets = total_target[unseen_mask]
        unseen_pred = total_pred[unseen_mask]
        unseen_binary = (unseen_targets == 2).astype(int)

        if np.any(unseen_binary == 1) and np.any(unseen_binary == 0):
            unseen_roc, unseen_pr = aucPerformance(unseen_pred, unseen_binary)
        else:
            unseen_roc, unseen_pr = float('nan'), float('nan')

        with open(self.args.experiment_dir + '/result.txt', mode='a+', encoding="utf-8") as w:
            for label, score in zip(total_target, total_pred):
                w.write(str(label) + '   ' + str(score) + "\n")

        normal_mask = (total_target == 0)
        outlier_mask = (total_target != 0)
        plt.clf()
        plt.bar(np.arange(total_pred.size)[normal_mask], total_pred[normal_mask], color='green')
        plt.bar(np.arange(total_pred.size)[outlier_mask], total_pred[outlier_mask], color='red')
        plt.ylabel("Anomaly score")
        plt.savefig(self.args.experiment_dir + "/vis.png")

        print(f"[Eval] Overall   ROC: {total_roc:.4f}, PR: {total_pr:.4f}")
        print(f"[Eval] Seen-only ROC: {seen_roc:.4f}, PR: {seen_pr:.4f}")
        print(f"[Eval] Unseen-only ROC: {unseen_roc:.4f}, PR: {unseen_pr:.4f}")

        return total_roc, total_pr, seen_roc,seen_pr,unseen_roc,unseen_pr

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

    def load_weights(self, filename):
        path = os.path.join(WEIGHT_DIR, filename)
        # load checkpoint onto the model's device to avoid cross-device tensors
        state = _torch_load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(state)

    def init_network_weights_from_pretraining(self):

        net_dict = self.model.state_dict()
        ae_net_dict = self.ae_model.state_dict()

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.model.load_state_dict(net_dict)

    def start_tensorboard(self, logdir="/root/tf-logs", port=6006):
        """Start TensorBoard in the background (non-blocking)."""
        os.makedirs(logdir, exist_ok=True)

        subprocess.call(f"fuser -k {port}/tcp", shell=True)

        print(f"[TensorBoard] Starting TensorBoard at http://localhost:{port}")
        print(f"[TensorBoard] Logdir = {logdir}")

        cmd = f"tensorboard --logdir {logdir} --port {port}"
        subprocess.Popen(cmd, shell=True)

        time.sleep(2)
def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")


    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='open_industry', help="dataset name: open_industry, anomaly_shapenet")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, required=True, help="dataset root path")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_14', help="dataset root")
    parser.add_argument('--classname', type=str, default='fangxiedianpian', help="dataset class")
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument(
        '--know_class',   
        nargs='+',
        type=str,
        default=None
    )
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=4, help="number of head in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--xyz_backbone', type=str, default='Point_MAE', help="Backbone: Point_MAE or Point_BERT")

    parser.add_argument('--use_pseudo_anomaly', type=int, default=1, help="whether to use pseudo anomaly data (1=True, 0=False)")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    args = parser.parse_args()

    # ----------------- Reproducibility (controlled by --ramdn_seed) -----------------
    SEED = int(args.ramdn_seed)
    print(f"[Seed] Using seed: {SEED}")
    # Make hash-based operations deterministic
    os.environ['PYTHONHASHSEED'] = str(SEED)
    # For certain CUDA/cuBLAS nondeterministic ops (optional, may affect perf)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Python, numpy, torch RNGs
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Enforce deterministic algorithms where possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Device: {args.device}")
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("batch_size:", args.batch_size)
    trainer = Trainer(args)

    argsDict = args.__dict__
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    print('Total Epoches:', trainer.args.epochs)
    
    # move model and criterion to the trainer.device (set in Trainer)
    trainer.model = trainer.model.to(trainer.device)
    trainer.criterion = trainer.criterion.to(trainer.device)
    result_file = os.path.join(args.experiment_dir, "result2.txt")
    with open(result_file, "w") as f:
        f.write("==== Training Log Started ====\n")

    TOP_K = 5
    best_checkpoints = []
    for epoch in range(0, trainer.args.epochs):
        if epoch % 5 == 0 and epoch > 0:
            
            total_roc, total_pr, seen_roc, seen_pr, unseen_roc, unseen_pr = trainer.eval()

            log_line = (
                f"Epoch {epoch:03d} | "
                f"Overall ROC={total_roc:.4f}, PR={total_pr:.4f} | "
                f"Seen ROC={seen_roc:.4f}, PR={seen_pr:.4f} | "
                f"Unseen ROC={unseen_roc:.4f}, PR={unseen_pr:.4f}\n"
            )

            with open(result_file, "a") as f:
                f.write(log_line)

            print(f"[Eval] {log_line.strip()}  --> wrote {result_file}")

            score = total_roc

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

        trainer.training(epoch)

    trainer.eval()
    trainer.save_weights(args.savename)

