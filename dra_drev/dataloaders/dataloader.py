from dataloaders.datasets.open_industry import OpenIndustry
from dataloaders.datasets.anomaly_shapenet import AnomalyShapeNet
from torch.utils.data import DataLoader
import torch
import random
import numpy as np


class initDataloader():

    @staticmethod
    def build(args, **kwargs):

        assert args.ramdn_seed is not None, "ramdn_seed must be set for reproducible dataloading"

        g = torch.Generator()
        g.manual_seed(int(args.ramdn_seed))

        def worker_init_fn(worker_id):
            worker_seed = int(args.ramdn_seed) + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        if args.dataset == "open_industry":
            train_set = OpenIndustry(args, train=True)
            test_set = OpenIndustry(args, train=False)
            train_set._init_feature_extractor()
            test_set._init_feature_extractor()

        elif args.dataset == "anomaly_shapenet":
            train_set = AnomalyShapeNet(args, train=True)
            test_set = AnomalyShapeNet(args, train=False)
            train_set._init_feature_extractor()
            test_set._init_feature_extractor()

        else:
            raise NotImplementedError(f"Unsupported dataset: {args.dataset}")

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            generator=g,
            **kwargs
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            generator=g,
            **kwargs
        )

        return train_loader, test_loader
