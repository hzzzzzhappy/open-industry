from data.real3d import get_real_loader,real3d_classes
from data.anomalyshape import get_shapenet_loader,shapenet3d_classes
from data.mc3dad import get_mbc3dad_loader
import torch
from tqdm import tqdm
from feature_extractors.FPFH import FPFHFeatures
import numpy as np
import os
from feature_extractors.pointnet2_utils import *

import numpy as np
import open3d as o3d
import torch


class PatchCore():
    def __init__(self, ckp = '', image_size=224, args=None):
        self.args = args
        self.method  = FPFHFeatures(args=args)
        self.dataset_name = self.args.dataset
        self.level = self.args.level
        self.known_defects = args.known_defects
        self.pollution_per_defect = args.pollution_per_defect

    def get_dataloader(self,dataset_name,split,class_name,level='ALL',seed = 0):
        if dataset_name == 'real':
            return get_real_loader(split, class_name=class_name,known_defects = self.known_defects,pollution_per_defect=self.pollution_per_defect,seed =seed)
        if dataset_name == 'shapenet':
            return get_shapenet_loader(split, class_name=class_name,known_defects = self.known_defects,pollution_per_defect=self.pollution_per_defect,seed =seed)
        if dataset_name == 'mc3dad':
            return get_mbc3dad_loader(split, class_name=class_name,known_defects = self.known_defects,pollution_per_defect=self.pollution_per_defect,seed =seed)
    def fit(self, class_name,abnormal_ratio_threshold = 0.4,seed = 0):
        train_loader = self.get_dataloader(self.dataset_name,'train',class_name,level=self.level,seed=seed)
        for pc, mask, sample_label, path in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            self.method.collect_features(pc, mask=mask, abnormal_ratio_threshold=abnormal_ratio_threshold)
            self.method.name_list.append(path)
        print(f'\n\nRunning coreset on class {class_name}...')
        self.method.run_coreset()
        self.method.purify_abnormal_lib_by_normal_nn_delete()
        self.method.run_coreset(use_ab_coreset = True)


    def evaluate(self, class_name, lam = 0.3,seed = 0):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = self.get_dataloader(self.dataset_name,'test',class_name,level=self.level ,seed = seed)
        with torch.no_grad():
            self.method.init_para()
            self.method.name_list = []
            self.method.test_patch_lib = []
            print("Enable the abnormal patch library for normality score computation")
            for pc, mask, label, path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                self.method.predict(pc, mask, label,path)
            self.method.fuse_scores_and_store_back(lam = lam)
        method_name = "Simple3D"
        self.method.calculate_metrics()
        image_rocaucs[method_name] = round(self.method.image_rocauc, 4)
        pixel_rocaucs[method_name] = round(self.method.pixel_rocauc, 4)
        au_pros[method_name] = round(self.method.au_pro, 4)
        print(
            f'Class: {class_name}, {method_name} Image ROCAUC: {self.method.image_rocauc:.4f}, {method_name} Pixel ROCAUC: {self.method.pixel_rocauc:.4f}, {method_name} AU-PRO: {self.method.au_pro:.4f}')
        return image_rocaucs, pixel_rocaucs, au_pros

