import argparse
from patchcore_runner import PatchCore
from data.real3d import real3d_classes
from data.anomalyshape import shapenet3d_classes
from data.mc3dad import mbc3dad_classes
# from data.quan import quan_classes
import pandas as pd
import torchvision
import os

def write_experiment_log(expname, strs):
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{expname}.txt")
    with open(log_path, "a") as f:
        f.write(strs)


def run_3d_ads(args):

    if args.dataset == 'real':
        classes = real3d_classes()
    if args.dataset == 'shapenet':
        classes = shapenet3d_classes()
    if args.dataset == 'mc3dad':
        classes = mbc3dad_classes()
    METHOD_NAMES = [
        "Simple3D",
        ]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for category in classes:
        patchcore = PatchCore(args=args)
        patchcore.fit(category ,abnormal_ratio_threshold = args.abnormal_ratio_threshold ,seed = args.seed)
        cls = category
        write_experiment_log(args.expname,f"\nRunning on class {cls} , abnormal_ratio_threshold = {args.abnormal_ratio_threshold} , lam = {args.lam}\n")
        image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(cls , lam = args.lam,seed = args.seed)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)

        print(f"\nFinished running on class {cls}\n")
        write_experiment_log(
            args.expname,
            f"[{cls}] Image ROC-AUC: {image_rocaucs}\n"
        )
        write_experiment_log(
            args.expname,
            f"[{cls}] Pixel ROC-AUC: {pixel_rocaucs}\n"
        )
        write_experiment_log(
            args.expname,
            f"[{cls}] AU-PRO: {au_pros}\n"
        )
        write_experiment_log(args.expname,f"Finished running on class {cls}\n")
        print("################################################################################\n\n")
        

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),4)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),4)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),4)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open3dad anomaly detection")

    # basic
    parser.add_argument("--expname", type=str, default="None", help="exp name")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--dataset", type=str, default="shapenet", help="dataset")

    # grouping
    parser.add_argument("--max_nn", type=int, default=100, help="max neighbors")
    parser.add_argument("--num_group", type=int, default=2048, help="num groups")
    parser.add_argument("--group_size", type=int, default=128, help="group size")

    # modules
    parser.add_argument("--use_MSND", type=bool, default=False, help="use MSND")
    parser.add_argument("--use_LFSA", type=bool, default=False, help="use LFSA")
    parser.add_argument("--vis_save", type=bool, default=False, help="save vis")
    parser.add_argument("--num_MSND", type=int, default=2, help="MSND blocks")

    # feature
    parser.add_argument("--feature", type=str, default="FPFH", help="feature type")
    parser.add_argument("--level", type=str, default="ALL", help="eval level")

    # open-set
    parser.add_argument("--known_defects", type=str, nargs="+", default=None, help="known defects")
    parser.add_argument("--pollution_per_defect", type=int, default=0, help="pollution per defect")
    parser.add_argument("--abnormal_ratio_threshold", type=float, default=0.4, help="abnormal ratio thr")
    parser.add_argument("--lam", type=float, default=0.1, help="lambda weight")

    # reproducibility
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    print(args)

    run_3d_ads(args)
