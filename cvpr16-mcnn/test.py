import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

from src.crowd_count import CrowdCounter
from src.data_loader import ImageDataLoader
from src import utils

# cudnn
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True         # cudnn accelerates model operation on GPU.
    torch.backends.cudnn.benchmark = False      # it's better to turn it off when each image has a different size.

# argparse
parser = argparse.ArgumentParser(description='cvpr16-mcnn (github.com/unique-chan)')
parser.add_argument('--data_path', type=str)
parser.add_argument('--gt_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--pre_load', action='store_true')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--save_output', action='store_true')
args = parser.parse_args()

# make directories for inference results
model_name = os.path.basename(args.model_path).split('.')[0]
result_file = os.path.join(args.output_dir, f'results_{model_name}.txt')
density_maps_path = os.path.join(args.output_dir, f'density_maps_{model_name}')
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(density_maps_path, exist_ok=True)

# crowd-counter
net = CrowdCounter()
net.load_net(args.model_path)  # h5 -> net
if torch.cuda.is_available():
    net.cuda()
    net.eval()

# inference
with torch.no_grad:
    mae, mse = 0.0, 0.0
    loader = ImageDataLoader(args.data_path, args.gt_path, shuffle=False, gt_downsample=True, pre_load=True)
    for blob in tqdm(loader, desc='Test Inference ', mininterval=0.01):
        img, gt_density = blob['data'], blob['gt_density']
        estimated_density = net(img, gt_density)
        estimated_density = estimated_density.data.cpu().numpy()  # gpu -> cpu -> ram
        gt_cnt = np.sum(gt_density)
        est_cnt = np.sum(estimated_density)
        dif = gt_cnt - est_cnt
        mae += abs(dif)
        mse += np.power(dif, 2)
        if args.vis:
            utils.display_result(img, gt_density, estimated_density)
        if args.save_output:
            utils.save_density_map(estimated_density, density_maps_path, f"output_{blob['fname'].split('.')[0]}.png")
    time.sleep(0.01)
    mae = mae / len(loader)
    mse = np.sqrt(mse / len(loader))
    result_msg = f'MAE: {mae:.2f}, (R)MSE: {mse:.2f}'
    print(result_msg)

# store the result
f = open(result_file, 'w')
f.write(result_msg)
f.close()
