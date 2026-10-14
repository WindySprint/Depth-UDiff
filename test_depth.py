import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import argparse
import torch
import torch.optim
from tqdm import tqdm

from models.sde_net import SDENet
from utils import dataloader, model_util, data_util
from sam import SamAutomaticMaskGenerator, sam_model_registry

from thop import profile

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--dnet_name', type=str, default="sde_net")
parser.add_argument('--sam_type', type=str, default="vit_h")
parser.add_argument('--sam_checkpoint', type=str, default="./sam/weights/sam_vit_h_4b8939.pth")
parser.add_argument('--dataset_name', type=str, default="UIEB")
parser.add_argument('--raw_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/test")
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
parser.add_argument('--result_path', type=str, default="./results/")
config = parser.parse_args()

def test(config):
    d_net = SDENet(config.n_feats).cuda()
    model_util.load_checkpoint(d_net, os.path.join(config.checkpoint_path, config.dnet_name, 'model_best.pth'))

    sam_name = config.sam_type
    sam = sam_model_registry[sam_name](checkpoint=config.sam_checkpoint).cuda()
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(os.path.join(config.raw_images_path, config.dataset_name))
    test_dataset = dataloader.test_loader(os.path.join(config.raw_images_path, config.dataset_name), mask_generator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                              num_workers=config.num_workers, drop_last=False, pin_memory=True)
    # create results folds
    if not os.path.exists(os.path.join(config.result_path, config.dnet_name)):
        os.mkdir(os.path.join(config.result_path, config.dnet_name))

    if not os.path.exists(os.path.join(config.result_path, config.dnet_name, config.dataset_name)):
        os.mkdir(os.path.join(config.result_path, config.dnet_name, config.dataset_name))

    result_dir = os.path.join(config.result_path, config.dnet_name, config.dataset_name)

    d_net.eval()

    with torch.no_grad():
        for _, (img_raw, img_mask, sam_features, filenames) in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_raw = img_raw.cuda()
            img_mask = img_mask.cuda()
            sam_features = sam_features.cuda()

            img_d = d_net(img_raw, img_mask, sam_features)
            data_util.save_image(img_d, result_dir, filenames)

    flops, params = profile(d_net, inputs=(img_raw, img_mask, sam_features))
    print('flops: %.4f G, params: %.4f M' % (flops / (1e9), params / (1e6)))


if __name__ == '__main__':
    start_time = time.time()
    test(config)
    print("test_time:" + str((time.time() - start_time)))
