import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import argparse
import torch
import torch.optim

from tqdm import tqdm
from utils import dataloader, model_util, data_util

from models.sde_net import SDENet
from models.schedule import Schedule
from models.diffusion_net import DiffNet
from models.diffusion_Sample import GaussianDiffusion
from sam import SamAutomaticMaskGenerator, sam_model_registry

from thop import profile

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--dnet_name', type=str, default="sde_net")
parser.add_argument('--net_name', type=str, default="ddc_diff")
parser.add_argument('--timesteps', type=int, default=500)
parser.add_argument('--schedule', type=str, default="linear")
parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--sam_type', type=str, default="vit_h")
parser.add_argument('--sam_checkpoint', type=str, default="./sam/weights/sam_vit_h_4b8939.pth")
parser.add_argument('--num_resblocks', type=int, default=1)
parser.add_argument('--dataset_name', type=str, default="UIEB")
parser.add_argument('--raw_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/test/")
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
parser.add_argument('--result_path', type=str, default="./results/")
config = parser.parse_args()

def test(config):
    d_net = SDENet(config.n_feats).cuda()
    model_util.load_checkpoint(d_net, os.path.join(config.checkpoint_path, config.dnet_name, 'model_best.pth'))

    net = DiffNet(config.n_feats, config.channel_mult, config.num_resblocks).cuda()
    schedule = Schedule(config.schedule, config.timesteps)
    diffusion = GaussianDiffusion(net.denoiser, config.timesteps, schedule).cuda()
    model_util.load_checkpoint(net, os.path.join(config.checkpoint_path, config.net_name, 'model_best.pth'))

    sam_name = config.sam_type
    sam = sam_model_registry[sam_name](checkpoint=config.sam_checkpoint).cuda()
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(os.path.join(config.raw_images_path, config.dataset_name))
    test_dataset = dataloader.test_loader(os.path.join(config.raw_images_path, config.dataset_name), mask_generator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                              num_workers=config.num_workers, drop_last=False, pin_memory=True)
    # create results folds
    if not os.path.exists(os.path.join(config.result_path, config.net_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name))

    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # lab_result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name, "LAB")
    # if not os.path.exists(lab_result_dir):
    #     os.mkdir(lab_result_dir)
    #
    # sam_result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name, "SAM")
    # if not os.path.exists(sam_result_dir):
    #     os.mkdir(sam_result_dir)

    d_net.eval()
    net.eval()

    with torch.no_grad():
        for _, (img_raw, img_mask, sam_features, filenames) in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_raw = img_raw.cuda()
            img_lab = data_util.rgb_to_lab(img_raw)
            img_noisy = torch.randn_like(img_raw).cuda()

            img_mask = img_mask.cuda()
            sam_features = sam_features.cuda()

            img_d = d_net(img_raw, img_mask, sam_features)
            pred_en = net.predictor(img_raw, img_lab, img_d)
            sampledImgs = diffusion(img_noisy, img_raw, pred_en, img_d)
            # sampledImgs = diffusion(img_noisy, img_raw, pred_en)

            data_util.save_image(sampledImgs, result_dir, filenames)
            # data_util.save_image(pred_en, lab_result_dir, filenames)
            # data_util.save_image(img_mask, sam_result_dir, filenames)

    dflops, dparams = profile(d_net, inputs=(img_raw, img_mask, sam_features))
    flops1, params1 = profile(net.predictor, inputs=(img_raw, img_lab, img_d))
    flops2, params2 = profile(diffusion, inputs=(img_noisy, img_raw, pred_en, img_d))

    print('dflops: %.4f G, dparams: %.4f M' % (dflops / 1e9, dparams / 1e6))
    print('flops1: %.4f G, params1: %.4f M' % (flops1 / 1e9, params1 / 1e6))
    print('flops2: %.4f G, params2: %.4f M' % (flops2 / 1e9, params2 / 1e6))
    print('all flops: %.4f G, all params: %.4f M' %
          ((dflops + flops1 + flops2) / 1e9, (dparams + params1 + params2) / 1e6))

if __name__ == '__main__':
    start_time = time.time()
    test(config)
    print("test_time:" + str((time.time() - start_time)))
