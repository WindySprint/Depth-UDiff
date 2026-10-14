import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import time
import argparse
import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from models import loss
from utils import dataloader, data_util

from models.schedule import Schedule
from models.diffusion_net import DiffNet
from models.diffusion_Sample import GaussianDiffusion

# torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument('--net_name', type=str, default="ddc_diff")
parser.add_argument('--timesteps', type=int, default=500)
parser.add_argument('--schedule', type=str, default="linear")
parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--num_resblocks', type=int, default=1)
parser.add_argument('--gt_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/train/UL/gt/")
parser.add_argument('--raw_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/train/UL/input/")
parser.add_argument('--d_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/train/UL/Depth/")
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--val_epoch', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--checkpoint_path', type=str, default="checkpoints/")
parser.add_argument('--out_img', type=bool, default=True)
config = parser.parse_args()

def train(config):
    ######### Model ###########
    net = DiffNet(config.n_feats, config.channel_mult, config.num_resblocks).cuda()
    schedule = Schedule(config.schedule, config.timesteps)
    diffusion = GaussianDiffusion(net.denoiser, config.timesteps, schedule).cuda()

    if not os.path.exists(os.path.join(config.checkpoint_path, config.net_name)):
        os.mkdir(os.path.join(config.checkpoint_path, config.net_name))

    ######### Loss ###########
    criterion_lab = loss.LAB_Loss(1.0, 0.5)
    criterion_mse = loss.MSE_Loss()
    criterion_vgg = loss.VGG_Loss()

    train_dataset = dataloader.train_val_loader(config.gt_images_path, config.raw_images_path, config.d_images_path, 'train')
    val_dataset = dataloader.train_val_loader(config.gt_images_path, config.raw_images_path, config.d_images_path, 'val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

    ######### Adam optimizer ###########
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
                                                            eta_min=config.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    # Record best index and corresponding epoch
    best_psnr = 0
    best_epoch = 0

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        # Record train loss and validation index
        train_loss = []
        val_psnr = []
        print("*" * 30 + "The %i epoch" % epoch + "*" * 30 + '\n')

        net.train()
        for _, (img_gt, img_raw, img_d) in enumerate(tqdm(train_loader)):
            t = torch.randint(0, config.timesteps, (config.train_batch_size,)).long().cuda()
            img_gt = img_gt.cuda()
            img_raw = img_raw.cuda()
            img_lab = data_util.rgb_to_lab(img_raw)
            img_d = img_d.cuda()

            try:
                denoised_img, noise_ref, pred_en = net(img_gt, img_raw, img_lab, img_d, t, diffusion)
                loss_lab = criterion_lab(pred_en, img_gt)
                # Compute loss at each stage
                if config.out_img:
                    loss_mse = criterion_mse(denoised_img, img_gt)
                    loss_vgg = criterion_vgg(denoised_img, img_gt)
                else:
                    loss_mse = criterion_mse(denoised_img, noise_ref)
                    loss_vgg = criterion_vgg(denoised_img, noise_ref)

                sum_loss = loss_mse + 0.2 * loss_vgg + 0.1 * loss_lab

                train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format
              (epoch, time.time() - epoch_start_time, np.mean(train_loss), scheduler.get_lr()[0]))
        with open(os.path.join(config.checkpoint_path, config.net_name, "loss.log"), "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % (epoch, np.mean(train_loss)) + "\n"
            f.write(s)

        if epoch % config.val_epoch == 0:
            print('===> Model Verifying')
            #### Evaluation ####
            net.eval()
            with torch.no_grad():
                for _, (img_gt, img_raw, img_d) in enumerate(val_loader):
                    img_gt = img_gt.cuda()
                    img_raw = img_raw.cuda()
                    img_lab = data_util.rgb_to_lab(img_raw)
                    img_d = img_d.cuda()
                    img_noisy = torch.randn_like(img_raw).cuda()

                    pred_en = net.predictor(img_raw, img_lab, img_d)
                    sampledImgs = diffusion(img_noisy, img_raw, pred_en, img_d)
                    # sampledImgs = diffusion(img_noisy, img_raw, pred_en)

                    if config.out_img:
                        psnr = loss.torchPSNR(img_gt, sampledImgs)
                    else:
                        psnr = loss.torchPSNR(img_noisy, sampledImgs)

                    val_psnr.append(psnr.item())

            val_psnr = np.mean(np.array(val_psnr))

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                torch.save({'state_dict': net.state_dict()},
                           os.path.join(config.checkpoint_path, config.net_name, "model_best.pth"))

            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                  (epoch, val_psnr, best_epoch, best_psnr))
            with open(os.path.join(config.checkpoint_path, config.net_name, "val_PSNR.log"), "a+",
                      encoding="utf-8") as f:
                f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                        (epoch, val_psnr, best_epoch, best_psnr) + "\n")

            scheduler.step()
            torch.save({'state_dict': net.state_dict()},
                       os.path.join(config.checkpoint_path, config.net_name, "model_latest.pth"))

if __name__ == "__main__":
    start_time = time.time()
    train(config)
    e = time.time()
    print("train_time:" + str(time.time() - start_time))
