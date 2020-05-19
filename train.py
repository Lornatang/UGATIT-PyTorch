# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import itertools
import os
import random
import time

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms

from ugatit_pytorch import Discriminator
from ugatit_pytorch import Generator
from ugatit_pytorch import ImageFolder
from ugatit_pytorch import bgr2rgb
from ugatit_pytorch import RhoClipper
from ugatit_pytorch import cam
from ugatit_pytorch import denorm
from ugatit_pytorch import tensor2numpy

parser = argparse.ArgumentParser(description="PyTorch Generate Realistic Animation Face.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("--dataset", type=str, default="selfie2anime",
                    help="dataset name.  (default:`selfie2anime`)"
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, selfie2anime]")
parser.add_argument("--light", action="store_true",
                    help="Enables U-GAT-IT light version, else Enables U-GAT-IT full version.")
parser.add_argument("--iteration", default=1000000, type=int, metavar="N",
                    help="The number of training iterations. (default:1000000)")
parser.add_argument("--image-size", type=int, default=128,
                    help="Size of the data crop (squared assumed). (default:256)")
parser.add_argument("-b", "--batch-size", default=1, type=int,
                    metavar="N",
                    help="mini-batch size (default: 1), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
parser.add_argument("--weight-decay", type=float, default=0.0001, help="The weight decay")
parser.add_argument("-p", "--print-freq", default=1000, type=int,
                    metavar="N", help="Print frequency. (default:1000)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG_A2B", default="", help="path to netG_A2B (to continue training)")
parser.add_argument("--netG_B2A", default="", help="path to netG_B2A (to continue training)")
parser.add_argument("--netD_A", default="", help="path to netD_A (to continue training)")
parser.add_argument("--netD_B", default="", help="path to netD_B (to continue training)")
parser.add_argument("--netL_A", default="", help="path to netL_A (to continue training)")
parser.add_argument("--netL_B", default="", help="path to netL_B (to continue training)")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs("weights")
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset
trainA_dataset = ImageFolder(root=os.path.join(args.dataroot, args.dataset, "train", "A"),
                             transform=transforms.Compose([
                                 transforms.Resize((args.image_size + 30, args.image_size + 30)),
                                 transforms.RandomCrop(args.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                             ]))

trainA_dataloader = iter(torch.utils.data.DataLoader(trainA_dataset, batch_size=args.batch_size,
                                                     shuffle=True, pin_memory=True))
trainB_dataset = ImageFolder(root=os.path.join(args.dataroot, args.dataset, "train", "B"),
                             transform=transforms.Compose([
                                 transforms.Resize((args.image_size + 30, args.image_size + 30)),
                                 transforms.RandomCrop(args.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                             ]))

trainB_dataloader = iter(torch.utils.data.DataLoader(trainB_dataset, batch_size=args.batch_size,
                                                     shuffle=True, pin_memory=True))

try:
    os.makedirs(os.path.join(args.outf, args.dataset, "A"))
    os.makedirs(os.path.join(args.outf, args.dataset, "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights", args.dataset))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
netG_A2B = Generator(image_size=args.image_size).to(device)
netG_B2A = Generator(image_size=args.image_size).to(device)
netD_A = Discriminator(5).to(device)
netD_B = Discriminator(5).to(device)
netL_A = Discriminator(7).to(device)
netL_B = Discriminator(7).to(device)

# load model
if args.netG_A2B != "":
    netG_A2B.load_state_dict(torch.load(args.netG_A2B))
if args.netG_B2A != "":
    netG_B2A.load_state_dict(torch.load(args.netG_B2A))
if args.netD_A != "":
    netD_A.load_state_dict(torch.load(args.netD_A))
if args.netD_B != "":
    netD_B.load_state_dict(torch.load(args.netD_B))
if args.netL_A != "":
    netL_A.load_state_dict(torch.load(args.netL_A))
if args.netL_B != "":
    netL_B.load_state_dict(torch.load(args.netL_B))

# define loss function (adversarial_loss) and optimizer
cycle_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)
identity_loss = torch.nn.BCEWithLogitsLoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters(),
                                               netL_A.parameters(), netL_B.parameters()),
                               lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

# Define Rho clipper to constraint the value of rho in AdaILN and ILN
Rho_clipper = RhoClipper(0, 1)

start_iter = 1
start_time = time.time()
for step in range(start_iter, args.iteration + 1):
    # Dynamic adjustment of learning rate
    if step > (args.iteration // 2):
        optimizer_G.param_groups[0]["lr"] -= (args.lr / (args.iteration // 2))
        optimizer_D.param_groups[0]["lr"] -= (args.lr / (args.iteration // 2))

    # get batch size data
    real_image_A, _ = next(trainA_dataloader)
    real_image_B, _ = next(trainB_dataloader)

    real_image_A, real_image_B = real_image_A.to(device), real_image_B.to(device)

    # Update D
    optimizer_D.zero_grad()

    fake_image_B, _, _ = netG_A2B(real_image_A)
    fake_image_A, _, _ = netG_B2A(real_image_B)

    real_output_GA, real_output_GA_cam, _ = netD_A(real_image_A)
    real_output_LA, real_output_LA_cam, _ = netL_A(real_image_A)
    real_output_GB, real_output_GB_cam, _ = netD_B(real_image_B)
    real_output_LB, real_output_LB_cam, _ = netL_B(real_image_B)

    fake_output_GA, fake_output_GA_cam, _ = netD_A(fake_image_A)
    fake_output_LA, fake_output_LA_cam, _ = netL_A(fake_image_A)
    fake_output_GB, fake_output_GB_cam, _ = netD_B(fake_image_B)
    fake_output_LB, fake_output_LB_cam, _ = netL_B(fake_image_B)

    D_real_adversarial_loss_GA = adversarial_loss(real_output_GA, torch.ones_like(real_output_GA, device=device))
    D_fake_adversarial_loss_GA = adversarial_loss(fake_output_GA, torch.zeros_like(fake_output_GA, device=device))
    D_adversarial_loss_GA = D_real_adversarial_loss_GA + D_fake_adversarial_loss_GA
    D_real_adversarial_loss_GB = adversarial_loss(real_output_GB, torch.ones_like(real_output_GB, device=device))
    D_fake_adversarial_loss_GB = adversarial_loss(fake_output_GB, torch.zeros_like(fake_output_GB, device=device))
    D_adversarial_loss_GB = D_real_adversarial_loss_GB + D_fake_adversarial_loss_GB

    D_real_adversarial_loss_cam_GA = adversarial_loss(real_output_GA_cam,
                                                      torch.ones_like(real_output_GA_cam, device=device))
    D_fake_adversarial_loss_cam_GA = adversarial_loss(fake_output_GA_cam,
                                                      torch.zeros_like(fake_output_GA_cam, device=device))
    D_adversarial_loss_cam_GA = D_real_adversarial_loss_cam_GA + D_fake_adversarial_loss_cam_GA
    D_real_adversarial_loss_cam_GB = adversarial_loss(real_output_GB_cam,
                                                      torch.ones_like(real_output_GB_cam, device=device))
    D_fake_adversarial_loss_cam_GB = adversarial_loss(fake_output_GB_cam,
                                                      torch.zeros_like(fake_output_GB_cam, device=device))
    D_adversarial_loss_cam_GB = D_real_adversarial_loss_cam_GB + D_fake_adversarial_loss_cam_GB

    D_real_adversarial_loss_LA = adversarial_loss(real_output_LA, torch.ones_like(real_output_LA, device=device))
    D_fake_adversarial_loss_LA = adversarial_loss(fake_output_LA, torch.zeros_like(fake_output_LA, device=device))
    D_adversarial_loss_LA = D_real_adversarial_loss_LA + D_fake_adversarial_loss_LA
    D_real_adversarial_loss_LB = adversarial_loss(real_output_LB, torch.ones_like(real_output_LB, device=device))
    D_fake_adversarial_loss_LB = adversarial_loss(fake_output_LB, torch.zeros_like(fake_output_LB, device=device))
    D_adversarial_loss_LB = D_real_adversarial_loss_LB + D_fake_adversarial_loss_LB

    D_real_adversarial_loss_cam_LA = adversarial_loss(real_output_LA_cam,
                                                      torch.ones_like(real_output_LA_cam, device=device))
    D_fake_adversarial_loss_cam_LA = adversarial_loss(fake_output_LA_cam,
                                                      torch.zeros_like(fake_output_LA_cam, device=device))
    D_adversarial_loss_cam_LA = D_real_adversarial_loss_cam_LA + D_fake_adversarial_loss_cam_LA
    D_real_adversarial_loss_cam_LB = adversarial_loss(real_output_LB_cam,
                                                      torch.ones_like(real_output_LB_cam, device=device))
    D_fake_adversarial_loss_cam_LB = adversarial_loss(fake_output_LB_cam,
                                                      torch.zeros_like(fake_output_LB_cam, device=device))
    D_adversarial_loss_cam_LB = D_real_adversarial_loss_cam_LB + D_fake_adversarial_loss_cam_LB

    loss_D_A = D_adversarial_loss_GA + D_adversarial_loss_cam_GA + D_adversarial_loss_LA + D_adversarial_loss_cam_LA
    loss_D_B = D_adversarial_loss_GB + D_adversarial_loss_cam_GB + D_adversarial_loss_LB + D_adversarial_loss_cam_LB

    errD = loss_D_A + loss_D_B
    errD.backward()
    optimizer_D.step()

    # Update G
    optimizer_G.zero_grad()

    fake_image_B, fake_image_B_cam_output, _ = netG_A2B(real_image_A)
    fake_image_A, fake_image_A_cam_output, _ = netG_B2A(real_image_B)

    fake_image_B2A, _, _ = netG_B2A(fake_image_B)
    fake_image_A2B, _, _ = netG_A2B(fake_image_A)

    fake_image_A2A, fake_image_A2A_cam_output, _ = netG_B2A(real_image_A)
    fake_image_B2B, fake_image_B2B_cam_output, _ = netG_A2B(real_image_B)

    fake_output_GA, fake_output_GA_cam, _ = netD_A(fake_image_A)
    fake_output_LA, fake_output_LA_cam, _ = netL_A(fake_image_A)
    fake_output_GB, fake_output_GB_cam, _ = netD_B(fake_image_B)
    fake_output_LB, fake_output_LB_cam, _ = netL_B(fake_image_B)

    G_adversarial_loss_GA = adversarial_loss(fake_output_GA, torch.ones_like(fake_output_GA, device=device))
    G_adversarial_loss_cam_GA = adversarial_loss(fake_output_GA_cam, torch.ones_like(fake_output_GA_cam, device=device))
    G_adversarial_loss_LA = adversarial_loss(fake_output_LA, torch.ones_like(fake_output_LA, device=device))
    G_adversarial_loss_cam_LA = adversarial_loss(fake_output_GA_cam, torch.ones_like(fake_output_GA_cam, device=device))
    G_adversarial_loss_GB = adversarial_loss(fake_output_GB, torch.ones_like(fake_output_GB, device=device))
    G_adversarial_loss_cam_GB = adversarial_loss(fake_output_GB_cam, torch.ones_like(fake_output_GB_cam, device=device))
    G_adversarial_loss_LB = adversarial_loss(fake_output_LB, torch.ones_like(fake_output_LB, device=device))
    G_adversarial_loss_cam_LB = adversarial_loss(fake_output_LB_cam, torch.ones_like(fake_output_LB_cam, device=device))

    G_recovered_loss_A = cycle_loss(fake_image_B2A, real_image_A)
    G_recovered_loss_B = cycle_loss(fake_image_A2B, real_image_B)

    G_identity_loss_A = cycle_loss(fake_image_A2A, real_image_A)
    G_identity_loss_B = cycle_loss(fake_image_B2B, real_image_B)

    G_real_loss_cam_A = identity_loss(fake_image_A_cam_output, torch.ones_like(fake_image_A_cam_output, device=device))
    G_fake_loss_cam_A = identity_loss(fake_image_A2A_cam_output,
                                      torch.zeros_like(fake_image_A2A_cam_output, device=device))
    G_cam_loss_A = G_real_loss_cam_A + G_fake_loss_cam_A

    G_real_loss_cam_B = identity_loss(fake_image_B_cam_output, torch.ones_like(fake_image_B_cam_output, device=device))
    G_fake_loss_cam_B = identity_loss(fake_image_B2B_cam_output,
                                      torch.zeros_like(fake_image_B2B_cam_output, device=device))
    G_cam_loss_B = G_real_loss_cam_B + G_fake_loss_cam_B

    G_adversarial_loss_A = G_adversarial_loss_GA + G_adversarial_loss_cam_GA + G_adversarial_loss_LA + G_adversarial_loss_cam_LA
    G_adversarial_loss_B = G_adversarial_loss_GB + G_adversarial_loss_cam_GB + G_adversarial_loss_LB + G_adversarial_loss_cam_LB
    loss_G_A = G_adversarial_loss_A + 10 * G_recovered_loss_A + 10 * G_identity_loss_A + 1000 * G_cam_loss_A
    loss_G_B = G_adversarial_loss_B + 10 * G_recovered_loss_B + 10 * G_identity_loss_B + 1000 * G_cam_loss_B

    errG = loss_G_A + loss_G_B
    errG.backward()
    optimizer_G.step()

    # clip parameter of AdaILN and ILN, applied after optimizer step
    netG_A2B.apply(Rho_clipper)
    netG_B2A.apply(Rho_clipper)

    print(f"[{step:5d}/{args.iteration:5d}] "
          f"time: {time.time() - start_time:4.4f}s "
          f"d_loss: {errD.item():.8f} "
          f"g_loss: {errG.item():.8f}.")
    if step % args.print_freq == 0:
        train_sample_num = 5
        A2B = np.zeros((args.image_size * 7, 0, 3))
        B2A = np.zeros((args.image_size * 7, 0, 3))

        fake_image_B, _, fake_image_B_heatmap = netG_A2B(real_image_A)
        fake_image_A, _, fake_image_A_heatmap = netG_B2A(real_image_B)

        fake_image_B2A, _, fake_image_B2A_heatmap = netG_B2A(fake_image_B)
        fake_image_A2B, _, fake_image_A2B_heatmap = netG_A2B(fake_image_A)

        fake_image_A2A, _, fake_image_A2A_heatmap = netG_B2A(real_image_A)
        fake_image_B2B, _, fake_image_B2B_heatmap = netG_A2B(real_image_B)

        A2B = np.concatenate((A2B, np.concatenate((bgr2rgb(tensor2numpy(denorm(real_image_A[0]))),
                                                   cam(tensor2numpy(fake_image_A2A_heatmap[0]), args.image_size),
                                                   bgr2rgb(tensor2numpy(denorm(fake_image_A2A[0]))),
                                                   cam(tensor2numpy(fake_image_B_heatmap[0]), args.image_size),
                                                   bgr2rgb(tensor2numpy(denorm(fake_image_B[0]))),
                                                   cam(tensor2numpy(fake_image_B2A_heatmap[0]), args.image_size),
                                                   bgr2rgb(tensor2numpy(denorm(fake_image_B2A[0])))), 0)), 1)

        B2A = np.concatenate((B2A, np.concatenate((bgr2rgb(tensor2numpy(denorm(real_image_B[0]))),
                                                   cam(tensor2numpy(fake_image_B2B_heatmap[0]), args.image_size),
                                                   bgr2rgb(tensor2numpy(denorm(fake_image_B2B[0]))),
                                                   cam(tensor2numpy(fake_image_A_heatmap[0]), args.image_size),
                                                   bgr2rgb(tensor2numpy(denorm(fake_image_A[0]))),
                                                   cam(tensor2numpy(fake_image_A2B_heatmap[0]), args.image_size),
                                                   bgr2rgb(tensor2numpy(denorm(fake_image_A2B[0])))), 0)), 1)

        cv2.imwrite(os.path.join(args.outf, args.dataset, "B", f"{step:07d}.png"), A2B * 255.0)
        cv2.imwrite(os.path.join(args.outf, args.dataset, "A", f"{step:07d}.png"), B2A * 255.0)

        # do check pointing
        torch.save(netG_A2B.state_dict(), f"weights/netG_A2B.pth")
        torch.save(netG_B2A.state_dict(), f"weights/netG_B2A.pth")
        torch.save(netD_A.state_dict(), f"weights/netD_A.pth")
        torch.save(netD_B.state_dict(), f"weights/netD_B.pth")
        torch.save(netL_A.state_dict(), f"weights/netL_A.pth")
        torch.save(netL_B.state_dict(), f"weights/netL_B.pth")
