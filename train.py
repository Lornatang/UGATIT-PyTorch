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

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from ugatit_pytorch import DecayLR
from ugatit_pytorch import Discriminator
from ugatit_pytorch import Generator
from ugatit_pytorch import ImageDataset
from ugatit_pytorch import RhoClipper

parser = argparse.ArgumentParser(description="PyTorch Generate Realistic Animation Face.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("--dataset", type=str, default="selfie2anime",
                    help="dataset name.  (default:`selfie2anime`)"
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, selfie2anime]")
parser.add_argument("--light", action="store_true",
                    help="Enables U-GAT-IT light version, else Enables U-GAT-IT full version.")
parser.add_argument("--epochs", default=300, type=int, metavar="N",
                    help="number of total epochs to run. (default:300)")
parser.add_argument("--image-size", type=int, default=256,
                    help="Size of the data crop (squared assumed). (default:256)")
parser.add_argument("--decay_epochs", type=int, default=150,
                    help="epoch to start linearly decaying the learning rate to 0. (default:150)")
parser.add_argument("-b", "--batch-size", default=1, type=int,
                    metavar="N",
                    help="mini-batch size (default: 1), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
parser.add_argument("--weight-decay", type=float, default=0.0001, help="The weight decay")
parser.add_argument("-p", "--print-freq", default=100, type=int,
                    metavar="N", help="Print frequency. (default:100)")
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
dataset = ImageDataset(root=os.path.join(args.dataroot, args.dataset),
                       transform=transforms.Compose([
                           transforms.Resize((args.image_size + 30, args.image_size + 30)),
                           transforms.RandomCrop(args.image_size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ]),
                       unaligned=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, pin_memory=True)

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

lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)

# Define Rho clipper to constraint the value of rho in AdaILN and ILN
Rho_clipper = RhoClipper(0, 1)

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # get batch size data
        real_image_A = data["A"].to(device)
        real_image_B = data["B"].to(device)

        # Update D
        optimizer_D.zero_grad()

        fake_image_B, _ = netG_A2B(real_image_A)
        fake_image_A, _ = netG_B2A(real_image_B)

        real_output_GA, real_output_GA_cam = netD_A(real_image_A)
        real_output_LA, real_output_LA_cam = netL_A(real_image_A)
        real_output_GB, real_output_GB_cam = netD_B(real_image_B)
        real_output_LB, real_output_LB_cam = netL_B(real_image_B)

        fake_output_GA, fake_output_GA_cam = netD_A(fake_image_A.detach())
        fake_output_LA, fake_output_LA_cam = netL_A(fake_image_A.detach())
        fake_output_GB, fake_output_GB_cam = netD_B(fake_image_B.detach())
        fake_output_LB, fake_output_LB_cam = netL_B(fake_image_B.detach())

        D_real_adversarial_loss_GA = adversarial_loss(real_output_GA, torch.ones_like(real_output_GA, device=device))
        D_fake_adversarial_loss_GA = adversarial_loss(fake_output_GA, torch.zeros_like(fake_output_GA, device=device))
        D_adversarial_loss_GA = D_real_adversarial_loss_GA + D_fake_adversarial_loss_GA

        D_real_adversarial_loss_GB = adversarial_loss(real_output_GB, torch.ones_like(real_output_GB, device=device))
        D_fake_adversarial_loss_GB = adversarial_loss(fake_output_GB, torch.zeros_like(fake_output_GB, device=device))
        D_adversarial_loss_GB = D_real_adversarial_loss_GB + D_fake_adversarial_loss_GB

        D_real_adversarial_loss_GA_cam = adversarial_loss(real_output_GA_cam,
                                                          torch.ones_like(real_output_GA_cam, device=device))
        D_fake_adversarial_loss_GA_cam = adversarial_loss(fake_output_GA_cam,
                                                          torch.zeros_like(fake_output_GA_cam, device=device))
        D_adversarial_loss_GA_cam = D_real_adversarial_loss_GA_cam + D_fake_adversarial_loss_GA_cam

        D_real_adversarial_loss_GB_cam = adversarial_loss(real_output_GB_cam,
                                                          torch.ones_like(real_output_GB_cam, device=device))
        D_fake_adversarial_loss_GB_cam = adversarial_loss(fake_output_GB_cam,
                                                          torch.zeros_like(fake_output_GB_cam, device=device))
        D_adversarial_loss_GB_cam = D_real_adversarial_loss_GB_cam + D_fake_adversarial_loss_GB_cam

        D_real_adversarial_loss_LA = adversarial_loss(real_output_LA, torch.ones_like(real_output_LA, device=device))
        D_fake_adversarial_loss_LA = adversarial_loss(fake_output_LA, torch.zeros_like(fake_output_LA, device=device))
        D_adversarial_loss_LA = D_real_adversarial_loss_LA + D_fake_adversarial_loss_LA

        D_real_adversarial_loss_LB = adversarial_loss(real_output_LB, torch.ones_like(real_output_LB, device=device))
        D_fake_adversarial_loss_LB = adversarial_loss(fake_output_LB, torch.zeros_like(fake_output_LB, device=device))
        D_adversarial_loss_LB = D_real_adversarial_loss_LB + D_fake_adversarial_loss_LB

        D_real_adversarial_loss_LA_cam = adversarial_loss(real_output_LA_cam,
                                                          torch.ones_like(real_output_LA_cam, device=device))
        D_fake_adversarial_loss_LA_cam = adversarial_loss(fake_output_LA_cam,
                                                          torch.zeros_like(fake_output_LA_cam, device=device))
        D_adversarial_loss_LA_cam = D_real_adversarial_loss_LA_cam + D_fake_adversarial_loss_LA_cam

        D_real_adversarial_loss_LB_cam = adversarial_loss(real_output_LB_cam,
                                                          torch.ones_like(real_output_LB_cam, device=device))
        D_fake_adversarial_loss_LB_cam = adversarial_loss(fake_output_LB_cam,
                                                          torch.zeros_like(fake_output_LB_cam, device=device))
        D_adversarial_loss_LB_cam = D_real_adversarial_loss_LB_cam + D_fake_adversarial_loss_LB_cam

        loss_D_A = D_adversarial_loss_GA + D_adversarial_loss_GA_cam + D_adversarial_loss_LA + D_adversarial_loss_LA_cam
        loss_D_B = D_adversarial_loss_GB + D_adversarial_loss_GB_cam + D_adversarial_loss_LB + D_adversarial_loss_LB_cam

        errD = loss_D_A + loss_D_B
        errD.backward()
        optimizer_D.step()

        # Update G
        optimizer_G.zero_grad()

        fake_image_B, fake_image_B_cam = netG_A2B(real_image_A)
        fake_image_A, fake_image_A_cam = netG_B2A(real_image_B)

        fake_image_B2A, _ = netG_B2A(fake_image_B.detach())
        fake_image_A2B, _ = netG_A2B(fake_image_A.detach())

        fake_image_A2A, fake_output_A2A_cam = netG_B2A(real_image_A)
        fake_image_B2B, fake_output_B2B_cam = netG_A2B(real_image_B)

        fake_output_GA, fake_output_GA_cam = netD_A(fake_image_A.detach())
        fake_output_LA, fake_output_LA_cam = netL_A(fake_image_A.detach())
        fake_output_GB, fake_output_GB_cam = netD_B(fake_image_B.detach())
        fake_output_LB, fake_output_LB_cam = netL_B(fake_image_B.detach())

        G_adversarial_loss_GA = adversarial_loss(fake_output_GA, torch.ones_like(fake_output_GA, device=device))
        G_adversarial_loss_GA_cam = adversarial_loss(fake_output_GA_cam,
                                                     torch.ones_like(fake_output_GA_cam, device=device))
        G_adversarial_loss_LA = adversarial_loss(fake_output_LA, torch.ones_like(fake_output_LA, device=device))
        G_adversarial_loss_LA_cam = adversarial_loss(fake_output_GA_cam,
                                                     torch.ones_like(fake_output_GA_cam, device=device))
        G_adversarial_loss_GB = adversarial_loss(fake_output_GB, torch.ones_like(fake_output_GB, device=device))
        G_adversarial_loss_GB_cam = adversarial_loss(fake_output_GB_cam,
                                                     torch.ones_like(fake_output_GB_cam, device=device))
        G_adversarial_loss_LB = adversarial_loss(fake_output_LB, torch.ones_like(fake_output_LB, device=device))
        G_adversarial_loss_LB_cam = adversarial_loss(fake_output_LB_cam,
                                                     torch.ones_like(fake_output_LB_cam, device=device))

        G_recovered_loss_A = cycle_loss(fake_image_B2A, real_image_A)
        G_recovered_loss_B = cycle_loss(fake_image_A2B, real_image_B)

        G_identity_loss_A = cycle_loss(fake_image_A2A, real_image_A)
        G_identity_loss_B = cycle_loss(fake_image_B2B, real_image_B)

        G_real_loss_A_cam = identity_loss(fake_image_A_cam, torch.ones_like(fake_image_A_cam, device=device))
        G_fake_loss_A_cam = identity_loss(fake_output_A2A_cam, torch.zeros_like(fake_output_A2A_cam, device=device))
        G_loss_A_cam = G_real_loss_A_cam + G_fake_loss_A_cam

        G_real_loss_B_cam = identity_loss(fake_image_B_cam, torch.ones_like(fake_image_B_cam, device=device))
        G_fake_loss_B_cam = identity_loss(fake_output_B2B_cam, torch.zeros_like(fake_output_B2B_cam, device=device))
        G_loss_B_cam = G_real_loss_B_cam + G_fake_loss_B_cam

        G_adversarial_loss_A = G_adversarial_loss_GA + G_adversarial_loss_GA_cam + G_adversarial_loss_LA + G_adversarial_loss_LA_cam
        G_adversarial_loss_B = G_adversarial_loss_GB + G_adversarial_loss_GB_cam + G_adversarial_loss_LB + G_adversarial_loss_LB_cam
        loss_G_A = G_adversarial_loss_A + 10 * G_recovered_loss_A + 10 * G_identity_loss_A + 1000 * G_loss_A_cam
        loss_G_B = G_adversarial_loss_B + 10 * G_recovered_loss_B + 10 * G_identity_loss_B + 1000 * G_loss_B_cam

        errG = loss_G_A + loss_G_B
        errG.backward()
        optimizer_G.step()

        # clip parameter of AdaILN and ILN, applied after optimizer step
        netG_A2B.apply(Rho_clipper)
        netG_B2A.apply(Rho_clipper)

        progress_bar.set_description(
            f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {errD.item():.4f} "
            f"Loss_G: {errG.item():.4f}")

        if i % args.print_freq == 0:
            vutils.save_image(real_image_A,
                              f"{args.outf}/{args.dataset}/A/real_samples.png",
                              normalize=True)
            vutils.save_image(real_image_B,
                              f"{args.outf}/{args.dataset}/B/real_samples.png",
                              normalize=True)

            fake_image_B, _ = netG_A2B(real_image_A)
            fake_image_A, _ = netG_B2A(real_image_B)

            vutils.save_image(fake_image_A.detach(),
                              f"{args.outf}/{args.dataset}/A/fake_samples_epoch_{epoch}.png",
                              normalize=True)
            vutils.save_image(fake_image_B.detach(),
                              f"{args.outf}/{args.dataset}/B/fake_samples_epoch_{epoch}.png",
                              normalize=True)
    # do check pointing
    torch.save(netG_A2B.state_dict(), f"weights/{args.dataset}/netG_A2B_epoch_{epoch}.pth")
    torch.save(netG_B2A.state_dict(), f"weights/{args.dataset}/netG_B2A_epoch_{epoch}.pth")
    torch.save(netD_A.state_dict(), f"weights/{args.dataset}/netD_A_epoch_{epoch}.pth")
    torch.save(netD_B.state_dict(), f"weights/{args.dataset}/netD_B_epoch_{epoch}.pth")
    torch.save(netL_A.state_dict(), f"weights/{args.dataset}/netL_A_epoch_{epoch}.pth")
    torch.save(netL_B.state_dict(), f"weights/{args.dataset}/netL_B_epoch_{epoch}.pth")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
