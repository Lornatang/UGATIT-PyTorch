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
import os
import random
import time

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize

from ugatit_pytorch import Generator

parser = argparse.ArgumentParser(description="PyTorch Generate Realistic Animation Face.")
parser.add_argument("--file", type=str, default="assets/testA_1.jpg",
                    help="Selfie image name. (default:`assets/testA_1.jpg`)")
parser.add_argument("--model-name", type=str, default="selfie2anime",
                    help="dataset name.  (default:`selfie2anime`)"
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, selfie2anime]")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:1" if args.cuda else "cpu")

# create model
model = Generator(image_size=args.image_size).to(device)

# Load state dicts
model.load_state_dict(torch.load(os.path.join("weights", str(args.model_name), "netG_A2B.pth")))

# Set model mode
model.eval()

# Load image
image = Image.open(args.file)
image = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(ToTensor()(Resize(args.image_size)(image))).unsqueeze(0)
image = image.to(device)

start = time.clock()
fake_image, _ = model(image)
elapsed = (time.clock() - start)
print(f"cost {elapsed:.4f}s")
vutils.save_image(fake_image.detach(), "result.jpg", normalize=True)
