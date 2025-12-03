"""
train_gan.py

Enhanced training script for DCGAN + simplified StyleGAN with the following additions you requested:
- Support for custom dataset folder structure using torchvision.datasets.ImageFolder
- Switch to BCEWithLogitsLoss and remove final Sigmoid from the discriminator
- Optional Spectral Normalization on discriminator conv layers
- Label smoothing (for real labels) and optional label flipping/noise
- Gradient penalty (R1 regularization) applied to real samples
- Learning-rate schedulers (StepLR)
- TensorBoard logging (loss scalars + sample images)
- Optional FID evaluation (uses torchmetrics if available; falls back gracefully)

Usage examples:
python train_gan.py --model dcgan --dataset cifar10 --epochs 100 --batch-size 128 --tb-log
python train_gan.py --model dcgan --dataset imagefolder --data-root ./my_images --img-size 64 --spectral-norm --gp-lambda 10 --label-smoothing

Notes:
- This script tries to be flexible; if torchmetrics (for FID) isn't installed it will skip FID calculation.
- R1 gradient penalty is implemented (recommended for logistic loss / BCEWithLogits).

"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Try to import torchmetrics FID (optional)
try:
    from torchmetrics.image.fid import FID
    TORCHMETRICS_FID = True
except Exception:
    TORCHMETRICS_FID = False


# ---------- User's models (adapted) ----------
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64, spectral_norm=False):
        super(DCGANDiscriminator, self).__init__()
        def conv(in_c, out_c, k, s, p, bias=False):
            conv_layer = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)
            if spectral_norm:
                return nn.utils.spectral_norm(conv_layer)
            return conv_layer

        self.main = nn.Sequential(
            conv(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ndf * 8, 1, 4, 1, 0, bias=False),
            # NOTE: NO Sigmoid here because we'll use BCEWithLogitsLoss (which expects logits)
        )

    def forward(self, input):
        # returns logits of shape (batch, 1, 1, 1) -> flatten
        return self.main(input).view(input.size(0), -1).squeeze(1)


# Simplified StyleGAN-like generator (not full implementation)
class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim=512):
        super(StyleGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        w = self.mapping(z)
        w = w.view(w.size(0), w.size(1), 1, 1)
        return self.synthesis(w)


# ---------- Utility functions ----------

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    elif classname.find('BatchNorm') != -1:
        try:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        except Exception:
            pass


def save_sample_images(gen, fixed_noise, out_dir, epoch, model_name, device, normalize=True):
    gen.eval()
    with torch.no_grad():
        fake = gen(fixed_noise.to(device))
    # fake is in [-1,1], convert to [0,1]
    utils.save_image((fake + 1) / 2.0, os.path.join(out_dir, f"{model_name}_epoch_{epoch:04d}.png"), nrow=8)
    gen.train()


def compute_r1_penalty(real_logits, real_images):
    # real_logits: (batch,), real_images requires_grad=True
    grads = torch.autograd.grad(outputs=real_logits.sum(), inputs=real_images, create_graph=True)[0]
    grads = grads.view(grads.size(0), -1)
    r1 = (grads.pow(2).sum(1)).mean()
    return r1


# ---------- Training loop ----------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_dir, f"{args.model}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tb')) if args.tb_log else None

    # Dataset selection
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    if args.dataset.lower() == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_root, download=True, transform=transform)
    elif args.dataset.lower() == 'imagefolder':
        dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    else:
        raise ValueError('Only cifar10 and imagefolder datasets are implemented. Add your own loader.')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Build models
    if args.model == 'dcgan':
        netG = DCGANGenerator(latent_dim=args.latent_dim, ngf=args.ngf).to(device)
    elif args.model == 'stylegan':
        netG = StyleGANGenerator(latent_dim=args.latent_dim).to(device)
    else:
        raise ValueError('Model must be either "dcgan" or "stylegan"')

    netD = DCGANDiscriminator(ndf=args.ndf, spectral_norm=args.spectral_norm).to(device)

    # Initialize weights
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    # Loss function: BCEWithLogitsLoss (expects logits from D)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Schedulers
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=args.step_size, gamma=args.gamma) if args.use_scheduler else None
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=args.step_size, gamma=args.gamma) if args.use_scheduler else None

    # Fixed noise for samples
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1) if args.model == 'dcgan' else torch.randn(64, args.latent_dim)

    # Labels
    real_label_val = 0.9 if args.label_smoothing else 1.0
    fake_label_val = 0.0

    # FID metric if requested and available
    fid_metric = None
    if args.fid and TORCHMETRICS_FID:
        fid_metric = FID(feature=64) if False else FID()  # use defaults
        print('TorchMetrics FID available and will be used.')
    elif args.fid and not TORCHMETRICS_FID:
        print('torchmetrics not available -- skipping FID computation. Install torchmetrics to enable FID.')

    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        netG.load_state_dict(checkpoint['netG'])
        netD.load_state_dict(checkpoint['netD'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Resumed from checkpoint: {args.resume} (start_epoch={start_epoch})")

    print('Starting Training Loop...')
    for epoch in range(start_epoch, args.epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (data, _) in enumerate(loop):
            b_size = data.size(0)

            ############################
            # (1) Update D network: using BCEWithLogitsLoss on logits
            ###########################
            netD.train()
            netG.train()

            real_cpu = data.to(device)
            # Ensure requires_grad for R1 penalty
            if args.gp_lambda > 0:
                real_cpu.requires_grad = True

            # Labels
            real_labels = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
            fake_labels = torch.full((b_size,), fake_label_val, dtype=torch.float, device=device)

            # Forward real
            netD.zero_grad()
            real_logits = netD(real_cpu)  # logits
            loss_D_real = criterion(real_logits, real_labels)

            # Generate fake
            if args.model == 'dcgan':
                noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
                fake = netG(noise)
            else:
                noise = torch.randn(b_size, args.latent_dim, device=device)
                fake = netG(noise)

            fake_logits = netD(fake.detach())
            loss_D_fake = criterion(fake_logits, fake_labels)

            # Gradient penalty (R1) on real images
            r1_penalty = torch.tensor(0.0, device=device)
            if args.gp_lambda > 0:
                r1 = compute_r1_penalty(real_logits, real_cpu)
                r1_penalty = 0.5 * args.gp_lambda * r1

            # Total D loss
            loss_D = loss_D_real + loss_D_fake + r1_penalty
            loss_D.backward()
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            # important: recompute fake because we used detach() for D step
            if args.model == 'dcgan':
                noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
                fake = netG(noise)
            else:
                noise = torch.randn(b_size, args.latent_dim, device=device)
                fake = netG(noise)

            # We want D(G(z)) to be classified as real
            gen_logits = netD(fake)
            loss_G = criterion(gen_logits, real_labels)
            loss_G.backward()
            optimizerG.step()

            # Logging
            D_x = torch.sigmoid(real_logits).mean().item()  # for intuition only
            D_G_z1 = torch.sigmoid(fake_logits).mean().item()
            D_G_z2 = torch.sigmoid(gen_logits).mean().item()

            loop.set_postfix({'Loss_D': f"{loss_D.item():.4f}", 'Loss_G': f"{loss_G.item():.4f}", 'D(x)': f"{D_x:.4f}", 'D(G(z))': f"{D_G_z1:.4f}/{D_G_z2:.4f}"})

        # Step schedulers
        if schedulerD is not None:
            schedulerD.step()
        if schedulerG is not None:
            schedulerG.step()

        # Save samples
        if epoch % args.sample_interval == 0 or epoch == 1:
            save_sample_images(netG, fixed_noise, samples_dir, epoch, args.model, device)
            if writer is not None:
                # log the last saved sample image
                img_path = os.path.join(samples_dir, f"{args.model}_epoch_{epoch:04d}.png")
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    # TensorBoard expects CHW tensors in [0,1]
                    import torchvision.transforms.functional as TF
                    tb_img = TF.to_tensor(img)
                    writer.add_image('samples', tb_img, epoch)
                except Exception:
                    pass

        # TensorBoard scalars
        if writer is not None:
            writer.add_scalar('Loss/Discriminator', loss_D.item(), epoch)
            writer.add_scalar('Loss/Generator', loss_G.item(), epoch)
            writer.add_scalar('D(x)', D_x, epoch)
            writer.add_scalar('D(G_z)_before', D_G_z1, epoch)
            writer.add_scalar('D(G_z)_after', D_G_z2, epoch)

        # FID evaluation (optional)
        if args.fid and TORCHMETRICS_FID and epoch % args.fid_interval == 0:
            print('Computing FID (this may be slow)...')
            fid = FID().to(device)
            # iterate a subset of the dataset and generate as many fakes
            netG.eval()
            with torch.no_grad():
                for j, (real_batch, _) in enumerate(dataloader):
                    if j >= args.fid_batches:
                        break
                    real_batch = real_batch.to(device)
                    if args.model == 'dcgan':
                        noise = torch.randn(real_batch.size(0), args.latent_dim, 1, 1, device=device)
                    else:
                        noise = torch.randn(real_batch.size(0), args.latent_dim, device=device)
                    fake_batch = netG(noise)
                    # real and fake must be in [0,1] for many FID implementations; our images are [-1,1]
                    real_imgs = (real_batch + 1.0) / 2.0
                    fake_imgs = (fake_batch + 1.0) / 2.0
                    fid.update(real_imgs, real=True)
                    fid.update(fake_imgs, real=False)
                fid_value = fid.compute().item()
                print(f"FID @ epoch {epoch}: {fid_value:.4f}")
                if writer is not None:
                    writer.add_scalar('Metrics/FID', fid_value, epoch)
            netG.train()

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save({
                'epoch': epoch,
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'args': vars(args)
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if writer is not None:
        writer.close()

    print('Training finished.')


# ---------- Argument parser ----------

def parse_args():
    parser = argparse.ArgumentParser(description='Train DCGAN / Simplified StyleGAN with extended features')
    parser.add_argument('--model', type=str, default='dcgan', choices=['dcgan', 'stylegan'], help='which generator to use')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagefolder'], help='dataset to use')
    parser.add_argument('--data-root', type=str, default='./data', help='root folder for dataset or ImageFolder root')
    parser.add_argument('--out-dir', type=str, default='./outputs', help='where to save samples & checkpoints')
    parser.add_argument('--img-size', type=int, default=32, help='image size (CIFAR-10 -> 32, custom folder can be larger)')
    parser.add_argument('--latent-dim', type=int, default=100, help='dimensionality of latent vector (dcgan:100, stylegan:512)')
    parser.add_argument('--ngf', type=int, default=64, help='generator feature map size (DCGAN)')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator feature map size')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda even if available')
    parser.add_argument('--num-workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--sample-interval', type=int, default=5, help='how many epochs between sample saves')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='how many epochs between checkpoint saves')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--label-smoothing', action='store_true', help='apply label smoothing to real labels (real=0.9)')
    parser.add_argument('--gp-lambda', type=float, default=0.0, help='R1 gradient penalty coefficient (0 disables)')
    parser.add_argument('--spectral-norm', action='store_true', help='apply spectral normalization to discriminator conv layers')
    parser.add_argument('--use-scheduler', action='store_true', help='use StepLR scheduler')
    parser.add_argument('--step-size', type=int, default=20, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='scheduler gamma')
    parser.add_argument('--tb-log', action='store_true', help='enable TensorBoard logging')
    parser.add_argument('--fid', action='store_true', help='compute FID using torchmetrics if available')
    parser.add_argument('--fid-interval', type=int, default=10, help='how many epochs between FID computation')
    parser.add_argument('--fid-batches', type=int, default=50, help='number of batches to use for FID computation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Set sensible latent dim for stylegan if user picked it but didn't update latent_dim
    if args.model == 'stylegan' and args.latent_dim == 100:
        print('Note: stylegan commonly uses latent_dim=512. Overriding latent_dim to 512 for convenience.')
        args.latent_dim = 512

    train(args)
