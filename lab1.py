# Lab 1 Orchestrator

import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# import model class
from model import autoencoderMLP4Layer



# Device selection (MPS for macOS first)
def select_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            # sanity check that we can actually allocate on MPS
            _ = torch.ones(1, device="mps")
            return torch.device("mps"), "mps (Apple Silicon)"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"



# Utility
def to_float01(img_u8: torch.Tensor) -> torch.Tensor:
    # (28,28) uint8 [0..255] -> float32 [0..1]
    return img_u8.to(torch.float32) / 255.0



# Plot helpers
def plot_clean_reconstruction(model, device, dataset, idx, out_path):
    img_u8 = dataset.data[idx]
    img = to_float01(img_u8).view(1, -1).to(device)

    model.eval()
    with torch.no_grad():
        y = model(img).view(28, 28).detach().cpu()

    plt.figure(figsize=(4.5, 3))
    ax = plt.subplot(1, 2, 1)
    ax.imshow(img_u8, cmap='gray'); ax.set_title("Original"); ax.axis('off')
    ax = plt.subplot(1, 2, 2)
    ax.imshow(y, cmap='gray', vmin=0, vmax=1); ax.set_title("Reconstruction"); ax.axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    plt.show()


def plot_denoise_reconstruction(model, device, dataset, idx, noise_std, out_path):
    img_u8 = dataset.data[idx]
    clean = to_float01(img_u8)                 # (28,28) [0,1]
    noisy = torch.clamp(clean + torch.randn_like(clean) * noise_std, 0.0, 1.0)
    x = noisy.view(1, -1).to(device)

    model.eval()
    with torch.no_grad():
        y = model(x).view(28, 28).detach().cpu()

    plt.figure(figsize=(6.5, 3))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(img_u8, cmap='gray'); ax.set_title("Original"); ax.axis('off')
    ax = plt.subplot(1, 3, 2)
    ax.imshow(noisy, cmap='gray', vmin=0, vmax=1); ax.set_title(f"Noisy (σ={noise_std})"); ax.axis('off')
    ax = plt.subplot(1, 3, 3)
    ax.imshow(y, cmap='gray', vmin=0, vmax=1); ax.set_title("Reconstruction"); ax.axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    plt.show()


def plot_interpolation_row(model, device, dataset, idx1, idx2, steps, out_path):
    encode, decode = model.encode, model.decode

    img1_u8 = dataset.data[idx1]
    img2_u8 = dataset.data[idx2]
    x1 = to_float01(img1_u8).view(1, -1).to(device)
    x2 = to_float01(img2_u8).view(1, -1).to(device)

    model.eval()
    with torch.no_grad():
        # latent codes
        z1 = encode(x1)
        z2 = encode(x2)

        ts = torch.linspace(0.0, 1.0, steps=steps, device=device)
        mids = []
        for t in ts:
            zt = torch.lerp(z1, z2, t)
            yt = decode(zt).view(28, 28).detach().cpu()
            mids.append(yt)

    total = steps + 2
    plt.figure(figsize=(1.6 * total, 3))

    # left anchor (original A)
    ax = plt.subplot(1, total, 1)
    ax.imshow(img1_u8, cmap='gray'); ax.set_title(f"A\nidx={idx1}"); ax.axis('off')

    # middle: reconstructions at t∈[0..1]
    for i, y in enumerate(mids):
        ax = plt.subplot(1, total, i + 2)
        ax.imshow(y, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"t={ts[i]:.2f}")
        ax.axis('off')

    # right anchor (original B)
    ax = plt.subplot(1, total, total)
    ax.imshow(img2_u8, cmap='gray'); ax.set_title(f"B\nidx={idx2}"); ax.axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    plt.show()



# Main

def main():
    parser = argparse.ArgumentParser(description="ELEC 475 Lab 1 runner (clean, denoise, interpolate)")
    parser.add_argument('-l', '--load', required=True, type=str, help='path to trained weights .pth')
    parser.add_argument('-z', '--bottleneck', type=int, default=8, help='bottleneck size used in training [8]')
    parser.add_argument('--clean_idx', type=int, default=0, help='index for clean reconstruction [0]')
    parser.add_argument('--denoise_idx', type=int, default=0, help='index for denoise reconstruction [0]')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Gaussian noise std for denoising [0.2]')
    parser.add_argument('--interp_idx1', type=int, default=12, help='first index for interpolation [12]')
    parser.add_argument('--interp_idx2', type=int, default=3456, help='second index for interpolation [3456]')
    parser.add_argument('--steps', type=int, default=8, help='# of interpolation steps [8]')
    parser.add_argument('--set', choices=['train', 'test'], default='test', help='dataset split [test]')
    args = parser.parse_args()

    device, device_name = select_device()
    print(f'Using device: {device_name}')

    # dataset
    transform = transforms.ToTensor()
    dataset = MNIST('./data/mnist', train=(args.set == 'train'), download=True, transform=transform)

    # model
    N = 28 * 28
    model = autoencoderMLP4Layer(N_input=N, N_bottleneck=args.bottleneck, N_output=N)
    state = torch.load(args.load, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)

    # 1) clean recon
    plot_clean_reconstruction(
        model, device, dataset,
        idx=args.clean_idx,
        out_path='recon.clean.png'
    )

    # 2) denoise recon
    plot_denoise_reconstruction(
        model, device, dataset,
        idx=args.denoise_idx,
        noise_std=args.noise_std,
        out_path='recon.denoise.png'
    )

    # 3) latent interpolation
    plot_interpolation_row(
        model, device, dataset,
        idx1=args.interp_idx1, idx2=args.interp_idx2,
        steps=args.steps,
        out_path='interp.png'
    )


if __name__ == '__main__':
    main()
