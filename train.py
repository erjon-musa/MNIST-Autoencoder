import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import argparse
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchsummary import summary
from model import autoencoderMLP4Layer
from contextlib import nullcontext


# defaults (can be overridden by CLI)
save_file = 'weights.pth'
n_epochs = 30
batch_size = 256
bottleneck_size = 32
plot_file = 'plot.png'

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

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plot_file=None):
    print('training:')
    model.train()

    # Small performance boost on MPS by allowing higher-precision matmul kernels
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


    losses_train = []
    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0

        for imgs, _ in train_loader:
            # flatten to (B, 784) and move to device as float32
            imgs = imgs.view(imgs.shape[0], -1).to(device=device, dtype=torch.float32)

            if epoch == 1 and loss_train == 0.0:
                print("model param device:", next(model.parameters()).device)
                print("batch device:", imgs.device, "dtype:", imgs.dtype)

            optimizer.zero_grad(set_to_none=True)

            # forward + loss (AMP on MPS)
            with amp_context(device):
                outputs = model(imgs)
                loss = loss_fn(outputs, imgs)

            # backward/step
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train.append(loss_train / len(train_loader))

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, losses_train[-1]))

        if save_file is not None:
            torch.save(model.state_dict(), save_file)

        if plot_file is not None:
            plt.figure(num=2, figsize=(12, 7), clear=True)
            plt.plot(losses_train, label='train')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def amp_context(device):
    # enable autocast on CUDA/CPU only; no-op on MPS
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    elif device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        # MPS or anything else → no autocast
        return nullcontext()

def main():
    global bottleneck_size, save_file, n_epochs, batch_size, plot_file

    print('running main ...')

    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    parser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    parser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    parser.add_argument('-b', metavar='batch size', type=int, help='batch size [256]')
    parser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    # optional override: --device auto|cpu|mps
    parser.add_argument('--device', choices=['auto','cpu','mps'], default='auto', help='device override [auto]')
    args = parser.parse_args()

    if args.s is not None:
        save_file = args.s
    if args.z is not None:
        bottleneck_size = args.z
    if args.e is not None:
        n_epochs = args.e
    if args.b is not None:
        batch_size = args.b
    if args.p is not None:
        plot_file = args.p

    print('bottleneck size = ', bottleneck_size)
    print('n epochs = ', n_epochs)
    print('batch size = ', batch_size)
    print('save file = ', save_file)
    print('plot file = ', plot_file)

    # device
    device, device_name = select_device()
    if args.device == 'cpu':
        device, device_name = torch.device('cpu'), 'cpu (forced)'
    elif args.device == 'mps':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device, device_name = torch.device('mps'), 'mps (forced)'
        else:
            print("WARNING: --device mps requested but not available; using CPU.")
            device, device_name = torch.device('cpu'), 'cpu (fallback)'
    print('using device ', device_name)

    # model
    N_input = 28 * 28
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.apply(init_weights)  # init once on CPU

    # Optional: summary on CPU; skip if unsupported
    try:
        summary(model, model.input_shape)
    except Exception as e:
        print("torchsummary skipped on this backend:", e)

    model.to(device)  # move once to target device

    # data
    transform = transforms.ToTensor()
    train_set = MNIST('./data/mnist', train=True, download=True, transform=transform)

    # persistent_workers only if num_workers > 0
    num_workers = 2
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    # optim/loss/sched
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.MSELoss()

    # train
    train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
        save_file=save_file,
        plot_file=plot_file,
    )

###################################################################

if __name__ == '__main__':
    main()
