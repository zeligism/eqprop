
import time
import os
import argparse
import torch
import matplotlib.pyplot as plt

from dataset import MNISTDataset
from eqprop.eqprop import EqPropNet
from eqprop.eqprop_nograd import EqPropNet_NoGrad


def index_to_onehot(index, num_indices=10):
    onehot = torch.zeros(*index.size(), num_indices)
    onehot.scatter_(-1, index.unsqueeze(dim=-1), 1)
    return onehot

def count_hits(net, true_y):
    pred_y, _ = net.output_state().max(dim=-1, keepdim=True)
    hits = (pred_y * true_y == 1).sum().item()
    return hits

def save_plot(title, y, plots_dir="plots"):
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    plot_path = os.path.join(plots_dir, "{}.png".format(title))

    plt.figure()
    plt.title(title)
    plt.plot(y)
    plt.savefig(plot_path)

def train(trainloader,
          eqpropnet,
          num_epochs=5,
          report_interval=20,
          save_interval=1e6):

    if num_epochs == 0:
        return

    energies = []
    mean_energies = []
    losses = []
    mean_losses = []
    start_time = time.time()
    last_save_time = start_time

    print("Training...")
    for epoch in range(num_epochs):
        for iters, data in enumerate(trainloader):
            x, y = data
            output_size = eqpropnet.output_state().size(-1)
            y = index_to_onehot(y, output_size)

            # Train on (x,y) using equilibrium propagation
            energy, loss = eqpropnet.eqprop(x, y)
            energies.append(energy.mean().item())
            losses.append(loss.mean().item())

            # Prepare stuff for report
            if (iters + 1) % report_interval == 0:
                mean_energies.append(sum(energies)/len(energies))
                mean_losses.append(sum(losses)/len(losses))
                print("[%d, %d] energy = %.3f, loss = %.3f, hits = %d/%d" % (
                    epoch + 1, iters + 1, mean_energies[-1], mean_losses[-1],
                    count_hits(eqpropnet, y), y.size(0)))
                print("Time elapsed = %ds" % (time.time() - start_time))
                energies = []
                losses = []

            # Save model
            if (time.time() - last_save_time) > save_interval:
                eqpropnet.save_parameters("model@epochs={},iters={}.pt".format(epoch, iters+1))
                last_save_time = time.time()
        
    # Save model after finishing training
    eqpropnet.save_parameters("model.pt")
    
    save_plot("Energy", mean_energies)
    save_plot("Loss", mean_losses)
            

def test(testloader, eqpropnet, report_interval):
    print("Testing...")
    total_hits = 0.
    for iters, data in enumerate(testloader):
        x, y = data
        output_size = eqpropnet.output_state().size(-1)
        y = index_to_onehot(y, output_size)

        # Train on (x,y) using equilibrium propagation
        energy, cost = eqpropnet.eqprop(x, y, train=False)
        hits = count_hits(eqpropnet, y)
        total_hits += hits

        if (iters + 1) % report_interval == 0:
            print("[%d] energy = %.3f, cost = %.3f" % (iters, energy.mean(), cost.mean()))
            print("hits =", hits, "out of", y.size(0))

    print()
    print("Average hits =", total_hits / len(testloader))
    error = 1 - total_hits / (len(testloader) * y.size(0))
    print("Error = %.3f%%" % (error * 100))


def main(args):
    # Set random seed if given
    torch.manual_seed(args.random_seed)

    # Define dataset
    trainloader, testloader = MNISTDataset(args.batch_size, args.num_workers)

    # Set model parameters
    model_params = {
        "batch_size": args.batch_size,
        "layer_sizes": args.layer_sizes,
        "learning_rates": args.learning_rates,
        "free_iters": args.free_iters,
        "clamped_iters": args.clamped_iters,
        "beta": args.beta,
        "dt": args.dt,
    }

    # Define network
    eqpropnet = EqPropNet(**model_params) if not args.no_grad else EqPropNet_NoGrad(**model_params)
    if args.load_model:
        eqpropnet.load_parameters(args.load_model)

    # Train
    train(trainloader, eqpropnet,
          num_epochs=args.num_epochs,
          report_interval=args.report_interval,
          save_interval=args.save_interval)

    # Validate
    test(testloader, eqpropnet, report_interval=args.report_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains EqPropNet.")

    parser.add_argument("-r", "--random-seed", type=int, default=1337, help="random seed.")
    parser.add_argument("-n", "--num-epochs", type=int, default=5, help="number of epochs.")
    parser.add_argument("--batch-size", type=int, default=20, help="batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="number of data loading workers.")
    parser.add_argument("--report-interval", type=int, default=20, help="report interval (per iteration).")
    parser.add_argument("--save-interval", type=int, default=300, help="model saving interval (per second).")

    parser.add_argument("--free-iters", type=int, default=20,
        help="number of training iterations in the free phase.")
    parser.add_argument("--clamped-iters", type=int, default=4,
        help="number of training iterations in the clamped phase.")
    parser.add_argument("--beta", type=float, default=0.5,
        help="clamping coefficient (higher = stronger clamping to output).")
    parser.add_argument("--dt", type=float, default=0.5,
        help="time delta (lower = finer, more accurate dynamics but slower convergence).")

    parser.add_argument("--layer-sizes", type=int, nargs="+", default=[784, 500, 10],
        help="sizes for each layer in order from input to output, inclusive.")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.1, 0.05],
        help="learning rates for each layer (must be equal to the number of layers minus 1).")

    parser.add_argument("--no-grad", action="store_true", help="trains without autograd.")
    parser.add_argument("--load-model", type=str, default=None,
        help="path of model to be loaded.")

    args = parser.parse_args()
    main(args)







        