
import time
from matplotlib.pyplot import imshow
import torch

from training_configs import *
from dataset import dataset
from eqprop_torch import EqPropNet
from eqprop_torch_nograd import EqPropNet_NoGrad


def index_to_onehot(index, num_indices):
    onehot = torch.zeros(*index.size(), num_indices)
    onehot.scatter_(-1, index.unsqueeze(dim=-1), 1)

    return onehot


def count_hits(net, y):
    _, index = net.output_state().max(dim=-1)
    hits = (y == index).sum().item()

    return hits


def train(net, trainloader):
    print("Training...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        running_energy = running_cost = 0.0
        for i, data in enumerate(trainloader):
            x, y = data

            # Prepare data
            x = x.view(-1, LAYER_SIZES[0])
            y_onehot = index_to_onehot(y, LAYER_SIZES[-1])

            # Train on (x,y) using equilibrium propagation
            energy, cost = net.eqprop(x, y_onehot)

            # Perpare stuff for checkpoint reports
            running_energy += energy.mean()
            running_cost += cost.mean()
            if (i + 1) % CHECKPOINT == 0:
                avg_energy = running_energy / CHECKPOINT
                avg_cost = running_cost / CHECKPOINT
                print("[%d, %d] energy = %.3f, cost = %.3f, hits = %d/%d" % (
                    epoch+1, i+1, avg_energy, avg_cost,
                    count_hits(net, y), y.size()[0]))
                print("Time elapsed = %ds" % (time.time() - start_time))
                running_energy = running_cost = 0.0

        print("\nEPOCH: [%d/%d] energy = %.3f, cost = %.3f\n, hits = %d/%d" % (
            epoch+1, EPOCHS, energy.mean(), cost.mean(), count_hits(net, y), y.size()[0]))

    # Save model
    net.save_parameters(FNAME)
            

def test(net, testloader):
    print("Testing...")
    running_hits = iterations = 0
    for i, data in enumerate(testloader):
        x, y = data

        # Prepare data
        x = x.view(-1, LAYER_SIZES[0])
        y_onehot = index_to_onehot(y, LAYER_SIZES[-1])

        # Train on (x,y) using equilibrium propagation
        energy, cost = net.eqprop(x, y_onehot, train=False)
        print("[%d] energy = %.3f, cost = %.3f" % (i, energy.mean(), cost.mean()))

        # Calculate hits
        hits = count_hits(net, y)
        print("hits =", hits, "out of", y.size()[0])

        running_hits += hits
        iterations += 1

    print()
    print("Average hits =", running_hits / iterations)
    error = 1 - running_hits / (iterations * y.size()[0])
    print("Error = %.3f%%" % (error * 100))


def main():
    # Set random seed if given
    torch.manual_seed(RANDOM_SEED or torch.initial_seed())

    # Define dataset
    trainloader, testloader = dataset(BATCH_SIZE)

    # Set model parameters
    model_params = {
        "batch_size": BATCH_SIZE,
        "layers_sizes": LAYER_SIZES,
        "learning_rates": LEARNING_RATES,
        "n_iter_1": N_ITER_1,
        "n_iter_2": N_ITER_2,
        "rho": lambda x: x.clamp(0,1),  # Assuming x is a torch.Tensor
        "beta": BETA,
        "dt": DELTA,
    }

    # Define network
    eqprop_net = EqPropNet_NoGrad(**model_params)

    # Train
    train(eqprop_net, trainloader)

    # Validate
    test(eqprop_net, testloader)


if __name__ == '__main__':
    main()







        