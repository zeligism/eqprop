
import time
from matplotlib.pyplot import imshow
import torch

from training_configs import *
from dataset import dataset
from eqprop.eqprop_graph import EqPropGraph

    
def index_to_onehot(index, num_indices):
    onehot = torch.zeros(*index.size(), num_indices)
    onehot.scatter_(-1, index.unsqueeze(dim=-1), 1)

    return onehot


def count_hits(graph, y):
    _, pred_index = max((graph.states[node], i)
        for i, node in enumerate(graph.output_nodes))

    return int(y == pred_index)


def train(graph, trainloader):
    print("Training...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        for i, data in enumerate(trainloader):
            x, y = data

            # Prepare data
            x = x.view(-1, LAYER_SIZES[0])
            y_onehot = index_to_onehot(y, LAYER_SIZES[-1])

            # Train on (x,y) using equilibrium propagation
            x_list = [x[0, i] for i in range(x.size()[-1])]
            y_list = [y_onehot[0, i] for i in range(y_onehot.size()[-1])]
            energy, cost = graph.eqprop(x_list, y_list)

            # Report
            print("[%d, %d] energy = %.3f, cost = %.3f, hit = %d" % (
                epoch+1, i+1, energy, cost, count_hits(graph, y)))
            print("Time elapsed = %ds" % (time.time() - start_time))

        print("\nEPOCH: [%d/%d] energy = %.3f, cost = %.3f\n, hit = %d" % (
            epoch+1, EPOCHS, energy, cost, count_hits(graph, y)))

            

def test(graph, testloader):
    print("Testing...")
    running_hits = iterations = 0
    for i, data in enumerate(testloader):
        x, y = data

        # Prepare data
        x = x.view(-1, LAYER_SIZES[0])
        y_onehot = index_to_onehot(y, LAYER_SIZES[-1])

        # Train on (x,y) using equilibrium propagation
        x_list = [x[0, i] for i in range(x.size()[-1])]
        y_list = [y_onehot[0, i] for i in range(y_onehot.size()[-1])]
        energy, cost = graph.eqprop(x_list, y_list, train=False)
        print("[%d] energy = %.3f, cost = %.3f" % (i, energy, cost))

        # Calculate hits
        hit = count_hits(graph, y)
        print("hit =", hit)

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
    trainloader, testloader = dataset(1)

    # Define network
    eqprop_graph = EqPropGraph(*create_graph_example())

    # Train
    train(eqprop_graph, trainloader)

    # Validate
    test(eqprop_graph, testloader)


def create_graph_example():
    name_format = "{}[{}]"
    l_sizes = [28*28, 500, 10]
    edges = []
    input_nodes = set()
    output_nodes = set()
    biases = set()
    layer = 0
    for l_size1, l_size2 in zip(l_sizes[:-1], l_sizes[1:]):
        for i in range(l_size1 + 1):  # plus bias
            node1 = name_format.format(layer, i)
            for j in range(l_size2):
                node2 = name_format.format(layer+1, j)
                edges.append((node1, node2))
        layer += 1

    # Save input nodes
    for i in range(l_sizes[0]):
        node = name_format.format(0, i)
        input_nodes.add(node)
    # Save output nodes
    for i in range(l_sizes[-1]):
        node = name_format.format(2, i)
        output_nodes.add(node)
    # Save biases
    for layer, l_size in enumerate(l_sizes):
        node = name_format.format(layer, l_size)
        biases.add(node)

    return edges, input_nodes, output_nodes, biases


if __name__ == '__main__':
    main()







        