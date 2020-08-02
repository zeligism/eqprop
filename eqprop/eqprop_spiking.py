
import torch

from .eqprop import EqPropNet
from .eqprop_nograd import EqPropNet_NoGrad


class StochasticEncoder:
    def __init__(self, **kwargs):
        pass

    def __call__(self, states):
        return [torch.bernoulli(state) for state in states]


class IdentityDecoder:
    def __call__(self, spikes):
        return spikes


class RobinsonMunroeAnnealer:
    def __init__(self, c0, eta):
        self.c0 = c0
        self.eta = eta
        self.iters = 1

    def __call__(self):
        dt = self.c0 / (self.iters ** self.eta)
        self.iters += 1
        return dt

    def restart(self):
        self.iters = 1


class SigmaDeltaEncoder:
    def __init__(self, states):
        self.potentials = [torch.zeros_like(state) for state in states]

    def __call__(self, states):
        spiked_potentials = [None] * len(states)
        quantized_spikes = [None] * len(states)
        for i in range(len(states)):
            spiked_potentials[i] = self.potentials[i] + states[i]
            quantized_spikes[i] = (spiked_potentials[i] > 0.5).float()
            self.potentials[i] = spiked_potentials[i] - quantized_spikes[i]
            self.potentials[i].detach_()

        return quantized_spikes


class PredictiveEncoder:
    def __init__(self, states, quantizer):
        self.quantizer = quantizer
        self.prev_states = [torch.zeros_like(state) for state in states]

    def __call__(self, states, lmbda=0.5):
        encoded_states = [None] * len(states)
        for i in range(len(states)):
            encoded_states[i] = (states[i] - (1 - lmbda) * self.prev_states[i]) / lmbda
            self.prev_states[i] = states[i].detach()
        return self.quantizer(encoded_states)


class PredictiveDecoder:
    def __init__(self, states):
        self.prev_decoded_spikes = [torch.zeros_like(state.detach()) for state in states[1:]]

    def __call__(self, weighted_spikes, lmbda=0.5):
        decoded_spikes = [None] * len(weighted_spikes)
        for i in range(len(weighted_spikes)):
            decoded_spikes[i] = lmbda * weighted_spikes[i] + (1 - lmbda) * self.prev_decoded_spikes[i]
            self.prev_decoded_spikes[i] = decoded_spikes[i].detach()
        return decoded_spikes


class EqPropSpikingNet(EqPropNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = PredictiveEncoder(self.states, quantizer=SigmaDeltaEncoder(self.states))
        self.decoder = PredictiveDecoder(self.states)
        self.dt_annealer = RobinsonMunroeAnnealer(c0=0.84, eta=0.092)
        self.lmbda_annealer = RobinsonMunroeAnnealer(c0=0.83, eta=0.58)
        self.lmbda = 0.5

    def eqprop(self, *args, **kwargs):
        self.dt_annealer.restart()
        self.lmbda_annealer.restart()
        return super().eqprop(*args, **kwargs)

    def energy(self, states):
        """
        Calculates the energy of the network.
        """

        encoded_states = self.encoder([self.rho(s) for s in states], self.lmbda)
        weighted_spikes = [encoded_states[i] @ self.weights[i] for i in range(len(self.weights))]
        decoded_spikes = self.decoder(weighted_spikes, self.lmbda)

        # Anneal step size and lambda
        self.dt = self.dt_annealer()
        self.lmbda = self.lmbda_annealer()

        # Sum of s_i * s_i for all i
        states_energy = sum(
            torch.sum(states[i] * states[i], dim=-1)
            for i in range(len(states))
        )

        # Sum of W_ij * rho(s_i) * rho(s_j) for all i, j
        spikes_energy = sum(
            torch.sum(self.rho(states[i+1]) * (decoded_spikes[i] + self.biases[i]), dim=-1)
            for i in range(len(self.weights))
        )

        return 0.5 * states_energy + spikes_energy


class EqPropSpikingNet_NoGrad(EqPropSpikingNet):
    """
    energy() from EqPropSpikingNet. XXX: not correct.
    """

    def eqprop(self, *args, **kwargs):
        with torch.no_grad():
            return super().eqprop(*args, **kwargs)

    def step(self, states, y=None):

        encoded_states = self.encoder([self.rho(s) for s in states], self.lmbda)
        weighted_spikes = [encoded_states[i] @ self.weights[i] for i in range(len(self.weights))]
        decoded_spikes = self.decoder(weighted_spikes, self.lmbda)

        ## Anneal step size and lambda
        self.dt = self.dt_annealer()
        self.lmbda = self.lmbda_annealer()

        # Update states
        for i in range(1, len(states)):
            # Calculate the gradient ds/dt = -dE/ds
            states_grad = -states[i] + self.rho_grad(states[i]) * (
                decoded_spikes[i-1] + self.biases[i-1])
            # Update and clamp
            states[i] += self.dt * states_grad
            states[i].clamp_(0,1)

        # Update last layer in weakly clamped phase
        if y is not None:
            states[-1] += self.dt * self.beta * (y - states[-1])

        return states

    def update_weights(self, free_states, clamped_states):
        """
        TODO
        """
        rho = self.rho

        for i in range(len(self.weights)):
            # Calculate weight gradient and update TODO: is there a faster way?
            weights_grad = \
                rho(clamped_states[i]).unsqueeze(dim=-1) \
                @ rho(clamped_states[i+1]).unsqueeze(dim=-2) \
                - rho(free_states[i]).unsqueeze(dim=-1) \
                @ rho(free_states[i+1]).unsqueeze(dim=-2)
            self.weights[i] += self.lr[i] / self.beta * weights_grad.mean(dim=0)

        for i in range(len(self.biases)):
            # Calculate weight gradient and update
            biases_grad = rho(clamped_states[i+1]) - rho(free_states[i+1])
            self.biases[i] += self.lr[i] / self.beta * biases_grad.mean(dim=0)

