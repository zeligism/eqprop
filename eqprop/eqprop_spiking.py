
import torch

from .eqprop import EqPropNet
from .eqprop_nograd import EqPropNet_NoGrad


class StochasticEncoder:
    def __init__(self, **kwargs):
        pass

    def __call__(self, states):
        return [torch.bernoulli(s) for s in states]


class IdentityDecoder:
    def __call__(self, spikes):
        return spikes


class RobinsonMunroeAnnealer:
    def __init__(self, c0, eta):
        self.c0 = c0
        self.eta = eta
        self.iters = 1

    def step(self):
        """
        Calculate current value, then increment t (i.e. iters),
        so it should be called once each eqprop step.
        You can get the value without stepping by calling value().
        """
        value = self.value(self.iters)
        self.iters += 1
        return value

    def value(self, t):
        """Return current value."""
        return self.c0 / (t ** self.eta)

    def restart(self):
        """Restarts iter counter and return inital value without incrementing."""
        self.iters = 1
        return self.value(self.iters)


class SigmaDeltaEncoder:
    def __init__(self, states):
        self.potentials = [torch.zeros_like(s) for s in states]

    def __call__(self, states):
        quantized_spikes = [None] * len(states)
        for i in range(len(states)):
            quantized_spikes[i] = (self.potentials[i] + states[i] > 0.5).float()
            self.potentials[i] = self.potentials[i] + states[i] - quantized_spikes[i]
            self.potentials[i].detach_()

        return quantized_spikes


class PredictiveEncoder:
    def __init__(self, states, quantizer):
        self.quantizer = quantizer
        self.prev_states = [torch.zeros_like(s) for s in states]

    def __call__(self, states, lmbda):
        encoded_states = [None] * len(states)
        for i in range(len(states)):
            encoded_states[i] = (states[i] - (1 - lmbda) * self.prev_states[i]) / lmbda
            self.prev_states[i] = states[i].detach()
        return self.quantizer(encoded_states)


class PredictiveDecoder:
    def __init__(self, states):
        self.prev_decoded_spikes = [torch.zeros_like(s) for s in states]

    def __call__(self, weighted_spikes, lmbda):
        decoded_spikes = [None] * len(weighted_spikes)
        for i in range(len(weighted_spikes)):
            decoded_spikes[i] = (1 - lmbda) * self.prev_decoded_spikes[i] + lmbda * weighted_spikes[i]
            self.prev_decoded_spikes[i] = decoded_spikes[i].detach()
        return decoded_spikes


class EqPropSpikingNet(EqPropNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = PredictiveEncoder(self.states, quantizer=SigmaDeltaEncoder(self.states))
        self.fore_decoder = PredictiveDecoder(self.states[1:])
        self.back_decoder = PredictiveDecoder(self.states[:-1])
        self.dt_annealer = RobinsonMunroeAnnealer(c0=0.84, eta=0.092)
        self.lmbda_annealer = RobinsonMunroeAnnealer(c0=0.83, eta=0.58)

        # Initialize dt and lambda
        self.dt = self.dt_annealer.restart()
        self.lmbda = self.lmbda_annealer.restart()

        self.prev_encoded_states = [torch.zeros_like(s) for s in self.states]

    def eqprop(self, *args, **kwargs):
        self.dt = self.dt_annealer.restart()
        self.lmbda = self.lmbda_annealer.restart()
        return super().eqprop(*args, **kwargs)

    def energy(self, states):
        """
        Calculates the energy of the network.
        (energy is called once each step())
        """

        # Return current annealed step size and lambda
        self.dt = self.dt_annealer.step()
        self.lmbda = self.lmbda_annealer.step()

        # Encode/quantize states
        encoded_states = self.encoder([self.rho(s) for s in states], self.lmbda)

        # Decode foreward and backward spikes or synapses or sorry-I-don't-know-much-neurosciences
        fore_spikes = self.fore_decoder(
            [s_i @ W_ij for s_i, W_ij in zip(encoded_states, self.weights)], self.lmbda)
        back_spikes = self.back_decoder(
            [s_j @ W_ij.t() for s_j, W_ij in zip(encoded_states[1:], self.weights)], self.lmbda)

        # Norm energy
        states_energy = sum(torch.sum(s * s, dim=1) for s in states)

        # Weights/spikes energy
        spikes_energy = sum(
            torch.sum(self.rho(states[i+1]) * (fore_spikes[i] + self.biases[i+1]), dim=1)
            for i in range(len(fore_spikes))
        )
        spikes_energy += sum(
            torch.sum(self.rho(states[i]) * (back_spikes[i]), dim=1)
            for i in range(len(back_spikes))
        )

        return 0.5 * states_energy - spikes_energy


class EqPropSpikingNet_NoGrad(EqPropSpikingNet, EqPropNet_NoGrad):
    """
    energy() from EqPropSpikingNet. XXX: not correct.
    """

    def eqprop(self, *args, **kwargs):
        with torch.no_grad():
            return super().eqprop(*args, **kwargs)

    def step(self, states, y=None):

        # Return current annealed step size and lambda
        self.dt = self.dt_annealer.step()
        self.lmbda = self.lmbda_annealer.step()

        # Encode/quantize states
        encoded_states = self.encoder([self.rho(s) for s in states], self.lmbda)

        # Decode foreward and backward spikes or synapses or sorry-I-don't-know-much-neurosciences
        fore_spikes = self.fore_decoder(
            [s_i @ W_ij for s_i, W_ij in zip(encoded_states, self.weights)], self.lmbda)
        back_spikes = self.back_decoder(
            [s_j @ W_ij.t() for s_j, W_ij in zip(encoded_states[1:], self.weights)], self.lmbda)

        # Update states
        for i in range(1, len(states)):
            # Calculate the gradient ds/dt = -dE/ds
            fore_spike = fore_spikes[i-1]
            back_spike = 0 if i == len(states)-1 else back_spikes[i]
            states_grad = -states[i] + self.rho_grad(states[i]) * (
                fore_spike + back_spike + self.biases[i])
            # Update and clamp
            states[i] += self.dt * states_grad
            states[i].clamp_(0,1)

        # Update last layer in weakly clamped phase
        if y is not None:
            states[-1] += self.dt * self.beta * (y - states[-1])

        return states

