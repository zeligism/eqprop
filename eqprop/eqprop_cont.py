
import torch
from math import sqrt
from .eqprop import EqPropNet, EqPropNet_NoGrad


class ContEqPropNet(EqPropNet):
    """
    This class implements C-EqProp, aka EqProp with continual weight updates.
    arxiv link: https://arxiv.org/abs/2005.04168.pdf.
    """

    def eqprop(self, x, y, train=True):
        """
        Trains the network on one example (x,y) using equilibrium propagation
        with continual weight updates.
        """

        # First clamp the input
        self.clamp_input(x)

        # Run free phase
        for _ in range(self.free_iters):
            self.step(self.states)

        if train:
            # Collect states and perturb them to the weakly clamped y
            states = [s.clone().detach() for s in self.states]
            for _ in range(self.clamped_iters):
                # Copy current states at time t
                prev_states = [s.clone() for s in states]
                # Do one eqprop step on states
                self.step(states, y)
                # Update weights based on a perturbation from states(t) to states(t+1)
                # so states(t+1) is clamped and states(t) is free, in some sense.
                self.update_weights(prev_states, states)

        with torch.no_grad():
            return self.energy(self.states), self.cost(self.states, y)


class ContEqPropNet_NoGrad(ContEqPropNet, EqPropNet_NoGrad):
    """
    EqProp with continual weight updates without using autograd.
    """
    def eqprop(self, *args, **kwargs):
        with torch.no_grad():
            return super().eqprop(*args, **kwargs)