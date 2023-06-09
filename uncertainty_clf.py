import numpy as np
import torch
from scipy.stats import norm

from net_utils import load_from_file

class Certainty():
    def __init__(self, net_file):
        self.net = load_from_file(net_file)
    
    def train(self, batch):
        a2s = [[], []]
        self.priors = [0, 0]
        with torch.no_grad():
            db = self.net.evaluate_debug(batch)
            for i in range(2): 
                a2s[i].extend(
                    db['a2'][db['data'][:][1] == i].flatten().tolist())
                self.priors[i] += sum(db['data'][:][1] == i)

        self.parameters = [
            [np.mean(a2s[0]), np.std(a2s[0])], # [mu_0, sig_0]
            [np.mean(a2s[1]), np.std(a2s[1])], # [mu_1, sig_1]
        ]

    def evaluate(self, batch):
        with torch.no_grad():
            db = self.net.evaluate_debug(batch)
        wpdf = lambda a2, i: self.priors[i] * norm.pdf(
            a2, loc=self.parameters[i][0], scale=self.parameters[i][1]
        )
        c = lambda a2: wpdf(a2, int(a2 > 0)) / (wpdf(a2, 0) + wpdf(a2, 1))
        return (
            np.array([c(a2).numpy()[0] for a2 in db['a2']]), 
            db['a2'].numpy().flatten(), 
            db['out'].numpy().flatten()
        )

