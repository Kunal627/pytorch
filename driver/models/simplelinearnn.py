import torch.nn as nn

import torch.nn as nn
from collections import OrderedDict

class SimpleLinearNN:

    @staticmethod
    def build(param, nodes, classes):
        model = nn.Sequential(OrderedDict([
            ('hidden_linear', nn.Linear(param, nodes)),
            ('hidden_activation', nn.Tanh()),
            ('output_linear', nn.Linear(nodes, classes))
            ]))

        return model
