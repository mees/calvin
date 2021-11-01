import numpy as np
from torch import nn


class IdentityEncoder(nn.Module):
    def __init__(self, proprioception_dims):
        super(IdentityEncoder, self).__init__()
        # remove a dimension if we convert robot orientation quaternion to euler angles
        self.n_state_obs = int(np.sum(np.diff([list(x) for x in [list(y) for y in proprioception_dims.keep_indices]])))
        self.identity = nn.Identity()

    @property
    def out_features(self):
        return self.n_state_obs

    def forward(self, x):
        return self.identity(x)
