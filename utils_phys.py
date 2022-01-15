
import torch
import numpy as np
if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)
print(f"TORCH DEVICE: {torch_device}")


class ScalarPhi4Action:
    def __init__(self, kappa, lam):
        self.kappa = kappa
        self.lam = lam
    def __call__(self, cfgs):
        # kinetic term (discrete Laplacian)
        dims = [i+1 for i in range(len(cfgs.shape)-1)]
        action = (1 - 2 * self.lam) * cfgs**2 + self.lam *cfgs**4
        action += -2. * self.kappa * cfgs * torch.roll(cfgs, 1, 1)
        action += -2. * self.kappa * cfgs * torch.roll(cfgs, 1, 2)

        return torch.sum(action, dims)


def grab(var):
    return var.detach().cpu().numpy()
# Observables
def jackknife(samples: np.ndarray):
    """Return mean and estimated lower error bound."""
    means = []

    for i in range(samples.shape[0]):
        means.append(np.delete(samples, i, axis=0).mean(axis=0))

    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt((samples.shape[0] - 1) * np.mean(np.square(means - mean), axis=0))
    
    return mean, error



def get_mag(cfgs: torch.Tensor):
    dims = tuple([i+1 for i in range(len(np.shape(cfgs))-1)])

    return np.mean(grab(cfgs), axis = dims)



def get_chi2(cfgs: np.ndarray):
    """Return mean and error of suceptibility."""
    V = np.prod(cfgs.shape[1:])
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
    mags = cfgs.mean(axis=axis)
    return jackknife(V * (mags**2 - mags.mean()**2))