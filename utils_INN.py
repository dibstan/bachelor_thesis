import os
_ = os.system("conda env list")
import base64
import json
import io
import pickle
import torch
import numpy as np
print(f'TORCH VERSION: {torch.__version__}')
import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
    raise RuntimeError('Torch versions lower than 1.5.0 not supported')

import torch.nn.functional as functions


if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)
print(f"TORCH DEVICE: {torch_device}")


def print_metrics(history, avg_last_N_epochs):
    print(f'== Era {era} | Epoch {epoch} metrics ==')
    for key, val in history.items():
        avgd = np.mean(val[-avg_last_N_epochs:])
        print(f'\t{key} {avgd:g}')



def grab(var):
    return var.detach().cpu().numpy()



class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)



def apply_flow_to_prior(prior, coupling_layers, conditioning, *, batch_size):
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x, conditioning)
        #print('inflow',x)
        logq = logq - logJ
    return x, logq



def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    checker = checker.to(torch_device)

    checker = torch.cat((checker.unsqueeze(0), checker.unsqueeze(0)))
    return checker



class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity):
        super().__init__()
        self.mask = make_checker_mask(mask_shape, mask_parity)
        self.net = net

    def forward(self, x, conditioning):
        conditioning = conditioning.unsqueeze(1)
        if len(x.size()) < len(conditioning.size()):
            x = x.unsqueeze(1)

        if x.size()[1] == conditioning.size()[1]:
            x = torch.cat((x, conditioning), dim = 1)

        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x
        net_out = self.net(x_frozen)
        s, t = net_out[:,0], net_out[:,1]
        
        fx = (1 - self.mask[0]) * t + x_active[:,0,:,:] * torch.exp(s) + x_frozen[:,0,:,:]
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask[0]) * s, dim=tuple(axes))
        return fx, logJ



def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    assert packaging.version.parse(torch.__version__) >= packaging.version.parse('1.5.0')
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'
    padding_size = ((kernel_size // 2))
    net = []
    for i in range(len(sizes) - 1):#
        net.append(torch.nn.Conv2d(
        sizes[i], sizes[i+1], kernel_size, padding=padding_size,
        stride=1, padding_mode='circular'))
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)



def make_phi4_affine_layers(*, n_layers, lattice_shape, hidden_sizes, kernel_size):
    layers = []
    for i in range(n_layers):
        parity = i % 2
        net = make_conv_net(
            in_channels=2, out_channels=2, hidden_sizes=hidden_sizes,
            kernel_size=kernel_size, use_final_tanh=True)
        coupling = AffineCoupling(net, mask_shape=lattice_shape, mask_parity=parity)
        layers.append(coupling)
    return torch.nn.ModuleList(layers)



def calc_dkl(logp, logq):
    return (logq - logp).mean()  # reverse KL, assuming samples from q



def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2*torch.logsumexp(logw, dim=0) - torch.logsumexp(2*logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg



def train_step(model, action, loss_fn, optimizer, metrics, batch_size, conditioning):
    layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()
    
    x, logq = apply_flow_to_prior(prior, layers, conditioning, batch_size=batch_size)
    logp = -action(x)
    loss = loss_fn(logp, logq)
    loss.backward()
    optimizer.step()

    metrics['loss'].append(grab(loss))
    metrics['logp'].append(grab(logp))
    metrics['logq'].append(grab(logq))
    metrics['ess'].append(grab(compute_ess(logp, logq) ))