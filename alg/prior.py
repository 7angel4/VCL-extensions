from constants import *
from torch.distributions import Exponential, Normal

class Prior:
    def __init__(self):
        pass

    def kl(self):
        """KL(q,p), where q is Gaussian and p is from the given prior family"""
        pass

    def init_params_for(self, model, init_mu=0.0, init_scale=0.1, init_const=-3.0):
        pass


class GaussianPrior(Prior):
    def __init__(self):
        super().__init__()

    def kl(self, q, p):
        q_mu, q_sigma = q['mu'], q['scale']
        p_mu, p_sigma = p['mu'], p['scale']
        
        ratio = (q_sigma / p_sigma) ** 2
        log_std = torch.log(1./ratio)
        mean_term = ratio + ((q_mu - p_mu) / p_sigma) ** 2
        return 0.5 * (log_std + mean_term - 1)

    # The following matches the VCL authors' code
    def init_params_for(self, model, init_mu=0.0, init_scale=0.1, init_const=-3.0):
        nn.init.trunc_normal_(model.W_mu, mean=init_mu, std=init_scale)
        nn.init.trunc_normal_(model.b_mu, mean=init_mu, std=init_scale)
        model.W_rho.data.fill_(init_const)  # to match the VCL authors' initialisation in code
        model.b_rho.data.fill_(init_const)


class ExponentialPrior(Prior):
    def __init__(self):
        super().__init__()

    def kl(self, q, p):  # capped KL - see Section 4 of our paper
        q_mu, q_sigma = q['mu'], q['scale']
        p_mu, p_sigma = p['mu'], p['scale']
        normcdf = Normal(loc=0., scale=1.).cdf
        E_abs = torch.mul(p_sigma, np.sqrt(2./np.pi)) * torch.exp(-p_mu**2 / (2*p_sigma**2)) + \
                p_mu * (1. - 2 * normcdf(-p_mu/p_sigma))
        lambda_hat = 1. / E_abs  # estimate for lambda
        logp = torch.log(lambda_hat) \
                - torch.div(lambda_hat * q_sigma, np.sqrt(2*np.pi)) * torch.exp(-q_mu**2 / (2*q_sigma**2)) \
                - lambda_hat * q_mu * (1.- normcdf(-q_mu/q_sigma))
        logq = -0.5 * (1. + torch.log(2 * np.pi * q_sigma**2))  # Gaussian entropy
        return logq - logp

    # initial scale set to 0.1 to match the VCL authors' code
    def init_params_for(self, model, init_mu=0.0, init_scale=0.1, init_const=-3.0):
        # approx scale = init_scale * np.sqrt(2/np.pi)
        exp_dist = Exponential(np.sqrt(np.pi/2) / init_scale) 
        model.W_mu.data = exp_dist.rsample(model.W_mu.shape) * torch.sign(torch.randn_like(model.W_mu))  # we still allow negative weights (see section C.1 of our paper)
        model.b_mu.data = exp_dist.rsample(model.b_mu.shape) * torch.sign(torch.randn_like(model.b_mu))
        model.W_rho.data.fill_(init_const)  # to match the VCL authors' initialisation in code
        model.b_rho.data.fill_(init_const)