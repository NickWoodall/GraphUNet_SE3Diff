"""R^3 diffusion methods. Should translate to OH dimensions on edges from molecular diff paper"""
import numpy as np
from scipy.special import gamma
import torch


class OHDiffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, oh_conf):
        """
        copying r3 for now, this based purl
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._oh_conf = oh_conf
        self.min_b = oh_conf.min_b
        self.max_b = oh_conf.max_b

    def b_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t*(self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, 3))

    def marginal_b_t(self, t):
        return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        #added scale to remove scale, consider other 
        """Samples marginal p(x(t) | x(t-1)).
        
        n is num nodes
        k is edges per node
        d is dimensionality per edge

        Args:
            x_0: [..., n, k, d] initial dims.
            t: continuous time in [0, 1].

        Returns:
            x_t: [...,n, k, d] positions at time t in Angstroms.
            score_t: [..., n, k, d] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float, scale=False):
        #added scale boolean to remove scaling
        """Samples marginal p(x(t) | x(0)).
        
                
        n is num nodes
        k is edges per node
        d is dimensionality per edge
        f is 2 [0 and 1] , we do both to pick out the right one during graph making

        Args:
            x_0: [..., n, k, d, f] initial dims.
            t: continuous time in [0, 1].

        Returns:
            x_t: [...,n, k, d, f] one hotd dims edges at time t 
            score_t: [..., n, k, d, f] score at time t in 
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        score_t = self.score(x_t, x_0, t)

        
        
        return x_t, score_t

    def score_scaling(self, t: float):
        return 1 / np.sqrt(self.conditional_var(t))
    

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step
        
                n is num nodes
                k is edges per node
                d is dimensionality per edge
                
                n,k,d replaces 3 in one hot dimensions on edges


        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
            
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        return x_t_1

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t, use_torch=use_torch)
