import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CauchyLoss(nn.Module):
    """Cauchy loss for extreme outliers"""
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
    
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.log(1 + (diff / self.c) ** 2)
        return loss.mean()

class LogCoshLoss(nn.Module):
    """Log-Cosh loss - smooth robust alternative"""
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.log(torch.cosh(diff))
        return loss.mean()

class QuantileLoss(nn.Module):
    """Quantile loss for asymmetric predictions"""
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        loss = torch.where(
            diff >= 0,
            self.quantile * diff,
            (self.quantile - 1) * diff
        )
        return loss.mean()

class HeteroscedasticLoss(nn.Module):
    """Heteroscedastic loss - model predicts both mean and variance"""
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, y_true):
        """
        Args:
            outputs: tensor of shape (batch, 2) where [:, 0] is mean, [:, 1] is log_var
            y_true: true values
        """
        # BOTH ARGUMENTS ARE NORMALISED, NOT JUST THE OUTPUTS.
        #
        # `mean` below is a COLUMN, (batch, 1). If y_true arrives 1-D, (batch,),
        # then `y_true - mean` BROADCASTS to (batch, batch) and the loss is
        # averaged over every molecule PAIR in the batch rather than over the
        # molecules. The fit still runs and still reports a number; it simply
        # does not learn. Measured on the laboratory runner, hERG, ECFP4:
        # R-squared -0.005 with a 1-D target against +0.289 with a column, same
        # data and seed, and predictions spanning 0.07 log units against a label
        # spread of 0.97.
        #
        # QM9 always passed a column (models.py views its targets as (-1, 1)) so
        # this never bit there and these reshapes are a no-op for it -- verified,
        # no QM9 number moves. The laboratory runner passes a ravelled 1-D array,
        # because every one of its other models has a one-wide head. Normalising
        # here fixes the class rather than one caller (found 2026-09-01 by
        # running both networks).
        #
        # reshape, not view: view raises on a non-contiguous tensor.
        outputs = outputs.reshape(-1, 2)
        y_true = y_true.reshape(-1, 1)

        mean = outputs[:, 0:1]
        log_var = outputs[:, 1:2]
        
        # Prevent numerical instability
        log_var = torch.clamp(log_var, -10, 10)
        
        # NLL = 0.5 * (log(var) + (y - mu)^2 / var)
        precision = torch.exp(-log_var)
        loss = 0.5 * (log_var + precision * (y_true - mean) ** 2)
        return loss.mean()

class BarronAdaptiveLoss(nn.Module):
    """Barron's adaptive loss with learnable alpha parameter"""
    def __init__(self, alpha_init=2.0, scale_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.scale = nn.Parameter(torch.tensor(scale_init))
    
    def forward(self, y_pred, y_true):
        diff = (y_pred - y_true) / self.scale
        alpha = self.alpha
        
        # Barron's general loss formula
        # L(x, α, c) = |α - 2|/α * ((x²/|α-2| + 1)^(α/2) - 1)
        if torch.abs(alpha - 2.0) < 1e-3:
            # When alpha ≈ 2, use MSE (limit case)
            loss = diff ** 2
        else:
            loss = (torch.abs(alpha - 2.0) / alpha) * \
                   ((diff ** 2 / torch.abs(alpha - 2.0) + 1) ** (alpha / 2.0) - 1)
        
        return loss.mean()

class DomainWeightedLoss(nn.Module):
    """Domain-weighted loss with learnable weights per domain"""
    def __init__(self, num_domains, base_loss='mse'):
        super().__init__()
        self.num_domains = num_domains
        self.domain_weights = nn.Parameter(torch.ones(num_domains))
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif base_loss == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def forward(self, y_pred, y_true, domain_labels):
        """
        Args:
            y_pred: predictions
            y_true: true values
            domain_labels: tensor of domain indices for each sample
        """
        base_losses = self.base_loss(y_pred, y_true).squeeze()
        
        # Apply softmax to ensure positive weights that sum to num_domains
        weights = F.softmax(self.domain_weights, dim=0) * self.num_domains
        
        # Weight each loss by its domain
        weighted_losses = base_losses * weights[domain_labels]
        
        return weighted_losses.mean()

class DomainBalancedLoss(nn.Module):
    """Domain-balanced loss - equal weight per domain"""
    def __init__(self, num_domains, base_loss='mse'):
        super().__init__()
        self.num_domains = num_domains
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif base_loss == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def forward(self, y_pred, y_true, domain_labels):
        """Equal weight per domain regardless of size"""
        base_losses = self.base_loss(y_pred, y_true).squeeze()
        
        # Calculate mean loss per domain, then average across domains
        domain_losses = []
        for domain_idx in range(self.num_domains):
            mask = domain_labels == domain_idx
            if mask.sum() > 0:
                domain_losses.append(base_losses[mask].mean())
        
        return torch.stack(domain_losses).mean()

class HeteroscedasticPerDomainLoss(nn.Module):
    """Heteroscedastic NLL with separate variance per domain"""
    def __init__(self, num_domains):
        super().__init__()
        self.num_domains = num_domains
        # Learnable log-variance per domain
        self.log_vars = nn.Parameter(torch.zeros(num_domains))
    
    def forward(self, y_pred, y_true, domain_labels):
        """
        Args:
            y_pred: mean predictions
            y_true: true values
            domain_labels: domain indices
        """
        # Get variance for each sample based on its domain
        log_var = self.log_vars[domain_labels].unsqueeze(1)
        log_var = torch.clamp(log_var, -10, 10)
        
        precision = torch.exp(-log_var)
        loss = 0.5 * (log_var + precision * (y_true - y_pred) ** 2)
        return loss.mean()

def get_loss_function(loss_name, **kwargs):
    """Factory function to get loss by name"""
    loss_map = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'huber': nn.HuberLoss(delta=kwargs.get('delta', 1.0)),
        'cauchy': CauchyLoss(c=kwargs.get('c', 1.0)),
        'log_cosh': LogCoshLoss(),
        'quantile_0.1': QuantileLoss(quantile=0.1),
        'quantile_0.5': QuantileLoss(quantile=0.5),
        'quantile_0.9': QuantileLoss(quantile=0.9),
        'heteroscedastic': HeteroscedasticLoss(),
        'barron': BarronAdaptiveLoss(
            alpha_init=kwargs.get('alpha_init', 2.0),
            scale_init=kwargs.get('scale_init', 1.0)
        ),
        'domain_weighted': DomainWeightedLoss(
            num_domains=kwargs.get('num_domains', 1),
            base_loss=kwargs.get('base_loss', 'mse')
        ),
        'domain_balanced': DomainBalancedLoss(
            num_domains=kwargs.get('num_domains', 1),
            base_loss=kwargs.get('base_loss', 'mse')
        ),
        'het_per_domain': HeteroscedasticPerDomainLoss(
            num_domains=kwargs.get('num_domains', 1)
        ),
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_name]

class EvidentialLoss(nn.Module):
    """Evidential Deep Learning for uncertainty quantification"""
    def __init__(self, coeff=1.0):
        super().__init__()
        self.coeff = coeff
    
    def forward(self, outputs, y_true):
        """
        Args:
            outputs: tensor of shape (batch, 4) where:
                [:, 0] = gamma (predicted value)
                [:, 1] = v (degrees of freedom, >1)
                [:, 2] = alpha (pseudo-observations, >1) 
                [:, 3] = beta (uncertainty scale, >0)
            y_true: true values
        """
        gamma = outputs[:, 0:1]
        v = outputs[:, 1:2]
        alpha = outputs[:, 2:3]
        beta = outputs[:, 3:4]
        
        # Ensure parameters are in valid ranges
        v = F.softplus(v) + 1.0  # v > 1
        alpha = F.softplus(alpha) + 1.0  # alpha > 1
        beta = F.softplus(beta)  # beta > 0
        
        # NLL loss for Normal-Inverse-Gamma
        diff = (y_true - gamma) ** 2
        
        nll = 0.5 * torch.log(np.pi / v) \
              - alpha * torch.log(2 * beta) \
              + (alpha + 0.5) * torch.log(v * diff + 2 * beta) \
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        # Regularization term to penalize high uncertainty on training data
        reg = diff * (2 * v + alpha)
        
        return (nll + self.coeff * reg).mean()


class AdaptiveDomainWeightedLoss(nn.Module):
    """Domain-weighted loss with adaptive reweighting based on domain performance"""
    def __init__(self, num_domains, base_loss='mse', adaptation_rate=0.01):
        super().__init__()
        self.num_domains = num_domains
        self.domain_weights = nn.Parameter(torch.ones(num_domains))
        self.adaptation_rate = adaptation_rate
        
        # Track domain performance over time
        self.register_buffer('domain_errors', torch.zeros(num_domains))
        self.register_buffer('domain_counts', torch.zeros(num_domains))
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif base_loss == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none')
    
    def forward(self, y_pred, y_true, domain_labels):
        base_losses = self.base_loss(y_pred, y_true).squeeze()
        
        # Update domain error tracking
        if self.training:
            with torch.no_grad():
                for domain_idx in range(self.num_domains):
                    mask = domain_labels == domain_idx
                    if mask.sum() > 0:
                        self.domain_errors[domain_idx] += base_losses[mask].mean()
                        self.domain_counts[domain_idx] += 1
        
        # Adaptive weights: upweight domains with higher errors
        if self.domain_counts.sum() > 0:
            avg_errors = self.domain_errors / (self.domain_counts + 1e-8)
            adaptive_weights = F.softmax(avg_errors / avg_errors.mean(), dim=0) * self.num_domains
        else:
            adaptive_weights = torch.ones(self.num_domains)
        
        # Combine learned and adaptive weights
        combined_weights = F.softmax(self.domain_weights, dim=0) * self.num_domains
        final_weights = (1 - self.adaptation_rate) * combined_weights + self.adaptation_rate * adaptive_weights
        
        weighted_losses = base_losses * final_weights[domain_labels]
        return weighted_losses.mean()


class FocalLoss(nn.Module):
    """Focal loss for regression - focuses on hard examples"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        # Normalize errors
        diff = torch.abs(y_pred - y_true)
        max_diff = diff.max() + 1e-8
        normalized_diff = diff / max_diff
        
        # Focal weight: (1 - p)^gamma where p = 1 - normalized_error
        focal_weight = (normalized_diff) ** self.gamma
        
        # Base MSE loss
        mse_loss = (y_pred - y_true) ** 2
        
        # Apply focal weighting
        loss = self.alpha * focal_weight * mse_loss
        return loss.mean()


class TruncatedLoss(nn.Module):
    """Truncated loss - ignore extreme outliers beyond threshold"""
    def __init__(self, base_loss='mse', quantile=0.9):
        super().__init__()
        self.quantile = quantile
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
    
    def forward(self, y_pred, y_true):
        losses = self.base_loss(y_pred, y_true).squeeze()
        
        # Only use losses below the quantile threshold
        threshold = torch.quantile(losses, self.quantile)
        mask = losses <= threshold
        
        if mask.sum() > 0:
            return losses[mask].mean()
        else:
            return losses.mean()


class MixtureDomainLoss(nn.Module):
    """Mixture of experts approach - separate loss per domain with gating"""
    def __init__(self, num_domains):
        super().__init__()
        self.num_domains = num_domains
        
        # Separate loss parameters per domain
        self.domain_scales = nn.Parameter(torch.ones(num_domains))
        self.domain_shapes = nn.Parameter(torch.ones(num_domains) * 2.0)  # For Barron-like loss
    
    def forward(self, y_pred, y_true, domain_labels):
        losses = []
        
        for domain_idx in range(self.num_domains):
            mask = domain_labels == domain_idx
            if mask.sum() == 0:
                continue
            
            y_pred_domain = y_pred[mask]
            y_true_domain = y_true[mask]
            
            # Domain-specific Barron loss
            scale = F.softplus(self.domain_scales[domain_idx])
            alpha = F.softplus(self.domain_shapes[domain_idx]) + 0.1
            
            diff = (y_pred_domain - y_true_domain) / scale
            
            if torch.abs(alpha - 2.0) < 1e-3:
                domain_loss = diff ** 2
            else:
                domain_loss = (torch.abs(alpha - 2.0) / alpha) * \
                             ((diff ** 2 / torch.abs(alpha - 2.0) + 1) ** (alpha / 2.0) - 1)
            
            losses.append(domain_loss.mean())
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

class EvidentialCauchyLoss(nn.Module):
    """Evidential regression with Cauchy likelihood - robust to extreme outliers"""
    def __init__(self, coeff=1.0):
        super().__init__()
        self.coeff = coeff
    
    def forward(self, outputs, y_true):
        gamma = outputs[:, 0:1]
        lambda_param = F.softplus(outputs[:, 1:2]) + 1e-6
        alpha = F.softplus(outputs[:, 2:3]) + 1.0
        beta = F.softplus(outputs[:, 3:4])
        
        diff = y_true - gamma
        cauchy_ll = -torch.log(lambda_param) - torch.log(1 + (diff / lambda_param) ** 2)
        prior_reg = -alpha * torch.log(beta) + torch.lgamma(alpha)
        evidence_reg = (diff ** 2) * (2 * alpha / beta)
        
        return (-cauchy_ll + prior_reg + self.coeff * evidence_reg).mean()

class EvidentialLaplaceLoss(nn.Module):
    """Evidential regression with Laplace likelihood - middle ground"""
    def __init__(self, coeff=1.0):
        super().__init__()
        self.coeff = coeff
    
    def forward(self, outputs, y_true):
        gamma = outputs[:, 0:1]
        b = F.softplus(outputs[:, 1:2]) + 1e-6
        alpha = F.softplus(outputs[:, 2:3]) + 1.0
        beta = F.softplus(outputs[:, 3:4])
        
        diff = torch.abs(y_true - gamma)
        laplace_ll = -torch.log(2 * b) - diff / b
        prior_reg = -alpha * torch.log(beta) + torch.lgamma(alpha)
        evidence_reg = diff * (2 * alpha / beta)
        
        return (-laplace_ll + prior_reg + self.coeff * evidence_reg).mean()

class SampleAdaptiveBarronLoss(nn.Module):
    """Barron loss where model learns per-sample robustness"""
    def __init__(self, scale_init=1.0, alpha_min=0.1, alpha_max=4.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
    
    def forward(self, outputs, y_true):
        y_pred = outputs[:, 0:1]
        alpha_raw = outputs[:, 1:2]
        alpha = torch.sigmoid(alpha_raw) * (self.alpha_max - self.alpha_min) + self.alpha_min
        
        diff = (y_true - y_pred) / self.scale
        eps = 1e-3
        loss = torch.where(
            torch.abs(alpha - 2.0) < eps,
            diff ** 2,
            (torch.abs(alpha - 2.0) / alpha) * 
            ((diff ** 2 / torch.abs(alpha - 2.0) + 1) ** (alpha / 2.0) - 1)
        )
        return loss.mean()

class StratifiedLoss(nn.Module):
    """Blend MSE for certain samples, Cauchy for uncertain"""
    def __init__(self, cauchy_scale=1.0):
        super().__init__()
        self.cauchy_scale = cauchy_scale
    
    def forward(self, outputs, y_true):
        y_pred = outputs[:, 0:1]
        uncertainty_logit = outputs[:, 1:2]
        uncertainty_prob = torch.sigmoid(uncertainty_logit)
        
        mse_loss = (y_true - y_pred) ** 2
        cauchy_loss = torch.log(1 + (y_true - y_pred) ** 2 / self.cauchy_scale)
        
        return ((1 - uncertainty_prob) * mse_loss + uncertainty_prob * cauchy_loss).mean()

# Update get_loss_function
def get_loss_function(loss_name, **kwargs):
    """Factory function to get loss by name"""
    loss_map = {
        # Basic losses
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'huber': nn.HuberLoss(delta=kwargs.get('delta', 1.0)),
        
        # Robust losses
        'cauchy': CauchyLoss(c=kwargs.get('c', 1.0)),
        'log_cosh': LogCoshLoss(),
        'focal': FocalLoss(gamma=kwargs.get('gamma', 2.0), alpha=kwargs.get('alpha', 0.25)),
        'truncated': TruncatedLoss(base_loss=kwargs.get('base_loss', 'mse'), quantile=kwargs.get('quantile', 0.9)),
        
        # Quantile losses
        'quantile_0.1': QuantileLoss(quantile=0.1),
        'quantile_0.5': QuantileLoss(quantile=0.5),
        'quantile_0.9': QuantileLoss(quantile=0.9),
        
        # Uncertainty-aware losses
        'heteroscedastic': HeteroscedasticLoss(),
        'evidential': EvidentialLoss(coeff=kwargs.get('coeff', 1.0)),
        
        # Adaptive losses
        'barron': BarronAdaptiveLoss(
            alpha_init=kwargs.get('alpha_init', 2.0),
            scale_init=kwargs.get('scale_init', 1.0)
        ),
        
        # Domain-specific losses
        'domain_weighted': DomainWeightedLoss(
            num_domains=kwargs.get('num_domains', 1),
            base_loss=kwargs.get('base_loss', 'mse')
        ),
        'domain_balanced': DomainBalancedLoss(
            num_domains=kwargs.get('num_domains', 1),
            base_loss=kwargs.get('base_loss', 'mse')
        ),
        'het_per_domain': HeteroscedasticPerDomainLoss(
            num_domains=kwargs.get('num_domains', 1)
        ),
        'adaptive_domain': AdaptiveDomainWeightedLoss(
            num_domains=kwargs.get('num_domains', 1),
            base_loss=kwargs.get('base_loss', 'mse'),
            adaptation_rate=kwargs.get('adaptation_rate', 0.01)
        ),
        'mixture_domain': MixtureDomainLoss(
            num_domains=kwargs.get('num_domains', 1)
        ),

        'evidential_cauchy': EvidentialCauchyLoss(coeff=kwargs.get('coeff', 1.0)),
        'evidential_laplace': EvidentialLaplaceLoss(coeff=kwargs.get('coeff', 1.0)),
        'sample_adaptive_barron': SampleAdaptiveBarronLoss(
            scale_init=kwargs.get('scale_init', 1.0),
            alpha_min=kwargs.get('alpha_min', 0.1),
            alpha_max=kwargs.get('alpha_max', 4.0)
        ),
        'stratified': StratifiedLoss(cauchy_scale=kwargs.get('cauchy_scale', 1.0)),
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_name]