import torch
import math
from torch import nn

class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        ) # q(y|x) is modeled as a Gaussian distribution. This MLP is used to estimate the mean.
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        ) # This MLP is used to estimate the variance.
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh(),
            nn.Linear(y_size // 4, y_size // 16),
            nn.Tanh(),
            # nn.Linear(y_size // 16, y_size // 64),
            # nn.Tanh()
        )  # Project the textual featuers Y into low dimensions (768->48) to estimate the differential entropy H(Y)
        # Large dimension will cause the computation unstable, cause the determinant of the covariance matrix will be singular

    def calculate_entropy(self, features, epsilon=1e-6, use_full_cov=None):
        """
            Calculate the multivariate Gaussian entropy of the reduced-dimensional features Y (batch, 96).

            Formula: H(Y) = 0.5 * k * (1 + ln(2*pi)) + 0.5 * ln(|Sigma|)

            Parameters:

            features (torch.Tensor): Features of dimension (batch_size, 96)

            epsilon (float): Numerical stability term to prevent the determinant from being zero

            Returns:

            entropy (torch.Tensor): Scalar entropy value
        """
        batch_size, feat_dim = features.shape

        # 1. Calculate the constant term: (k/2) * (1 + ln(2*pi))
        const_term = 0.5 * feat_dim * (1 + math.log(2 * math.pi))

        # 2. Determine whether the complete covariance matrix can be used.
        # To calculate the determinant, the following condition must be met: batch_size > feat_dim; otherwise, the matrix will be singular.
        # In our default settings, as batch size = 5, so use_full_cov = False
        if use_full_cov == None:
            use_full_cov = batch_size > feat_dim

        if use_full_cov:
            # --- When the batch size is large enough (e.g., >= 96), calculate the correlation between features. ---

            # Centralize the textual features
            diff = features - features.mean(dim=0)

            # Calculate the covariance matrix: (X^T * X) / (N - 1)
            # Dimensional change: (96, B) * (B, 96) -> (96, 96)
            cov_matrix = torch.matmul(diff.T, diff) / (batch_size - 1)

            # Adding jitter ensures positive definiteness (numerical stability).
            eye = torch.eye(feat_dim, device=features.device)
            cov_matrix = cov_matrix + epsilon * eye

            # Calculate the logarithm of the determinant (use slogdet to prevent numerical overflow).
            sign, logdet = torch.linalg.slogdet(cov_matrix)

            if sign <= 0:
                # Extreme case: Even with the epsilon matrix, the corvariance matrix is still not positive definite
                return torch.tensor(float('-inf')).to(features.device)

        else:
            # --- When batch size is small, we use the diagonal approximation of the covariance matrix ---
            # In this case, we assume that the features are independent across dimensions and don't estimate the dimensional correlation;
            # Log(|Sigma|) = Sum(Log(Var_i))

            variances = features.var(dim=0, unbiased=True) + epsilon
            logdet = torch.sum(torch.log(variances))

            # Print warnings for debugging (optional)
            # print(f"Warning: Batch size ({batch_size}) <= Dim ({feat_dim}). Using diagonal approximation.")

        # 3. Final entropy calculation
        entropy = const_term + 0.5 * logdet

        return entropy

    def forward(self, x, y):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x) # (bs, hidden_size)

        positive = -(mu - y)**2/2./torch.exp(logvar)
        lld = torch.mean(torch.sum(positive,-1))

        # For Gaussian Distribution Estimation
        y = self.entropy_prj(y) # For computing H(Y), the low-dimension projection is necessary
        H = self.calculate_entropy(y, use_full_cov=1)

        return lld, H

class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):  # Here, 'y' represents the fused feature.
        """Calulate the score 
        """
        x_pred = self.net(y)    # bs, emb_size # 这里也是用fusion特征估计原始特征

        # normalize to unit sphere
        if len(x_pred.shape) == 2:
            x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
            x = x / x.norm(dim=1, keepdim=True)
        else:
            x_pred = x_pred / x_pred.norm(dim=0, keepdim=True)
            x = x / x.norm(dim=0, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce

