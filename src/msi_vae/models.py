from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    mu: torch.Tensor
    log_var: torch.Tensor
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


class VAE(nn.Module):
    def __init__(
        self, input_dim, latent_dim, beta=1, dimensions=[256, 64], loss_type="BCE"
    ) -> None:
        super(VAE, self).__init__()
        assert len(dimensions) == 2
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dimensions = dimensions
        self.beta = beta
        self.loss_type = loss_type

        # Encoder
        self.linear_1_enc = nn.Linear(input_dim, dimensions[0])
        self.linear_1_enc_batch_norm = nn.BatchNorm1d(dimensions[0])

        self.linear_2_enc = nn.Linear(dimensions[0], dimensions[1])
        self.linear_2_enc_batch_norm = nn.BatchNorm1d(dimensions[1])

        self.mean = nn.Linear(dimensions[1], latent_dim)
        self.mean_batch_norm = nn.BatchNorm1d(latent_dim)
        self.log_var = nn.Linear(dimensions[1], latent_dim)
        self.log_var_batch_norm = nn.BatchNorm1d(latent_dim)

        # Decoder
        self.linear_1_dec = nn.Linear(latent_dim, dimensions[1])
        self.linear_1_dec_batch_norm = nn.BatchNorm1d(dimensions[1])

        self.linear_2_dec = nn.Linear(dimensions[1], dimensions[0])
        self.linear_2_dec_batch_norm = nn.BatchNorm1d(dimensions[0])

        self.linear_3_dec = nn.Linear(dimensions[0], input_dim)

        # Weight Initialisation
        self.initialise_weights(0, 0.01)

    def initialise_weights(self, mu, std):
        torch.nn.init.normal_(self.linear_1_enc.weight, mean=mu, std=std)
        torch.nn.init.zeros_(self.linear_1_enc.bias)

        torch.nn.init.normal_(self.linear_2_enc.weight, mean=mu, std=std)
        torch.nn.init.zeros_(self.linear_2_enc.bias)

        torch.nn.init.normal_(self.mean.weight, mean=mu, std=std)
        torch.nn.init.zeros_(self.mean.bias)

        torch.nn.init.normal_(self.log_var.weight, mean=mu, std=std)
        torch.nn.init.zeros_(self.log_var.bias)

        torch.nn.init.normal_(self.linear_1_dec.weight, mean=mu, std=std)
        torch.nn.init.zeros_(self.linear_1_dec.bias)

        torch.nn.init.normal_(self.linear_2_dec.weight, mean=mu, std=std)
        torch.nn.init.zeros_(self.linear_2_dec.bias)

        torch.nn.init.normal_(self.linear_3_dec.weight, mean=mu, std=std)
        torch.nn.init.zeros_(self.linear_3_dec.bias)

    def encoder(self, x):
        h = self.linear_1_enc(x)
        h = self.linear_1_enc_batch_norm(h)
        h = F.relu(h)

        h = self.linear_2_enc(h)
        h = self.linear_2_enc_batch_norm(h)
        h = F.relu(h)

        mean = self.mean(h)
        mean = self.mean_batch_norm(mean)

        log_var = self.log_var(h)
        log_var = self.log_var_batch_norm(log_var)

        return mean, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        h = self.linear_1_dec(z)
        h = self.linear_1_dec_batch_norm(h)
        h = F.relu(h)

        h = self.linear_2_dec(h)
        h = self.linear_2_dec_batch_norm(h)
        h = F.relu(h)

        return F.sigmoid(self.linear_3_dec(h))

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        x_hat = self.decoder(z)
        loss_recon, loss_kl = self.loss(x_hat, x, mu, log_var)
        return VAEOutput(
            mu=mu,
            log_var=log_var,
            z_sample=z,
            x_recon=x_hat,
            loss=loss_recon + (self.beta * loss_kl),
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )

    def loss(self, x_hat, x, mu, log_var):
        if self.loss_type == "BCE":
            loss_recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
        else:
            loss_recon = F.mse_loss(x_hat, x, reduction="sum")
        loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return loss_recon, loss_kl


def get_torch_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
