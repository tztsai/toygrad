from .basic import *


class AutoEncoder(Model):
    def fit(self, input, **kwds) -> dict:
        if vd := kwds.get('val_data', None):
            kwds['val_data'] = (vd, vd)
        return super().fit(input, input, **kwds)


class VAE(AutoEncoder):
    """ Variational AutoEncoder. """

    kl_div_weight = 1e-3  # weight of KL divergence in the loss
    
    def __init__(self, encoder, decoder, latent_dim):
        self.enc = encoder
        self.dec = decoder
        self.ld = latent_dim
        self.mu = affine(latent_dim)
        self.sigma = affine(latent_dim)
        
    def apply(self, input):
        e = self.enc(input)
        mu, sigma = self.mu(e), self.sigma(e)
        self.kl_loss = self.kl_div(mu, sigma) * self.kl_div_weight
        z = mu + sigma * np.random.randn(self.ld)
        return self.dec(z)

    @staticmethod
    def kl_div(mu, sigma):
        """ KL divergence w.r.t. the N(0, I) prior. """
        mu2, sigma2 = mu ** 2, sigma ** 2 + 1e-6
        return mu2.mean() + sigma2.mean() - sigma2.log().mean()

    def getloss(self, obj):
        loss = super().getloss(obj)
        def total_loss(input, output):
            return loss(input, output) + self.kl_loss
        return total_loss
