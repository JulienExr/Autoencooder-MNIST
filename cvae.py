import torch

def gn(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return torch.nn.GroupNorm(g, c)
    return torch.nn.GroupNorm(1, c)


class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.gn1 = gn(channels)
        self.act1 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = gn(channels)
        self.act2 = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.gn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        return x + out
    

class CVAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def forward(self, x, label):
        mu, logvar, embed_label = self.encoder(x, label)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z, embed_label)
        return decoded, mu, logvar

class CVAE_Encoder(torch.nn.Module):
    def __init__(self, latent_dim=256, embedding_dim=64):
        super(CVAE_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # 1x28x28 -> 32x14x14
        self.res1 = ResBlock(32)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x14x14 -> 64x7x7
        self.res2 = ResBlock(64)

        self.res3 = ResBlock(64)

        self.flatten = torch.nn.Flatten()
        self.fc_mu = torch.nn.Linear(64 * 7 * 7 + embedding_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(64 * 7 * 7 + embedding_dim, latent_dim)
        self.embed = torch.nn.Embedding(10, embedding_dim=embedding_dim)
    
    def forward(self, x, label):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.flatten(x)
        embed_label = self.embed(label)
        x = torch.cat([x, embed_label], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, embed_label

class CVAE_Decoder(torch.nn.Module):
    def __init__(self, latent_dim=256, embedding_dim=64):
        super(CVAE_Decoder, self).__init__()
        self.linear = torch.nn.Linear(latent_dim + embedding_dim, 64 * 7 * 7) 
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')                 # 64x7x7 -> 64x14x14

        self.res1 = ResBlock(64)
        self.deconv1 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # 64x14x14 -> 32x14x14

        self.res2 = ResBlock(32)
        self.deconv2 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)   # 32x28x28 -> 16x28x28

        self.res3 = ResBlock(16)
        self.deconv3 = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)   # 16x28x28 -> 1x28x28

        self.output_activation = torch.nn.Tanh()
    
    def forward(self, x, embed_label):
        x = torch.cat([x, embed_label], dim=1)
        x = self.linear(x)
        x = x.view(x.size(0), 64, 7, 7)
        
        x = self.res1(x)
        x = self.up(x)
        x = self.deconv1(x)

        x = self.res2(x)
        x = self.up(x)
        x = self.deconv2(x)

        x = self.res3(x)
        x = self.deconv3(x)
        x = self.output_activation(x)

        return x
    

def build_cvae(latent_dim=256, embed_dim=64):
    encoder = CVAE_Encoder(latent_dim=latent_dim, embedding_dim=embed_dim)
    decoder = CVAE_Decoder(latent_dim=latent_dim, embedding_dim=embed_dim)
    cvae = CVAE(encoder, decoder)
    return cvae