import torch
import torch.distributions as dist

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# In[2]:

class AddPoissonNoise(object):
    def __init__(self, intensity = 0.5, clip = 1):
        self.intensity = intensity
        self.clip = clip
        
    def __call__(self, tensor):
        
        # Create a Poisson distribution with the specified intensity
        poisson = dist.Poisson(self.intensity*255)

        # Generate random noise with the same size as the image
        noise = poisson.sample(tensor.size())/255

        # Add the noise to the image
        noisy_image = tensor + noise

        # Clip the image to ensure pixel values are within the valid range
        noisy_image = torch.clamp(noisy_image, 0, self.clip)

        # Convert the noisy image back to the original data type
        noisy_image = noisy_image.type(tensor.dtype)

        return noisy_image
    
    def __repr__(self):
        return self.__class__.__name__ + '(intensity={0})'.format(self.intensity)
# In[3]:

def perturb_model(perturbation_frac, model):
    for param in model.parameters():
        param.data.mul_(1 + perturbation_frac)
    return model
