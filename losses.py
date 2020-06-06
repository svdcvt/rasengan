import torch
from torch import nn

def reconstruction_loss(true, pred):
    return torch.mean(torch.sqrt(torch.mean((true - pred) ** 2, dim=1) + 1e-7))

def regularization_term(fake_prob, real_prob, fake_logit, real_logit):
    real_grad = torch.autograd.grad(outputs=real_prob, inputs=real_logit,
                                grad_outputs=torch.ones_like(real_prob),
                                retain_graph=True, create_graph=True, only_inputs=True)[0]
    fake_grad = torch.autograd.grad(outputs=fake_prob, inputs=fake_logit,
                                grad_outputs=torch.ones_like(fake_prob),
                                retain_graph=True, create_graph=True, only_inputs=True)[0]

    ep = torch.mean(torch.square(1 - real_prob) * torch.square(torch.norm(real_grad)))
    eq = torch.mean(torch.square(fake_prob) * torch.square(torch.norm(fake_grad)))
    return ep + eq

def disc_loss(fake_prob, real_prob):
    return -torch.mean(torch.log(real_prob + 1e-7)) - torch.mean(torch.log(1 - fake_prob + 1e-7))

def gen_loss(fake_prob):
    return torch.mean(torch.log(fake_prob + 1e-7))

def segsnr(true, pred):
    return torch.mean(10 * torch.log(1e-7 + torch.div(
                torch.mean(torch.square(true), 2), 
                torch.mean(torch.square(true - pred), 2))), 1)
