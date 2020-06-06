import torch
import torchaudio
import resampy

def lps(x, power=2.):
    if power is None or x.ndim == 4:
        return torch.log(torch.square(x[...,0]) + 1e-7)
    elif power == 1:
        return torch.log(torch.square(x) + 1e-7)
    elif power == 2:
        return torch.log(x + 1e-7)

def from_mag_phase(m, p):
    return torch.cat([(m * torch.cos(p)).unsqueeze(-1), 
                    (m * torch.sin(p)).unsqueeze(-1)], -1)

def full_lps(wave, win_length=32, hop_length=8, hann=False, power=2.0, T=32):
    win=lambda x: torch.ones(x) if not hann else torch.hann_window(x)
    spec = torchaudio.transforms.Spectrogram(n_fft=((141 + 129) * 2) - 1, 
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    window_fn=win, 
                                    power=power, normalized=False)(wave)
    return lps(spec)[...,:T]

def lowres_lps(wave, rate_div, win_length=32, hop_length=8, hann=False, power=2.0, T=32, rate=16000):
    win=lambda x: torch.ones(x) if not hann else torch.hann_window(x)
    wave = resampy.resample(wave.cpu().detach().numpy(),
                            rate, rate // rate_div, axis=-1, filter='sinc_window', num_zeros=64)
    wave = torch.tensor(wave)
    spec = torchaudio.transforms.Spectrogram(n_fft=(129 * 2) - 1, pad=3,
                                    win_length=win_length, hop_length=hop_length // rate_div, window_fn=win, 
                                    power=power, normalized=False)(wave)
    return lps(spec)[...,:T]