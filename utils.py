import torch
import torchaudio
import resampy

def lps(x, power=2.):
    if power is None or x.ndim == 4:
        return torch.log(torch.square(x[...,0]) + 1e-7)
    elif power == 2:
        return torch.log(torch.square(x) + 1e-7)
    elif power == 1:
        return torch.log(x + 1e-7)

def from_mag_phase(m, p):
    return torch.cat([(m * torch.cos(p)).unsqueeze(-1), 
                    (m * torch.sin(p)).unsqueeze(-1)], -1)

def get_art_phase(p, n=270):
    art_p = torch.cat([p, -torch.flip(p, [-2]), p], dim=-2)
    if p.ndim == 3:
        return art_p[:,:n]
    elif p.ndim == 2:
        return art_p[:n]

def full_lps(wave, win_length=32, hop_length=8, hann=False, power=2.0, T=32):
    win=lambda x: torch.ones(x) if not hann else torch.hann_window(x)
    spec = torchaudio.transforms.Spectrogram(n_fft=((141 + 129) * 2) - 1, 
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    window_fn=win, 
                                    power=power, normalized=False)(wave)
    return lps(spec)[...,:T] if T is not None else lps(spec)

def lowres_lps(wave, rate_div, win_length=32, hop_length=8, hann=False, power=2.0, T=32, rate=16000):
    win=lambda x: torch.ones(x) if not hann else torch.hann_window(x)
    wave = resampy.resample(wave.cpu().detach().numpy(),
                            rate, rate // rate_div, axis=-1, filter='sinc_window', num_zeros=64)
    wave = torch.tensor(wave)
    spec = torchaudio.transforms.Spectrogram(n_fft=(129 * 2) - 1, pad=3,
                                    win_length=win_length, hop_length=hop_length // rate_div, window_fn=win, 
                                    power=power, normalized=False)(wave)
    return lps(spec)[...,:T] if T is not None else lps(spec)

###########

def full_stft():
    
def lowres_m_p(wave, rate_div, win_length=32, hop_length=8, hann=False, T=None, rate=16000):
    if isinstance(wave, torch.Tensor):
        wave = wave.numpy()
    wave = resampy.resample(wave, rate, rate // rate_div, axis=-1, filter='sinc_window', num_zeros=64)
    wave = torch.tensor(wave)
    spec = torchaudio.functional.spectrogram(wave, pad=pad, window=win(win_length), n_fft=(129 * 2) - 1, 
                                      hop_length=hop_length // rate_div, win_length=win_length, 
                                      power=power, normalized=False)
    return lps(spec)[...,:T] if T is not None else lps(spec)

def wav_to_wav(wav_path, G, power=2, T=32):
    rate, wav = wavfile.read(wav_path)
    lowres_m, lowres_p = torchaudio.functional.magphase(lowres_stft(wav, 2))
    art_phase = get_art_phase(lowres_p)
    lf_lps = lps(lowres_m, power)
    fullres_wav = full_lps(torch.tensor(wav), T=None)
    lowres_wav_parts = torch.stack(torch.split(lf_lps, T, dim=-1)[:-1])
    highres_wav_parts = G(lowres_wav_parts)
    pred_mag = torch.cat([*torch.cat([lowres_wav_parts, highres_wav_parts], dim=1)], dim=1)
    pred_stft = from_mag_phase(pred_mag, art_phase)
    pred_audio = torchaudio.functional.istft(pred_stft, **spec_kwargs)
    wavfile.write('gen_' + wav_path, rate, pred_audio.numpy())
