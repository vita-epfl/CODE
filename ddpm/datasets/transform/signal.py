from __future__ import annotations
import logging

from typing import Any, List, Tuple, Dict
from pathlib import Path
import numpy as np

import torchaudio
import torchaudio.sox_effects as sox
import torchaudio.transforms as T
import torch
from torch import Tensor, nn

LOG = logging.getLogger(__name__)


def resample_signal(waveform: Tensor, sample_rate: int, resample_rate: int) -> Tuple[Tensor, int]:
    if sample_rate == resample_rate:
        return waveform, sample_rate

    ### kaiser_best
    resampler = T.Resample(
        sample_rate,
        resample_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="kaiser_window",
        beta=14.769656459379492,
    )

    resampled_waveform = resampler.forward(waveform)
    return resampled_waveform, resample_rate


def signal_to_mono(waveform: Tensor) -> Tensor:
    if waveform is None:
        return None
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


class SoxEffectTransform(nn.Module):
    # http://sox.sourceforge.net/sox.html
    # https://pytorch.org/audio/stable/sox_effects.html
    # https://pysox.readthedocs.io/en/latest/api.html
    # https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html

    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name
        self.effects: List[List[str]] = []

    def reset(self):
        self.effects = []

    def __repr__(self) -> str:
        res = ''
        if self.name:
            res = f'Name: {self.name}\n'
        for e in self.effects:
            res += str(e) + '\n'
        return res

    @staticmethod
    def from_config(cfg: Any) -> Dict[str, List[SoxEffectTransform]]:
        transforms = {}
        for mode in ['train','test','validation']:
            if cfg.dataset.low_quality_effect[mode]:
                transforms[mode] = []
                for effect in cfg.dataset.low_quality_effect[mode]['effects']:
                    sox = SoxEffectTransform(effect.name)
                    for param_string in effect.params:
                        params = param_string.strip().split(" ")
                        sox.add_effect(params)
                    if cfg.dataset.normalize_inputs_post_eq:
                        sox.normalize_gain()
                    transforms[mode].append(sox)
        return transforms


    @staticmethod
    def random_effects(cfg: Any) -> SoxEffectTransform:
        sox = SoxEffectTransform("random")
        cfg_effect = cfg.dataset.low_quality_effect['random']
        max_effect = cfg_effect.random_effects.max_effect
        min_effect = cfg_effect.random_effects.min_effect
        eq_freq = []
        treble = False
        bass = False
        overdrive = False
        reverb = False
        riaa = False
        echo = False
        band = False
        tremolo = False
        sinc = False
        hilbert = False
        flanger = False

        while len(sox.effects) < min_effect and len(sox.effects) < max_effect:
            
            #Equalizer
            while np.random.random() < cfg_effect.equalizer.proba and len(sox.effects) < max_effect:
                center_freq = cfg_effect.equalizer.freq_mean
                std_freq = cfg_effect.equalizer.freq_std
                for freq in eq_freq:
                    if np.abs(freq-center_freq) < 20:
                        continue
                eq_freq.append(center_freq)
                if np.random.random() > 0.5:
                    sox.add_equalizer(max(np.random.normal(center_freq, std_freq), 150), 
                        np.random.uniform(cfg_effect.equalizer.db_min,cfg_effect.equalizer.db_max), 
                        np.random.uniform(0.7, 5)
                    )
                else:
                    sox.add_equalizer(max(np.random.normal(center_freq, std_freq), 150), 
                        np.random.uniform(-cfg_effect.equalizer.db_max,-cfg_effect.equalizer.db_min), 
                        np.random.uniform(0.7, 5)
                    )
            #Treble
            if np.random.random() < cfg_effect.treble.proba and len(sox.effects) < max_effect and not treble:
                treble = True
                if np.random.random() > 0.5:
                    sox.add_treble(np.random.randint(cfg_effect.treble.freq_min, cfg_effect.treble.freq_max), 
                        np.random.uniform(cfg_effect.treble.db_min,cfg_effect.treble.db_max), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
                else:
                    sox.add_treble(np.random.randint(cfg_effect.treble.freq_min, cfg_effect.treble.freq_max), 
                        np.random.uniform(-cfg_effect.treble.db_max,cfg_effect.treble.db_min), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
            #Bass
            if np.random.random() < cfg_effect.bass.proba and len(sox.effects) < max_effect and not bass:
                bass = True
                if np.random.random() > 0.5:
                    sox.add_bass(np.random.randint(cfg_effect.bass.freq_min, cfg_effect.bass.freq_max),
                        np.random.uniform(cfg_effect.bass.db_min,cfg_effect.bass.db_max), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
                else:
                    sox.add_bass(np.random.randint(cfg_effect.bass.freq_min, cfg_effect.bass.freq_max),
                        np.random.uniform(-cfg_effect.bass.db_max,-cfg_effect.bass.db_min), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
            #Overdrive
            if np.random.random() < cfg_effect.overdrive.proba and len(sox.effects) < max_effect and not overdrive:
                overdrive = True
                sox.add_overdrive(np.random.randint(cfg_effect.overdrive.min_int, cfg_effect.overdrive.max_int), 
                    np.random.randint(0, 15)
                )

            #Reverb
            if np.random.random() < cfg_effect.reverb.proba and len(sox.effects) < max_effect and not reverb:
                reverb = True
                if np.random.random() < cfg_effect.reverb.proba_w:
                    w_true = False
                else: 
                    w_true = False

                reverberance = np.random.randint(1, 100)
                damping = np.random.randint(1, 100)
                room_scale = np.random.randint(1, 100)
                stereo_depth = np.random.randint(1, 100)
                if cfg_effect.reverb.proba_delay:
                    delay = np.random.randint(1, 500) #in millisecond
                else:
                    delay = 0
                wet_gain = np.random.uniform(-10, 10)
                if w_true:
                    params_string = f"reverb -w {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                else:
                    params_string = f"reverb {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                sox.add_effect(params_string.strip().split(" "))
                sox.add_effect('channels 1'.strip().split(" "))
            
            # Flanger
            if np.random.random() < cfg_effect.flanger.proba and len(sox.effects) < max_effect and not flanger:
                flanger = True
                params_string = 'flanger'
                sox.add_effect(params_string.strip().split(" "))

            # riaa
            if np.random.random() < cfg_effect.riaa.proba and len(sox.effects) < max_effect and not riaa:
                riaa = True
                params_string = 'riaa'
                sox.add_effect(params_string.strip().split(" "))

            # hilbert
            if np.random.random() < cfg_effect.hilbert.proba and len(sox.effects) < max_effect and not hilbert:
                hilbert = True
                params_string = 'hilbert'
                sox.add_effect(params_string.strip().split(" "))

            #band
            if np.random.random() < cfg_effect.band.proba and len(sox.effects) < max_effect and not band:
                band = True
                width = np.random.uniform(100,3000)
                center = np.random.uniform(cfg_effect.band.min_freq, cfg_effect.band.max_freq)
                if np.random.random() < cfg_effect.band.noise_proba:
                    params_string = f"band -n {center} {width}"
                else :
                    params_string = f"band {center} {width}"
                sox.add_effect(params_string.strip().split(" "))

            #sinc
            if np.random.random() < cfg_effect.sinc.proba and len(sox.effects) < max_effect and not sinc:
                sinc = True
                att = np.random.uniform(40, cfg_effect.sinc.att_max)
                sign = np.random.choice([-1, 1])
                freq = np.random.uniform(cfg_effect.sinc.min_freq, cfg_effect.sinc.max_freq)
                params_string = f"sinc -a {att} {sign * freq} "
                sox.add_effect(params_string.strip().split(" "))

            # #tremolo
            # if np.random.random() < cfg_effect.tremolo.proba and len(sox.effects) < max_effect and not tremolo:
            #     tremolo = True
            #     speed = np.random.uniform(cfg_effect.tremolo.speed_min, cfg_effect.tremolo.speed_max)
            #     depth = np.random.uniform(cfg_effect.tremolo.depth_min, cfg_effect.tremolo.depth_max)
            #     params_string = f"tremolo {speed} {depth}"
            #     sox.add_effect(params_string.strip().split(" "))

            #echo
            if np.random.random() < cfg_effect.echo.proba and len(sox.effects) < max_effect and not echo:
                gain_in = np.random.uniform(cfg_effect.echo.gain_in_min, cfg_effect.echo.gain_in_max)
                gain_out = np.random.uniform(cfg_effect.echo.gain_out_min, cfg_effect.echo.gain_out_max)
                delay = []
                decay = []
                params_string = f"echos {gain_in} {gain_out} "
                while (not echo) or cfg_effect.echo.proba_next_echo > np.random.random():
                    echo = True
                    delay_value = np.random.uniform(cfg_effect.echo.delay_min, cfg_effect.echo.delay_max)
                    delay.append(delay_value)
                    decay_value = np.random.uniform(cfg_effect.echo.decay_min, cfg_effect.echo.decay_max)
                    decay.append(decay_value)
                for i, _ in enumerate(delay):
                    if i == 0 :  
                        params_string += f"{delay[i]} {decay[i]} "               
                sox.add_effect(params_string.strip().split(" "))
              
        if cfg.dataset.normalize_inputs_post_eq:
            sox.normalize_gain()
        return sox

    def to_mono(self, prepend: bool = True) -> SoxEffectTransform:
        """If prepend is True, the first effect will transform the audio to mono and then apply the effects"""
        effect = ["remix", "-"]
        if prepend:
            self.effects.insert(0, effect)
        else:
            self.effects.append(effect)
        return self

    def add_equalizer(self, center_freq: float, gain: float, Q: float = 0.707) -> SoxEffectTransform:
        """
        Apply a two-pole peaking equalisation (EQ) filter. With this filter, the signal-level at and around a selected frequency can be increased or decreased, whilst (unlike band-pass and band-reject filters) that at all other frequencies is unchanged.
        frequency gives the filter's central frequency in Hz, width, the band-width, and gain the required gain or attenuation in dB. Beware of Clipping when using a positive gain

        Taken from: https://howtoeq.wordpress.com/2010/10/07/q-factor-and-bandwidth-in-eq-what-it-all-means/
        Q factor (float) controls the bandwidth—or number of frequencies—that will be cut or boosted by the equaliser. The lower the Q factor, the wider the bandwidth (and the more frequencies will be affected).
        The higher the Q factor, the narrower the bandwidth (and the fewer frequencies will be affected).
        Q-factor
        0.7  = 2 octaves
        1    = 1 1/3 octaves
        1.4  = 1 octave
        2.8  = 1/2 octave
        4.3  = 1/3 octave
        8.6  = 1/6 octave
        """
        effect = ["equalizer", str(center_freq), str(Q), str(gain)]
        self.effects.append(effect)
        return self

    def add_overdrive(self, gain: float, colour: float) -> SoxEffectTransform:
        """
        gain (float) desired gain at the boost (or attenuation) in dB [0 to 100]
        colour	(float): controls the amount of even harmonic content in the over-driven output [0, 100]
        """
        effect = ["overdrive", str(gain), str(colour)]
        self.effects.append(effect)
        return self

    def add_treble(self, center_freq: float, gain: float, Q: float = 0.707) -> SoxEffectTransform:
        """
        gain (float) desired gain at the boost (or attenuation) in dB [-100 to 100]
        Q factor (float) controls the bandwidth—or number of frequencies—that will be impacted
        """
        effect = ['treble', str(gain), str(center_freq), str(Q)]
        self.effects.append(effect)
        return self

    def add_bass(self, center_freq: float, gain: float, Q: float = 0.707) -> SoxEffectTransform:
        """
        gain (float) desired gain at the boost (or attenuation) in dB [-100 to 100]
        Q factor (float) controls the bandwidth—or number of frequencies—that will be impacted
        """
        effect = ['bass', str(gain), str(center_freq), str(Q)]
        self.effects.append(effect)
        return self

    def add_effect(self, effect: List[Any]) -> SoxEffectTransform:
        effect = list(map(lambda x: str(x), effect))
        self.effects.append(effect)
        return self

    def normalize_gain(self, level: float = None) -> SoxEffectTransform:
        effect = ["gain", "-n"]
        if level is not None:
            effect.append(str(level))
        self.effects.append(effect)
        return self

    def add_gain(self, gain: float = 0) -> SoxEffectTransform:
        effect = ["gain", str(gain)]
        self.effects.append(effect)
        return self

    def apply_tensor(self, tensor: Tensor, sample_rate: int) -> Tuple[Tensor, int]:
        return sox.apply_effects_tensor(tensor, sample_rate, self.effects, channels_first=True)

    def forward(self, tensor: Tensor, sample_rate: int) -> Tuple[Tensor, int]:
        return self.apply_tensor(tensor, sample_rate)

    def apply_file(self, filepath: str) -> Tuple[Tensor, int]:
        return sox.apply_effects_file(filepath, self.effects, channels_first=True)

    def process_file(self, input_filepath: str, output_folder: str, override: bool = False) -> str:
        """Apply effect directly on an audio file and creates the transformed version by
        using the original name and self.name returns the filepath of the transformed file
        if transformed file already exists skip unless override is True
        """
        eq_name = self.name
        if not eq_name:
            eq_name = "eq_def"
        inpath = Path(input_filepath).expanduser().resolve()
        filename = str(inpath.stem)
        ext = inpath.suffix

        # Create folder if needed
        outfolder = Path(output_folder).expanduser().resolve()
        outfolder.mkdir(parents=True, exist_ok=True)

        output_path = outfolder / (filename + "_" + eq_name + ext)
        if output_path.exists() and not override:
            LOG.info(f"File exist, skipping: {output_path}")
            return str(output_path)

        waveform, sr = self.apply_file(inpath)
        torchaudio.save(output_path, waveform, sr)
        LOG.info(f"Wrote: {output_path}")
        return str(output_path)
