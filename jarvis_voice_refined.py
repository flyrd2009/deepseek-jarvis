"""
JARVIS Voice System - REFINED EDITION (ERROR FIXED)
Professional grade voice synthesis with enhanced naturalness
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal, interpolate
from scipy.io import wavfile
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import threading
import queue
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ==================== ENUMS & CONSTANTS ====================
class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    WORRIED = "worried"
    SARCASTIC = "sarcastic"

class VoiceQuality(Enum):
    STANDARD = "standard"
    WARM = "warm"
    BRIGHT = "bright"
    DARK = "dark"
    BREATHY = "breathy"
    NASAL = "nasal"

# ==================== ADVANCED CONFIGURATION ====================
@dataclass
class ProsodyConfig:
    """Speech prosody parameters"""
    speech_rate: float = 1.0  # 0.5 to 2.0
    pitch_range: Tuple[float, float] = (80, 200)  # Hz range
    energy_range: Tuple[float, float] = (0.3, 1.0)  # Amplitude range
    pause_duration: float = 0.15  # Seconds between words
    sentence_pause: float = 0.5  # Seconds between sentences
    
@dataclass
class VoiceConfig:
    """Enhanced voice configuration - FIXED: Added all missing parameters"""
    # Basic parameters
    base_f0: float = 120.0
    gender: str = "male"
    age: str = "adult"
    
    # Advanced parameters
    formants: List[float] = None
    formant_bandwidths: List[float] = None
    formant_strength: float = 1.0
    
    # Voice quality
    breathiness: float = 0.08
    jitter: float = 0.01
    shimmer: float = 0.08
    creak: float = 0.0
    whisper: float = 0.0
    
    # Modulation
    vibrato_rate: float = 5.0
    vibrato_depth: float = 0.02
    tremolo_rate: float = 4.0
    tremolo_depth: float = 0.01
    
    # Spectral characteristics
    spectral_tilt: float = -6.0  # dB/octave
    noise_floor: float = -60.0  # dB
    
    # Effects - FIXED: Added modulation_freq here
    effect_type: str = "natural"
    modulation_freq: float = 0.0  # For robotic/alien effects
    reverb_amount: float = 0.0
    echo_delay: float = 0.0
    echo_decay: float = 0.0
    
    # Emotional state
    emotion: Emotion = Emotion.NEUTRAL
    emotion_intensity: float = 0.5
    
    # Voice quality
    quality: VoiceQuality = VoiceQuality.STANDARD
    
    # Speech rate - FIXED: Added missing parameter
    speech_rate: float = 1.0
    
    def __post_init__(self):
        if self.formants is None:
            self.set_formants_for_gender()
        if self.formant_bandwidths is None:
            self.formant_bandwidths = [60, 70, 100, 150, 200]
        self.apply_emotion_to_params()
        self.apply_quality_to_params()
    
    def set_formants_for_gender(self):
        """Professional formant settings"""
        if self.gender == "male":
            if self.age == "child":
                self.formants = [850, 1450, 2850, 3550, 4200]
            elif self.age == "young":
                self.formants = [620, 1250, 2650, 3350, 4000]
            elif self.age == "adult":
                self.formants = [550, 1150, 2550, 3250, 3900]
            elif self.age == "old":
                self.formants = [480, 1050, 2450, 3150, 3800]
        elif self.gender == "female":
            if self.age == "child":
                self.formants = [1050, 1850, 3250, 4050, 4700]
            elif self.age == "young":
                self.formants = [780, 1450, 2850, 3650, 4300]
            elif self.age == "adult":
                self.formants = [720, 1350, 2750, 3550, 4200]
            elif self.age == "old":
                self.formants = [680, 1250, 2650, 3450, 4100]
        else:  # child
            self.formants = [950, 1650, 3050, 3850, 4500]
    
    def apply_emotion_to_params(self):
        """Modify parameters based on emotion"""
        intensity = self.emotion_intensity
        
        if self.emotion == Emotion.HAPPY:
            self.base_f0 *= (1 + 0.2 * intensity)
            self.vibrato_rate += 2 * intensity
            self.vibrato_depth += 0.02 * intensity
            self.breathiness += 0.05 * intensity
            self.jitter += 0.01 * intensity
            
        elif self.emotion == Emotion.SAD:
            self.base_f0 *= (1 - 0.15 * intensity)
            self.speech_rate = max(0.7, 1.0 - 0.3 * intensity)
            self.vibrato_rate -= 1 * intensity
            self.breathiness += 0.1 * intensity
            self.jitter -= 0.005 * intensity
            
        elif self.emotion == Emotion.ANGRY:
            self.base_f0 *= (1 + 0.25 * intensity)
            self.formant_strength += 0.3 * intensity
            self.jitter += 0.02 * intensity
            self.shimmer += 0.1 * intensity
            self.creak += 0.1 * intensity
            
        elif self.emotion == Emotion.EXCITED:
            self.base_f0 *= (1 + 0.3 * intensity)
            self.speech_rate = min(2.0, 1.0 + 0.5 * intensity)
            self.vibrato_rate += 3 * intensity
            self.vibrato_depth += 0.03 * intensity
            # self.pitch_range = (100, 300)  # FIXED: pitch_range is in ProsodyConfig
            
        elif self.emotion == Emotion.CALM:
            self.base_f0 *= (1 - 0.1 * intensity)
            self.speech_rate = max(0.8, 1.0 - 0.2 * intensity)
            self.vibrato_depth *= (1 - 0.5 * intensity)
            self.jitter *= (1 - 0.5 * intensity)
            self.shimmer *= (1 - 0.5 * intensity)
    
    def apply_quality_to_params(self):
        """Apply voice quality modifications"""
        if self.quality == VoiceQuality.WARM:
            self.formants = [f * 0.9 for f in self.formants]
            self.breathiness += 0.1
            
        elif self.quality == VoiceQuality.BRIGHT:
            self.formants = [f * 1.15 for f in self.formants]
            self.spectral_tilt += 3
            
        elif self.quality == VoiceQuality.DARK:
            self.formants = [f * 0.85 for f in self.formants]
            self.spectral_tilt -= 3
            
        elif self.quality == VoiceQuality.BREATHY:
            self.breathiness += 0.2
            self.noise_floor += 10

# ==================== ENHANCED PHONEME DATABASE ====================
class PhonemeDatabase:
    """Professional phoneme database with coarticulation"""
    
    # IPA to internal mapping
    PHONEMES = {
        # Vowels with detailed formants
        'iy': {'type': 'vowel', 'name': 'heed', 'formants': [270, 2300, 3000, 3800, 4500], 'duration': 0.2},
        'ih': {'type': 'vowel', 'name': 'hid', 'formants': [390, 2000, 2600, 3400, 4200], 'duration': 0.18},
        'eh': {'type': 'vowel', 'name': 'head', 'formants': [530, 1850, 2500, 3300, 4100], 'duration': 0.18},
        'ae': {'type': 'vowel', 'name': 'had', 'formants': [660, 1700, 2400, 3200, 4000], 'duration': 0.2},
        'ah': {'type': 'vowel', 'name': 'hut', 'formants': [520, 1190, 2400, 3200, 4000], 'duration': 0.15},
        'aa': {'type': 'vowel', 'name': 'hot', 'formants': [730, 1090, 2450, 3250, 4050], 'duration': 0.18},
        'ao': {'type': 'vowel', 'name': 'hawed', 'formants': [570, 840, 2410, 3200, 4000], 'duration': 0.2},
        'uh': {'type': 'vowel', 'name': 'hood', 'formants': [440, 1020, 2250, 3100, 3900], 'duration': 0.15},
        'uw': {'type': 'vowel', 'name': 'who\'d', 'formants': [300, 870, 2250, 3100, 3900], 'duration': 0.2},
        'er': {'type': 'vowel', 'name': 'herd', 'formants': [490, 1350, 1700, 2500, 3500], 'duration': 0.22},
        
        # Consonants with detailed characteristics
        'p': {'type': 'plosive', 'voiced': False, 'place': 'bilabial', 'duration': 0.1, 'burst': True},
        'b': {'type': 'plosive', 'voiced': True, 'place': 'bilabial', 'duration': 0.1, 'burst': True},
        't': {'type': 'plosive', 'voiced': False, 'place': 'alveolar', 'duration': 0.1, 'burst': True},
        'd': {'type': 'plosive', 'voiced': True, 'place': 'alveolar', 'duration': 0.1, 'burst': True},
        'k': {'type': 'plosive', 'voiced': False, 'place': 'velar', 'duration': 0.12, 'burst': True},
        'g': {'type': 'plosive', 'voiced': True, 'place': 'velar', 'duration': 0.12, 'burst': True},
        
        'f': {'type': 'fricative', 'voiced': False, 'place': 'labiodental', 'duration': 0.18, 'noise_type': 'white'},
        'v': {'type': 'fricative', 'voiced': True, 'place': 'labiodental', 'duration': 0.16, 'noise_type': 'white'},
        'th': {'type': 'fricative', 'voiced': False, 'place': 'dental', 'duration': 0.16, 'noise_type': 'white'},
        'dh': {'type': 'fricative', 'voiced': True, 'place': 'dental', 'duration': 0.14, 'noise_type': 'white'},
        's': {'type': 'fricative', 'voiced': False, 'place': 'alveolar', 'duration': 0.2, 'noise_type': 'highpass'},
        'z': {'type': 'fricative', 'voiced': True, 'place': 'alveolar', 'duration': 0.18, 'noise_type': 'highpass'},
        'sh': {'type': 'fricative', 'voiced': False, 'place': 'postalveolar', 'duration': 0.2, 'noise_type': 'bandpass'},
        'zh': {'type': 'fricative', 'voiced': True, 'place': 'postalveolar', 'duration': 0.18, 'noise_type': 'bandpass'},
        'hh': {'type': 'fricative', 'voiced': False, 'place': 'glottal', 'duration': 0.12, 'noise_type': 'lowpass'},
        
        'ch': {'type': 'affricate', 'voiced': False, 'duration': 0.15},
        'jh': {'type': 'affricate', 'voiced': True, 'duration': 0.15},
        
        'm': {'type': 'nasal', 'voiced': True, 'place': 'bilabial', 'duration': 0.15, 'formants': [250, 1200, 2200]},
        'n': {'type': 'nasal', 'voiced': True, 'place': 'alveolar', 'duration': 0.14, 'formants': [250, 1300, 2300]},
        'ng': {'type': 'nasal', 'voiced': True, 'place': 'velar', 'duration': 0.15, 'formants': [250, 1100, 2100]},
        
        'l': {'type': 'liquid', 'voiced': True, 'place': 'alveolar', 'duration': 0.15, 'formants': [400, 1200, 2600]},
        'r': {'type': 'liquid', 'voiced': True, 'place': 'alveolar', 'duration': 0.15, 'formants': [350, 1100, 2200]},
        
        'w': {'type': 'glide', 'voiced': True, 'place': 'labiovelar', 'duration': 0.12, 'formants': [300, 800, 2100]},
        'y': {'type': 'glide', 'voiced': True, 'place': 'palatal', 'duration': 0.12, 'formants': [300, 2200, 2900]},
    }
    
    @classmethod
    def get_phoneme(cls, symbol):
        """Get phoneme data with error handling"""
        return cls.PHONEMES.get(symbol, cls.PHONEMES.get('ah'))  # Default to 'ah'

# ==================== ADVANCED TEXT PROCESSING ====================
class TextProcessor:
    """Advanced text processing with prosody markup"""
    
    # Enhanced word dictionary
    DICTIONARY = {
        # Greetings
        'hello': ['hh', 'ah', 'l', 'ow'],
        'hi': ['hh', 'ay'],
        'hey': ['hh', 'ey'],
        'good': ['g', 'uh', 'd'],
        'morning': ['m', 'ao', 'r', 'n', 'ih', 'ng'],
        'afternoon': ['ae', 'f', 't', 'er', 'n', 'uw', 'n'],
        'evening': ['iy', 'v', 'ih', 'n', 'ih', 'ng'],
        
        # Responses
        'yes': ['y', 'eh', 's'],
        'no': ['n', 'ow'],
        'okay': ['ow', 'k', 'ey'],
        'sure': ['sh', 'uh', 'r'],
        'absolutely': ['ae', 'b', 's', 'ah', 'l', 'uw', 't', 'l', 'iy'],
        
        # Actions
        'processing': ['p', 'r', 'aa', 's', 'eh', 's', 'ih', 'ng'],
        'analyzing': ['ae', 'n', 'ah', 'l', 'ay', 'z', 'ih', 'ng'],
        'computing': ['k', 'ah', 'm', 'p', 'y', 'uw', 't', 'ih', 'ng'],
        'calculating': ['k', 'ae', 'l', 'k', 'y', 'ah', 'l', 'ey', 't', 'ih', 'ng'],
        
        # Status
        'ready': ['r', 'eh', 'd', 'iy'],
        'online': ['aa', 'n', 'l', 'ay', 'n'],
        'active': ['ae', 'k', 't', 'ih', 'v'],
        'standby': ['s', 't', 'ae', 'n', 'd', 'b', 'ay'],
        
        # Jarvis specific
        'sir': ['s', 'er'],
        'boss': ['b', 'aa', 's'],
        'captain': ['k', 'ae', 'p', 't', 'ih', 'n'],
        'stark': ['s', 't', 'aa', 'r', 'k'],
        'jarvis': ['jh', 'aa', 'r', 'v', 'ih', 's'],
    }
    
    # Punctuation to pause mapping
    PAUSE_MARKS = {
        '.': 0.5,
        ',': 0.2,
        '!': 0.4,
        '?': 0.4,
        ':': 0.3,
        ';': 0.25,
    }
    
    @classmethod
    def process_text(cls, text, prosody: ProsodyConfig):
        """Process text with prosody information"""
        if not text:
            return [], []
        
        text = text.lower().strip()
        words = text.split()
        phonemes = []
        pauses = []
        
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalpha())
            punctuation = ''.join(c for c in word if c in '.,!?:;')
            
            # Get phonemes for word
            if clean_word in cls.DICTIONARY:
                word_phonemes = cls.DICTIONARY[clean_word]
            else:
                # Simple fallback - use each letter
                word_phonemes = []
                for char in clean_word:
                    if char in 'aeiou':
                        word_phonemes.append('ah')
                    else:
                        word_phonemes.append('t')
            
            phonemes.extend(word_phonemes)
            
            # Add pause for punctuation
            if punctuation:
                pause_dur = cls.PAUSE_MARKS.get(punctuation[-1], prosody.pause_duration)
                phonemes.append('pau')
                pauses.append(pause_dur)
            else:
                # Regular word pause
                phonemes.append('pau')
                pauses.append(prosody.pause_duration)
        
        return phonemes, pauses

# ==================== ENHANCED VOICE ENGINE ====================
class EnhancedVoiceEngine:
    """Professional voice synthesis engine"""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.phonemes = PhonemeDatabase()
        
    def generate_glottal_pulse(self, duration, f0, config):
        """Advanced glottal pulse with LF model"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples, False)
        
        # LF model parameters
        t0 = 1.0 / max(f0, 50)  # Avoid division by zero
        te = 0.8 * t0  # Open phase
        tp = 0.3 * t0   # Peak time
        ta = 0.05 * t0  # Return phase
        
        # Generate LF pulse
        pulse = np.zeros(n_samples)
        for i, time_point in enumerate(t):
            phase = (time_point % t0) / t0
            if phase < te/t0:
                # Opening phase
                pulse[i] = np.sin(np.pi * phase * t0 / tp)**2
            else:
                # Return phase
                pulse[i] = -np.exp(-(phase * t0 - te) / ta)
        
        # Add jitter
        jitter_mod = 1 + config.jitter * np.random.randn(n_samples)
        pulse = np.interp(np.arange(n_samples) * jitter_mod, np.arange(n_samples), pulse, left=0, right=0)
        pulse = np.nan_to_num(pulse)
        
        # Add shimmer
        shimmer_mod = 1 + config.shimmer * np.random.randn(n_samples)
        pulse = pulse * shimmer_mod
        
        # Add vibrato
        vibrato = 1 + config.vibrato_depth * np.sin(2 * np.pi * config.vibrato_rate * t)
        pulse = pulse * vibrato
        
        # Add tremolo
        tremolo = 1 + config.tremolo_depth * np.sin(2 * np.pi * config.tremolo_rate * t)
        pulse = pulse * tremolo
        
        return pulse
    
    def apply_vocal_tract(self, source, formants, bandwidths, config):
        """Apply sophisticated vocal tract model"""
        if len(source) == 0:
            return source
        
        result = source.copy()
        
        # Apply formants with resonance
        for f, bw in zip(formants[:3], bandwidths[:3]):  # Use first 3 for stability
            # Second-order resonator with higher Q
            r = np.exp(-np.pi * bw / self.sr)
            theta = 2 * np.pi * f / self.sr
            
            # Filter coefficients
            b = [1 - r**2]  # Normalize gain
            a = [1, -2 * r * np.cos(theta), r**2]
            
            # Apply filter
            try:
                result = signal.lfilter(b, a, result)
            except:
                pass
        
        # Apply spectral tilt
        if config.spectral_tilt != 0:
            # Pre-emphasis filter
            b = [1, -0.95]
            try:
                result = signal.lfilter(b, [1], result)
            except:
                pass
        
        return result * config.formant_strength
    
    def generate_phoneme(self, phoneme, duration, config):
        """Generate individual phoneme with context"""
        phoneme_data = self.phonemes.get_phoneme(phoneme)
        
        if phoneme == 'pau':
            return self.generate_silence(duration)
        
        if phoneme_data['type'] == 'vowel':
            return self.generate_vowel(phoneme_data, duration, config)
        else:
            return self.generate_consonant(phoneme_data, duration, config)
    
    def generate_vowel(self, data, duration, config):
        """Generate vowel with formants"""
        # Blend target formants with voice formants
        target_formants = data['formants']
        voice_formants = config.formants
        
        blended_formants = []
        for i in range(min(len(target_formants), len(voice_formants))):
            blended = (target_formants[i] * 0.7 + voice_formants[i] * 0.3)
            blended_formants.append(blended)
        
        # Generate source
        source = self.generate_glottal_pulse(duration, config.base_f0, config)
        if len(source) == 0:
            return np.array([])
        
        # Apply vocal tract
        voiced = self.apply_vocal_tract(source, blended_formants, 
                                        config.formant_bandwidths, config)
        
        # Apply natural envelope
        envelope = self.generate_amplitude_envelope(duration, 'vowel')
        if len(voiced) > len(envelope):
            voiced = voiced[:len(envelope)]
        elif len(voiced) < len(envelope):
            voiced = np.pad(voiced, (0, len(envelope) - len(voiced)))
        voiced = voiced * envelope
        
        return voiced
    
    def generate_consonant(self, data, duration, config):
        """Generate consonant with natural characteristics"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        cons_type = data['type']
        voiced = data.get('voiced', False)
        
        if cons_type in ['plosive', 'affricate']:
            # Burst + aspiration
            burst_len = min(int(0.01 * self.sr), n_samples)
            burst = np.random.randn(burst_len) * 0.5
            
            if voiced and burst_len > 0:
                # Add voicing during burst
                voice_source = self.generate_glottal_pulse(0.01, config.base_f0 * 0.8, config)
                if len(voice_source) >= burst_len:
                    burst = burst * 0.5 + voice_source[:burst_len] * 0.5
            
            aspiration_len = max(0, n_samples - burst_len)
            if aspiration_len > 0:
                aspiration = self.generate_aspiration(aspiration_len / self.sr, config)
                sound = np.concatenate([burst, aspiration])
            else:
                sound = burst
            
        elif cons_type == 'fricative':
            # Shaped noise
            noise = np.random.randn(n_samples)
            
            # Apply spectral shaping
            if data.get('noise_type') == 'highpass':
                b, a = signal.butter(4, 3000/(self.sr/2), 'high')
            elif data.get('noise_type') == 'lowpass':
                b, a = signal.butter(4, 1000/(self.sr/2), 'low')
            elif data.get('noise_type') == 'bandpass':
                b, a = signal.butter(4, [1000/(self.sr/2), 4000/(self.sr/2)], 'band')
            else:
                b, a = signal.butter(4, 2000/(self.sr/2), 'low')
            
            try:
                sound = signal.filtfilt(b, a, noise)
            except:
                sound = noise
            
            # Add voicing for voiced fricatives
            if voiced:
                voice_source = self.generate_glottal_pulse(duration, config.base_f0 * 0.7, config)
                if len(voice_source) == len(sound):
                    sound = sound * 0.4 + voice_source * 0.6
            
        elif cons_type == 'nasal':
            # Nasal sounds have formants with anti-resonances
            source = self.generate_glottal_pulse(duration, config.base_f0 * 0.8, config)
            if len(source) > 0:
                sound = self.apply_vocal_tract(source, data.get('formants', [250, 1200]), 
                                              [60, 100], config)
            else:
                sound = np.random.randn(n_samples) * 0.1
                
        else:  # liquids and glides
            source = self.generate_glottal_pulse(duration, config.base_f0, config)
            if len(source) > 0:
                sound = self.apply_vocal_tract(source, data.get('formants', [400, 1200]), 
                                              [80, 100], config)
            else:
                sound = np.random.randn(n_samples) * 0.1
        
        # Ensure correct length
        if len(sound) > n_samples:
            sound = sound[:n_samples]
        elif len(sound) < n_samples:
            sound = np.pad(sound, (0, n_samples - len(sound)))
        
        # Apply consonant envelope
        envelope = self.generate_amplitude_envelope(duration, cons_type)
        if len(sound) > len(envelope):
            sound = sound[:len(envelope)]
        elif len(sound) < len(envelope):
            sound = np.pad(sound, (0, len(envelope) - len(sound)))
        sound = sound * envelope
        
        return sound
    
    def generate_aspiration(self, duration, config):
        """Generate aspiration noise for plosives"""
        n_samples = int(self.sr * duration)
        if n_samples <= 0:
            return np.array([])
        
        # White noise
        noise = np.random.randn(n_samples)
        
        # Shape with low-pass filter
        b, a = signal.butter(2, 1500/(self.sr/2), 'low')
        try:
            aspiration = signal.filtfilt(b, a, noise)
        except:
            aspiration = noise
        
        # Apply amplitude envelope
        envelope = np.exp(-np.linspace(0, 5, n_samples))
        aspiration = aspiration * envelope
        
        return aspiration * 0.3
    
    def generate_amplitude_envelope(self, duration, sound_type):
        """Generate natural amplitude envelope"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples)
        
        if sound_type == 'vowel':
            # Smooth attack and release
            attack = int(0.02 * self.sr)
            release = int(0.05 * self.sr)
            envelope = np.ones(n_samples)
            
            if attack < n_samples:
                envelope[:attack] = np.linspace(0, 1, attack)
            if release < n_samples:
                envelope[-release:] = np.linspace(1, 0, release)
                
        elif sound_type in ['plosive', 'affricate']:
            # Quick attack, slower decay
            attack = int(0.005 * self.sr)
            decay = int(0.02 * self.sr)
            envelope = np.zeros(n_samples)
            
            if attack + decay < n_samples:
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[attack:attack+decay] = np.linspace(1, 0.2, decay)
                envelope[attack+decay:] = 0.2 * np.exp(-np.linspace(0, 2, n_samples - (attack+decay)))
                
        else:
            # Generic envelope
            envelope = np.ones(n_samples)
            fade = int(0.01 * self.sr)
            if fade * 2 < n_samples:
                envelope[:fade] = np.linspace(0, 1, fade)
                envelope[-fade:] = np.linspace(1, 0, fade)
        
        return envelope
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)
    
    def apply_reverb(self, audio, amount, decay):
        """Apply reverb effect"""
        if amount <= 0 or len(audio) == 0:
            return audio
        
        # Simple reverb
        reverb_len = int(0.1 * self.sr)
        impulse = np.zeros(reverb_len)
        impulse[0] = 1
        impulse[1:] = np.exp(-np.arange(1, reverb_len) / (decay * self.sr))
        
        reverb = np.convolve(audio, impulse * amount, mode='same')
        return audio + reverb
    
    def apply_echo(self, audio, delay, decay):
        """Apply echo effect"""
        if delay <= 0 or len(audio) == 0:
            return audio
        
        delay_samples = int(delay * self.sr)
        if delay_samples >= len(audio):
            return audio
        
        # Add delayed copies
        result = audio.copy()
        for i in range(1, 4):  # Multiple echos
            echo = np.roll(audio, i * delay_samples) * (decay ** i)
            echo[:i * delay_samples] = 0
            result += echo
        
        return result

# ==================== JARVIS AI VOICE SYSTEM ====================
class JarvisAIVoice:
    """Complete JARVIS AI voice system"""
    
    def __init__(self):
        self.engine = EnhancedVoiceEngine()
        self.text_processor = TextProcessor()
        self.config = VoiceConfig()
        self.prosody = ProsodyConfig()
        
        # Voice presets
        self.presets = self.load_presets()
        
    def load_presets(self):
        """Load professional voice presets"""
        return {
            # JARVIS variants
            'jarvis_classic': VoiceConfig(
                gender='male', age='adult', base_f0=110,
                formants=[550, 1150, 2550, 3250, 3900],
                breathiness=0.05, jitter=0.008, shimmer=0.08,
                vibrato_rate=5.2, vibrato_depth=0.018,
                emotion=Emotion.CALM, quality=VoiceQuality.WARM
            ),
            
            'jarvis_movie': VoiceConfig(
                gender='male', age='adult', base_f0=115,
                formants=[520, 1120, 2520, 3220, 3870],
                breathiness=0.03, jitter=0.005, shimmer=0.06,
                vibrato_rate=5.0, vibrato_depth=0.015,
                reverb_amount=0.05, echo_delay=0.02, echo_decay=0.3,
                emotion=Emotion.NEUTRAL, quality=VoiceQuality.STANDARD
            ),
            
            'jarvis_friendly': VoiceConfig(
                gender='male', age='young', base_f0=125,
                formants=[580, 1220, 2620, 3320, 3970],
                breathiness=0.12, jitter=0.015, shimmer=0.12,
                vibrato_rate=5.8, vibrato_depth=0.025,
                emotion=Emotion.HAPPY, emotion_intensity=0.4,
                quality=VoiceQuality.WARM
            ),
            
            # Female variants
            'friday': VoiceConfig(
                gender='female', age='young', base_f0=190,
                formants=[740, 1380, 2780, 3580, 4230],
                breathiness=0.1, jitter=0.012, shimmer=0.1,
                vibrato_rate=5.5, vibrato_depth=0.02,
                emotion=Emotion.CALM, quality=VoiceQuality.BRIGHT
            ),
            
            'vision': VoiceConfig(
                gender='female', age='adult', base_f0=180,
                formants=[700, 1320, 2720, 3520, 4170],
                breathiness=0.08, jitter=0.01, shimmer=0.09,
                vibrato_rate=5.3, vibrato_depth=0.018,
                emotion=Emotion.NEUTRAL, quality=VoiceQuality.WARM
            ),
            
            # Specialty voices - FIXED: Added modulation_freq
            'robotic': VoiceConfig(
                gender='male', age='adult', base_f0=130,
                effect_type='robotic', modulation_freq=30.0,
                reverb_amount=0.2, echo_delay=0.1, echo_decay=0.5,
                jitter=0.02, shimmer=0.15
            ),
            
            'alien': VoiceConfig(
                gender='male', age='young', base_f0=200,
                effect_type='alien', modulation_freq=20.0,
                formants=[800, 2000, 3500, 4500, 5500],
                vibrato_depth=0.04, vibrato_rate=8,
                reverb_amount=0.3
            ),
        }
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Voice set to: {name}")
        else:
            print(f"✗ Voice '{name}' not found")
    
    def set_emotion(self, emotion: Emotion, intensity: float = 0.5):
        """Set emotional state"""
        self.config.emotion = emotion
        self.config.emotion_intensity = intensity
        self.config.apply_emotion_to_params()
        print(f"✓ Emotion set to: {emotion.value} ({intensity})")
    
    def set_speech_rate(self, rate: float):
        """Set speech rate (0.5 to 2.0)"""
        self.prosody.speech_rate = max(0.5, min(2.0, rate))
        print(f"✓ Speech rate: {self.prosody.speech_rate}")
    
    def generate_speech(self, text, output_file=None):
        """Generate high-quality speech"""
        if not text:
            return None
        
        print(f"\n🎙️ Generating: '{text}'")
        
        # Process text
        phonemes, pauses = self.text_processor.process_text(text, self.prosody)
        print(f"📝 Phonemes: {len(phonemes)} sounds")
        
        if not phonemes:
            return None
        
        # Generate audio for each phoneme
        audio_segments = []
        
        for i, phoneme in enumerate(phonemes):
            # Get duration
            if phoneme == 'pau' and i < len(pauses):
                duration = pauses[i]
            else:
                phoneme_data = PhonemeDatabase.get_phoneme(phoneme)
                duration = phoneme_data.get('duration', 0.15) / self.prosody.speech_rate
            
            # Generate phoneme
            segment = self.engine.generate_phoneme(phoneme, duration, self.config)
            
            if segment is not None and len(segment) > 0:
                audio_segments.append(segment)
        
        if not audio_segments:
            return None
        
        # Combine all segments
        audio = np.concatenate(audio_segments)
        
        # Apply effects
        if self.config.reverb_amount > 0:
            audio = self.engine.apply_reverb(audio, self.config.reverb_amount, self.config.echo_decay)
        
        if self.config.echo_delay > 0:
            audio = self.engine.apply_echo(audio, self.config.echo_delay, self.config.echo_decay)
        
        # Apply special effects
        audio = self.apply_special_effects(audio)
        
        # Normalize
        if len(audio) > 0 and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # Save if requested
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"✓ Saved to: {output_file}")
        
        return audio
    
    def apply_special_effects(self, audio):
        """Apply special voice effects"""
        if self.config.effect_type == "robotic":
            # Ring modulation
            t = np.linspace(0, len(audio)/self.engine.sr, len(audio))
            mod_freq = self.config.modulation_freq if self.config.modulation_freq > 0 else 30
            mod = np.sin(2 * np.pi * mod_freq * t)
            audio = audio * (0.6 + 0.4 * mod)
            
        elif self.config.effect_type == "alien":
            # Frequency shifting + distortion
            audio = np.interp(np.arange(0, len(audio), 0.85), np.arange(len(audio)), audio, left=0, right=0)
            audio = np.nan_to_num(audio)
            audio = np.tanh(audio * 1.5)
        
        return audio
    
    def speak(self, text):
        """Speak text immediately"""
        audio = self.generate_speech(text)
        if audio is not None and len(audio) > 0:
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def speak_async(self, text):
        """Speak asynchronously (non-blocking)"""
        audio = self.generate_speech(text)
        if audio is not None:
            thread = threading.Thread(target=lambda: (sd.play(audio, self.engine.sr), sd.wait()))
            thread.daemon = True
            thread.start()
    
    def list_voices(self):
        """List all available voices"""
        print("\n🎭 Available Voices:")
        print("-" * 40)
        for name in self.presets.keys():
            config = self.presets[name]
            print(f"  • {name:15} ({config.gender}, {config.effect_type})")
    
    def demo(self):
        """Run a demonstration"""
        print("\n" + "="*60)
        print("🤖 JARVIS AI VOICE SYSTEM - REFINED EDITION")
        print("="*60)
        
        self.list_voices()
        
        # Test phrases
        phrases = [
            "Hello sir, JARVIS at your service",
            "Processing your request",
            "All systems are operational",
            "How may I assist you today?"
        ]
        
        # Test different voices
        voices_to_test = ['jarvis_classic', 'friday', 'robotic']
        
        for voice in voices_to_test:
            print(f"\n🎯 Testing voice: {voice}")
            self.set_voice(voice)
            for phrase in phrases[:2]:
                self.speak(phrase)
                time.sleep(0.5)
        
        # Test emotions
        print("\n🎭 Testing emotions with jarvis_classic")
        self.set_voice('jarvis_classic')
        
        emotions = [
            (Emotion.HAPPY, "I'm so happy to help you today!"),
            (Emotion.CALM, "Everything is under control, sir"),
            (Emotion.EXCITED, "This is absolutely fascinating!"),
        ]
        
        for emotion, phrase in emotions:
            self.set_emotion(emotion, 0.6)
            self.speak(phrase)
            time.sleep(0.5)
        
        print("\n✨ Demo complete!")

# ==================== INTERACTIVE CONSOLE ====================
class JarvisConsole:
    """Interactive console for JARVIS"""
    
    def __init__(self):
        self.jarvis = JarvisAIVoice()
        self.running = True
        
    def run(self):
        """Run interactive console"""
        print("\n" + "="*60)
        print("🎙️ JARVIS INTERACTIVE CONSOLE")
        print("="*60)
        print("\nCommands:")
        print("  /voices          - List all voices")
        print("  /voice <name>    - Set voice")
        print("  /emotion <name>  - Set emotion (happy/sad/angry/calm/excited)")
        print("  /rate <0.5-2.0>  - Set speech rate")
        print("  /save <file>     - Save last speech")
        print("  /demo            - Run demo")
        print("  /exit            - Exit")
        print("-"*60)
        print("Just type anything and JARVIS will speak!\n")
        
        last_audio = None
        
        while self.running:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    cmd = user_input[1:].split()
                    
                    if cmd[0] == 'exit':
                        self.running = False
                        print("Goodbye!")
                        
                    elif cmd[0] == 'voices':
                        self.jarvis.list_voices()
                        
                    elif cmd[0] == 'voice' and len(cmd) > 1:
                        self.jarvis.set_voice(cmd[1])
                        
                    elif cmd[0] == 'emotion' and len(cmd) > 1:
                        try:
                            emotion = Emotion(cmd[1].lower())
                            self.jarvis.set_emotion(emotion)
                        except:
                            print("Available emotions: happy, sad, angry, calm, excited")
                    
                    elif cmd[0] == 'rate' and len(cmd) > 1:
                        try:
                            rate = float(cmd[1])
                            self.jarvis.set_speech_rate(rate)
                        except:
                            print("Invalid rate")
                    
                    elif cmd[0] == 'demo':
                        self.jarvis.demo()
                    
                    elif cmd[0] == 'save' and len(cmd) > 1:
                        if last_audio is not None:
                            filename = cmd[1]
                            if not filename.endswith('.wav'):
                                filename += '.wav'
                            sf.write(filename, last_audio, self.jarvis.engine.sr)
                            print(f"✓ Saved to {filename}")
                        else:
                            print("No speech generated yet")
                    
                    else:
                        print("Unknown command")
                else:
                    # Speak the text
                    last_audio = self.jarvis.generate_speech(user_input)
                    if last_audio is not None:
                        sd.play(last_audio, self.jarvis.engine.sr)
                        sd.wait()
                        
            except KeyboardInterrupt:
                self.running = False
                print("\nGoodbye!")
            except Exception as e:
                print(f"Error: {e}")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JARVIS Refined Voice System")
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--voice', '-v', type=str, default='jarvis_classic', help='Voice preset')
    parser.add_argument('--text', '-t', type=str, help='Text to speak')
    parser.add_argument('--emotion', '-e', type=str, help='Emotion (happy/sad/angry/calm/excited)')
    parser.add_argument('--output', '-o', type=str, help='Output file')
    
    args = parser.parse_args()
    
    if args.demo:
        jarvis = JarvisAIVoice()
        jarvis.demo()
    elif args.interactive:
        console = JarvisConsole()
        console.run()
    elif args.text:
        jarvis = JarvisAIVoice()
        if args.voice:
            jarvis.set_voice(args.voice)
        if args.emotion:
            try:
                emotion = Emotion(args.emotion.lower())
                jarvis.set_emotion(emotion)
            except:
                print(f"Invalid emotion: {args.emotion}")
        
        if args.output:
            jarvis.generate_speech(args.text, args.output)
        else:
            jarvis.speak(args.text)
    else:
        # Default: interactive mode
        console = JarvisConsole()
        console.run()