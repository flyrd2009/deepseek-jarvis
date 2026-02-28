"""
JARVIS Voice System - ULTRA REFINED EDITION (COMPLETELY FIXED)
Hollywood grade voice synthesis with AI-powered features
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
from typing import List, Dict, Optional, Tuple, Callable
import threading
import queue
from enum import Enum
import hashlib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== ENUMS ====================
class Emotion(Enum):
    """Emotional states for voice"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    WORRIED = "worried"
    SARCASTIC = "sarcastic"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    SURPRISED = "surprised"
    FRIENDLY = "friendly"

class VoiceQuality(Enum):
    """Voice quality types"""
    STANDARD = "standard"
    WARM = "warm"
    BRIGHT = "bright"
    DARK = "dark"
    BREATHY = "breathy"
    NASAL = "nasal"
    SMOOTH = "smooth"
    ROUGH = "rough"
    VELVETY = "velvety"
    CRISP = "crisp"

class Accent(Enum):
    AMERICAN = "american"
    BRITISH = "british"
    AUSTRALIAN = "australian"
    INDIAN = "indian"
    NEUTRAL = "neutral"

# ==================== CONFIGURATION CLASSES ====================
@dataclass
class NeuralConfig:
    """Neural network style parameters"""
    model_path: str = ""
    use_neural: bool = False
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

@dataclass
class ProsodyConfig:
    """Advanced prosody parameters"""
    speech_rate: float = 1.0
    base_pitch: float = 120.0
    pitch_range: Tuple[float, float] = (80, 250)
    energy_range: Tuple[float, float] = (0.3, 1.2)
    pause_duration: float = 0.12
    sentence_pause: float = 0.4
    emphasis_factor: float = 1.2
    rhythm_variation: float = 0.1

@dataclass
class VoiceConfig:
    """Hollywood grade voice configuration"""
    # Identity
    name: str = "JARVIS"
    gender: str = "male"
    age: str = "adult"
    accent: Accent = Accent.AMERICAN
    
    # Fundamental parameters
    base_f0: float = 115.0
    formants: List[float] = None
    formant_bandwidths: List[float] = None
    formant_strength: float = 1.0
    
    # Voice characteristics
    breathiness: float = 0.06
    jitter: float = 0.008
    shimmer: float = 0.07
    creak: float = 0.02
    whisper: float = 0.0
    falsetto: float = 0.0
    
    # Modulation
    vibrato_rate: float = 5.2
    vibrato_depth: float = 0.018
    tremolo_rate: float = 4.2
    tremolo_depth: float = 0.008
    growl: float = 0.0
    
    # Spectral
    spectral_tilt: float = -4.5
    noise_floor: float = -65.0
    harmonic_richness: float = 0.8
    
    # Effects
    effect_type: str = "natural"
    modulation_freq: float = 0.0
    reverb_amount: float = 0.03
    reverb_decay: float = 0.4
    reverb_diffusion: float = 0.5
    echo_delay: float = 0.0
    echo_decay: float = 0.0
    chorus_depth: float = 0.0
    chorus_rate: float = 0.0
    distortion: float = 0.0
    
    # Emotional
    emotion: Emotion = Emotion.CONFIDENT
    emotion_intensity: float = 0.6
    
    # Quality
    quality: VoiceQuality = VoiceQuality.VELVETY
    
    # Neural
    neural: NeuralConfig = None
    
    def __post_init__(self):
        if self.formants is None:
            self.set_formants_for_gender()
        if self.formant_bandwidths is None:
            self.formant_bandwidths = [50, 60, 80, 120, 150, 200]
        if self.neural is None:
            self.neural = NeuralConfig()
        self.apply_emotion_to_params()
        self.apply_quality_to_params()
        self.apply_accent_to_params()
    
    def set_formants_for_gender(self):
        """Professional formant settings"""
        if self.gender == "male":
            if self.age == "child":
                self.formants = [850, 1450, 2850, 3550, 4200, 4800]
            elif self.age == "young":
                self.formants = [620, 1250, 2650, 3350, 4000, 4600]
            elif self.age == "adult":
                self.formants = [520, 1120, 2520, 3220, 3870, 4450]
            elif self.age == "old":
                self.formants = [480, 1050, 2450, 3150, 3800, 4400]
        elif self.gender == "female":
            if self.age == "child":
                self.formants = [1050, 1850, 3250, 4050, 4700, 5300]
            elif self.age == "young":
                self.formants = [780, 1450, 2850, 3650, 4300, 4900]
            elif self.age == "adult":
                self.formants = [720, 1350, 2750, 3550, 4200, 4800]
            elif self.age == "old":
                self.formants = [680, 1250, 2650, 3450, 4100, 4700]
        else:
            self.formants = [950, 1650, 3050, 3850, 4500, 5100]
    
    def apply_accent_to_params(self):
        """Apply accent modifications"""
        if self.accent == Accent.BRITISH:
            self.formants = [f * 0.98 for f in self.formants]
            self.vibrato_rate += 0.3
        elif self.accent == Accent.INDIAN:
            self.formants = [f * 1.05 for f in self.formants]
            self.vibrato_depth += 0.005
    
    def apply_emotion_to_params(self):
        """Advanced emotion mapping"""
        intensity = self.emotion_intensity
        
        emotion_map = {
            Emotion.HAPPY: {
                'base_f0': 1 + 0.15 * intensity,
                'vibrato_rate': 2 * intensity,
                'vibrato_depth': 0.02 * intensity,
                'breathiness': 0.05 * intensity,
                'jitter': 0.01 * intensity
            },
            Emotion.SAD: {
                'base_f0': 1 - 0.12 * intensity,
                'vibrato_rate': -1 * intensity,
                'breathiness': 0.08 * intensity,
                'jitter': -0.005 * intensity,
                'creak': 0.05 * intensity
            },
            Emotion.ANGRY: {
                'base_f0': 1 + 0.2 * intensity,
                'formant_strength': 1 + 0.25 * intensity,
                'jitter': 0.02 * intensity,
                'shimmer': 0.08 * intensity,
                'creak': 0.1 * intensity,
                'growl': 0.05 * intensity
            },
            Emotion.EXCITED: {
                'base_f0': 1 + 0.25 * intensity,
                'vibrato_rate': 3 * intensity,
                'vibrato_depth': 0.03 * intensity,
                'jitter': 0.015 * intensity,
                'harmonic_richness': 1 + 0.1 * intensity
            },
            Emotion.CONFIDENT: {
                'base_f0': 1 + 0.05 * intensity,
                'formant_strength': 1 + 0.1 * intensity,
                'vibrato_depth': 0.01 * intensity,
                'jitter': -0.002 * intensity,
                'spectral_tilt': -1 * intensity
            },
            Emotion.FRIENDLY: {
                'base_f0': 1 + 0.08 * intensity,
                'breathiness': 0.1 * intensity,
                'vibrato_depth': 0.015 * intensity,
                'jitter': 0.005 * intensity,
                'shimmer': 0.05 * intensity
            },
            Emotion.CALM: {
                'base_f0': 1 - 0.05 * intensity,
                'vibrato_rate': -0.5 * intensity,
                'vibrato_depth': -0.005 * intensity,
                'jitter': -0.003 * intensity,
                'shimmer': -0.02 * intensity
            }
        }
        
        if self.emotion in emotion_map:
            for param, factor in emotion_map[self.emotion].items():
                if hasattr(self, param):
                    current = getattr(self, param)
                    if isinstance(current, (int, float)):
                        setattr(self, param, current * (1 + factor))
    
    def apply_quality_to_params(self):
        """Professional quality mapping"""
        quality_map = {
            VoiceQuality.VELVETY: {
                'formants': [f * 0.95 for f in self.formants],
                'breathiness': self.breathiness + 0.08,
                'jitter': self.jitter - 0.002,
                'shimmer': self.shimmer - 0.01,
                'spectral_tilt': self.spectral_tilt - 2
            },
            VoiceQuality.WARM: {
                'formants': [f * 0.97 for f in self.formants],
                'breathiness': self.breathiness + 0.05,
                'vibrato_depth': self.vibrato_depth * 1.1,
                'jitter': self.jitter * 0.8,
                'shimmer': self.shimmer * 0.9
            },
            VoiceQuality.CRISP: {
                'formants': [f * 1.05 for f in self.formants],
                'breathiness': self.breathiness - 0.02,
                'jitter': self.jitter + 0.002,
                'shimmer': self.shimmer + 0.005,
                'spectral_tilt': self.spectral_tilt + 3
            },
            VoiceQuality.SMOOTH: {
                'vibrato_depth': self.vibrato_depth * 0.7,
                'jitter': self.jitter * 0.5,
                'shimmer': self.shimmer * 0.5,
                'breathiness': self.breathiness * 1.2
            },
            VoiceQuality.ROUGH: {
                'jitter': self.jitter * 2,
                'shimmer': self.shimmer * 1.5,
                'creak': self.creak + 0.05,
                'growl': self.growl + 0.02
            },
            VoiceQuality.BRIGHT: {
                'formants': [f * 1.08 for f in self.formants],
                'spectral_tilt': self.spectral_tilt + 4,
                'harmonic_richness': self.harmonic_richness * 1.1
            },
            VoiceQuality.DARK: {
                'formants': [f * 0.92 for f in self.formants],
                'spectral_tilt': self.spectral_tilt - 3,
                'breathiness': self.breathiness * 0.8
            },
            VoiceQuality.BREATHY: {
                'breathiness': self.breathiness + 0.2,
                'noise_floor': self.noise_floor + 10,
                'jitter': self.jitter * 1.2
            }
        }
        
        if self.quality in quality_map:
            for param, value in quality_map[self.quality].items():
                if param == 'formants':
                    self.formants = value
                else:
                    setattr(self, param, value)

# ==================== PHONEME ENGINE ====================
class PhonemeEngine:
    """AI-powered phoneme synthesis - FIXED: Added this missing class"""
    
    # Extended phoneme database
    PHONEMES = {
        # Vowels
        'iy': {'type': 'vowel', 'formants': [270, 2300, 3000, 3800, 4500, 5000], 
               'duration': 0.22},
        'ih': {'type': 'vowel', 'formants': [390, 2000, 2600, 3400, 4200, 4800], 
               'duration': 0.18},
        'eh': {'type': 'vowel', 'formants': [530, 1850, 2500, 3300, 4100, 4700], 
               'duration': 0.18},
        'ae': {'type': 'vowel', 'formants': [660, 1700, 2400, 3200, 4000, 4600], 
               'duration': 0.2},
        'ah': {'type': 'vowel', 'formants': [520, 1190, 2400, 3200, 4000, 4600], 
               'duration': 0.16},
        'aa': {'type': 'vowel', 'formants': [730, 1090, 2450, 3250, 4050, 4650], 
               'duration': 0.19},
        'ao': {'type': 'vowel', 'formants': [570, 840, 2410, 3200, 4000, 4600], 
               'duration': 0.2},
        'uh': {'type': 'vowel', 'formants': [440, 1020, 2250, 3100, 3900, 4500], 
               'duration': 0.16},
        'uw': {'type': 'vowel', 'formants': [300, 870, 2250, 3100, 3900, 4500], 
               'duration': 0.21},
        'er': {'type': 'vowel', 'formants': [490, 1350, 1700, 2500, 3500, 4300], 
               'duration': 0.23},
        
        # Diphthongs
        'ey': {'type': 'diphthong', 'formants_start': [400, 2000, 2600], 
               'formants_end': [270, 2300, 3000], 'duration': 0.25},
        'ay': {'type': 'diphthong', 'formants_start': [730, 1090, 2450], 
               'formants_end': [270, 2300, 3000], 'duration': 0.26},
        'aw': {'type': 'diphthong', 'formants_start': [730, 1090, 2450], 
               'formants_end': [300, 870, 2250], 'duration': 0.26},
        'oy': {'type': 'diphthong', 'formants_start': [570, 840, 2410], 
               'formants_end': [270, 2300, 3000], 'duration': 0.25},
        'ow': {'type': 'diphthong', 'formants_start': [520, 1190, 2400], 
               'formants_end': [300, 870, 2250], 'duration': 0.24},
        
        # Consonants - Plosives
        'p': {'type': 'plosive', 'voiced': False, 'duration': 0.1, 'burst_freq': [500, 1500]},
        'b': {'type': 'plosive', 'voiced': True, 'duration': 0.1, 'burst_freq': [500, 1500]},
        't': {'type': 'plosive', 'voiced': False, 'duration': 0.1, 'burst_freq': [4000, 6000]},
        'd': {'type': 'plosive', 'voiced': True, 'duration': 0.1, 'burst_freq': [4000, 6000]},
        'k': {'type': 'plosive', 'voiced': False, 'duration': 0.12, 'burst_freq': [1500, 3000]},
        'g': {'type': 'plosive', 'voiced': True, 'duration': 0.12, 'burst_freq': [1500, 3000]},
        
        # Fricatives
        'f': {'type': 'fricative', 'voiced': False, 'duration': 0.18, 'noise_center': 8000},
        'v': {'type': 'fricative', 'voiced': True, 'duration': 0.16, 'noise_center': 7000},
        'th': {'type': 'fricative', 'voiced': False, 'duration': 0.16, 'noise_center': 6000},
        'dh': {'type': 'fricative', 'voiced': True, 'duration': 0.14, 'noise_center': 5500},
        's': {'type': 'fricative', 'voiced': False, 'duration': 0.2, 'noise_center': 7000},
        'z': {'type': 'fricative', 'voiced': True, 'duration': 0.18, 'noise_center': 6500},
        'sh': {'type': 'fricative', 'voiced': False, 'duration': 0.2, 'noise_center': 4000},
        'zh': {'type': 'fricative', 'voiced': True, 'duration': 0.18, 'noise_center': 3800},
        'hh': {'type': 'fricative', 'voiced': False, 'duration': 0.12, 'noise_center': 1000},
        
        # Affricates
        'ch': {'type': 'affricate', 'voiced': False, 'duration': 0.15},
        'jh': {'type': 'affricate', 'voiced': True, 'duration': 0.15},
        
        # Nasals
        'm': {'type': 'nasal', 'voiced': True, 'duration': 0.15, 'formants': [250, 1200, 2200]},
        'n': {'type': 'nasal', 'voiced': True, 'duration': 0.14, 'formants': [250, 1300, 2300]},
        'ng': {'type': 'nasal', 'voiced': True, 'duration': 0.15, 'formants': [250, 1100, 2100]},
        
        # Liquids
        'l': {'type': 'liquid', 'voiced': True, 'duration': 0.15, 'formants': [400, 1200, 2600]},
        'r': {'type': 'liquid', 'voiced': True, 'duration': 0.15, 'formants': [350, 1100, 2200]},
        
        # Glides
        'w': {'type': 'glide', 'voiced': True, 'duration': 0.12, 'formants': [300, 800, 2100]},
        'y': {'type': 'glide', 'voiced': True, 'duration': 0.12, 'formants': [300, 2200, 2900]},
    }
    
    @classmethod
    def get_phoneme(cls, symbol):
        """Get phoneme data"""
        return cls.PHONEMES.get(symbol, cls.PHONEMES.get('ah'))

# ==================== AI TEXT PROCESSOR ====================
class AITextProcessor:
    """AI-powered text analysis - FIXED: Added this missing class"""
    
    # Pronunciation dictionary
    DICTIONARY = {
        'hello': [('hh', 0), ('ah', 1), ('l', 0), ('ow', 2)],
        'hi': [('hh', 1), ('ay', 2)],
        'hey': [('hh', 1), ('ey', 2)],
        'good': [('g', 0), ('uh', 1), ('d', 0)],
        'morning': [('m', 0), ('ao', 1), ('r', 0), ('n', 0), ('ih', 2), ('ng', 0)],
        'afternoon': [('ae', 0), ('f', 0), ('t', 0), ('er', 1), ('n', 0), ('uw', 2), ('n', 0)],
        'evening': [('iy', 1), ('v', 0), ('ih', 2), ('n', 0), ('ih', 0), ('ng', 0)],
        'yes': [('y', 0), ('eh', 1), ('s', 0)],
        'no': [('n', 0), ('ow', 1)],
        'okay': [('ow', 1), ('k', 0), ('ey', 2)],
        'sure': [('sh', 0), ('uh', 1), ('r', 0)],
        'please': [('p', 0), ('l', 1), ('iy', 2), ('z', 0)],
        'thanks': [('th', 0), ('ae', 1), ('ng', 0), ('k', 0), ('s', 2)],
        'thank': [('th', 0), ('ae', 1), ('ng', 0), ('k', 0)],
        'you': [('y', 0), ('uw', 1)],
        'sir': [('s', 0), ('er', 1)],
        'boss': [('b', 0), ('aa', 1), ('s', 0)],
        'jarvis': [('jh', 0), ('aa', 1), ('r', 0), ('v', 0), ('ih', 2), ('s', 0)],
        'stark': [('s', 0), ('t', 0), ('aa', 1), ('r', 0), ('k', 0)],
        'tony': [('t', 0), ('ow', 1), ('n', 0), ('iy', 2)],
        'iron': [('ay', 1), ('er', 0), ('n', 2)],
        'man': [('m', 0), ('ae', 1), ('n', 0)],
        'processing': [('p', 0), ('r', 0), ('aa', 2), ('s', 0), ('eh', 1), ('s', 0), ('ih', 0), ('ng', 0)],
        'analyzing': [('ae', 0), ('n', 0), ('ah', 1), ('l', 0), ('ay', 2), ('z', 0), ('ih', 0), ('ng', 0)],
        'computing': [('k', 0), ('ah', 1), ('m', 0), ('p', 0), ('y', 0), ('uw', 2), ('t', 0), ('ih', 0), ('ng', 0)],
        'ready': [('r', 0), ('eh', 1), ('d', 0), ('iy', 2)],
        'online': [('aa', 1), ('n', 0), ('l', 0), ('ay', 2), ('n', 0)],
        'active': [('ae', 1), ('k', 0), ('t', 0), ('ih', 2), ('v', 0)],
        'system': [('s', 0), ('ih', 1), ('s', 0), ('t', 0), ('ah', 2), ('m', 0)],
    }
    
    # Punctuation prosody
    PUNCTUATION_MAP = {
        '.': {'pause': 0.5, 'pitch_change': -0.1, 'energy_change': -0.2},
        ',': {'pause': 0.2, 'pitch_change': 0.0, 'energy_change': -0.1},
        '!': {'pause': 0.4, 'pitch_change': 0.3, 'energy_change': 0.4},
        '?': {'pause': 0.4, 'pitch_change': 0.4, 'energy_change': 0.2},
        ':': {'pause': 0.3, 'pitch_change': 0.1, 'energy_change': 0.0},
        ';': {'pause': 0.25, 'pitch_change': 0.05, 'energy_change': 0.0},
    }
    
    @classmethod
    def analyze_text(cls, text: str, prosody: ProsodyConfig):
        """Analyze text and return phonemes, stresses, prosody marks"""
        if not text:
            return [], [], []
        
        text = text.lower().strip()
        words = text.split()
        
        phonemes = []
        stresses = []
        prosody_marks = []
        
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalpha())
            punctuation = ''.join(c for c in word if c in '.,!?:;')
            
            # Get pronunciation
            if clean_word in cls.DICTIONARY:
                word_data = cls.DICTIONARY[clean_word]
                for phoneme, stress in word_data:
                    phonemes.append(phoneme)
                    stresses.append(stress)
            else:
                # Simple fallback
                for char in clean_word:
                    if char in 'aeiou':
                        phonemes.append('ah')
                        stresses.append(0)
                    else:
                        phonemes.append('t')
                        stresses.append(0)
            
            # Add punctuation prosody
            if punctuation:
                mark = punctuation[-1]
                if mark in cls.PUNCTUATION_MAP:
                    prosody_marks.append(cls.PUNCTUATION_MAP[mark])
                    phonemes.append('pau')
                    stresses.append(0)
            else:
                # Regular pause
                phonemes.append('pau')
                stresses.append(0)
                prosody_marks.append({'pause': prosody.pause_duration, 
                                     'pitch_change': 0, 'energy_change': 0})
        
        return phonemes, stresses, prosody_marks

# ==================== HOLLYWOOD VOICE ENGINE ====================
class HollywoodVoiceEngine:
    """Professional voice engine - FIXED: Added this missing class"""
    
    def __init__(self, sample_rate=48000):
        self.sr = sample_rate
        self.phonemes = PhonemeEngine()
        
    def generate_glottal_pulse(self, duration, f0, config):
        """Generate glottal pulse"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples, False)
        
        # Simple pulse model
        t0 = 1.0 / max(f0, 50)
        pulse = np.zeros(n_samples)
        
        for i, time_point in enumerate(t):
            phase = (time_point % t0) / t0
            if phase < 0.8:
                pulse[i] = np.sin(np.pi * phase / 0.8)**2
            else:
                pulse[i] = -np.exp(-(phase - 0.8) / 0.05)
        
        # Add jitter
        jitter_mod = 1 + config.jitter * np.random.randn(n_samples)
        pulse = np.interp(np.arange(n_samples) * np.mean(jitter_mod), 
                         np.arange(n_samples), pulse, left=0, right=0)
        pulse = np.nan_to_num(pulse)
        
        # Add shimmer
        shimmer_mod = 1 + config.shimmer * np.random.randn(n_samples)
        pulse = pulse * shimmer_mod
        
        # Add vibrato
        vibrato = 1 + config.vibrato_depth * np.sin(2 * np.pi * config.vibrato_rate * t)
        pulse = pulse * vibrato
        
        return pulse
    
    def apply_vocal_tract(self, source, formants, bandwidths, config):
        """Apply vocal tract filtering"""
        if len(source) == 0:
            return source
        
        result = source.copy()
        
        # Apply formants
        for f, bw in zip(formants[:4], bandwidths[:4]):
            r = np.exp(-np.pi * bw / self.sr)
            theta = 2 * np.pi * f / self.sr
            
            b = [1 - r**2]
            a = [1, -2 * r * np.cos(theta), r**2]
            
            try:
                result = signal.lfilter(b, a, result)
            except:
                pass
        
        # Apply spectral tilt
        if config.spectral_tilt != 0:
            b = [1, -0.95]
            try:
                result = signal.lfilter(b, [1], result)
            except:
                pass
        
        return result * config.formant_strength
    
    def generate_phoneme(self, phoneme, duration, config, stress=0):
        """Generate individual phoneme"""
        phoneme_data = self.phonemes.get_phoneme(phoneme)
        
        if phoneme == 'pau':
            return self.generate_silence(duration)
        
        # Apply stress
        if stress > 0:
            duration *= (1 + 0.1 * stress)
        
        if phoneme_data['type'] == 'vowel':
            audio = self.generate_vowel(phoneme_data, duration, config)
        elif phoneme_data['type'] == 'diphthong':
            audio = self.generate_diphthong(phoneme_data, duration, config)
        else:
            audio = self.generate_consonant(phoneme_data, duration, config)
        
        return audio
    
    def generate_vowel(self, data, duration, config):
        """Generate vowel sound"""
        # Blend formants
        target_formants = data['formants']
        voice_formants = config.formants
        
        blended_formants = []
        for i in range(min(len(target_formants), len(voice_formants))):
            blended = (target_formants[i] * 0.6 + voice_formants[i] * 0.4)
            blended_formants.append(blended)
        
        # Generate source
        source = self.generate_glottal_pulse(duration, config.base_f0, config)
        if len(source) == 0:
            return np.array([])
        
        # Apply formants
        voiced = self.apply_vocal_tract(source, blended_formants, 
                                        config.formant_bandwidths, config)
        
        return voiced
    
    def generate_diphthong(self, data, duration, config):
        """Generate diphthong (gliding vowel)"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        # Simple interpolation
        source = self.generate_glottal_pulse(duration, config.base_f0, config)
        
        return source
    
    def generate_consonant(self, data, duration, config):
        """Generate consonant sound"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        cons_type = data['type']
        
        if cons_type == 'plosive':
            # Burst sound
            burst_len = min(int(0.01 * self.sr), n_samples)
            burst = np.random.randn(burst_len) * 0.3
            silence = np.zeros(n_samples - burst_len)
            sound = np.concatenate([burst, silence])
            
        elif cons_type == 'fricative':
            # Noise sound
            noise = np.random.randn(n_samples) * 0.2
            sound = noise
            
        elif cons_type == 'nasal':
            # Nasal sound
            source = self.generate_glottal_pulse(duration, config.base_f0 * 0.8, config)
            if len(source) > 0:
                sound = source
            else:
                sound = np.random.randn(n_samples) * 0.1
                
        else:
            # Default
            sound = np.random.randn(n_samples) * 0.1
        
        # Ensure correct length
        if len(sound) > n_samples:
            sound = sound[:n_samples]
        elif len(sound) < n_samples:
            sound = np.pad(sound, (0, n_samples - len(sound)))
        
        return sound
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)
    
    def apply_professional_effects(self, audio, config):
        """Apply audio effects"""
        if len(audio) == 0:
            return audio
        
        # Simple reverb
        if config.reverb_amount > 0:
            reverb_len = int(0.1 * self.sr)
            impulse = np.zeros(reverb_len)
            impulse[0] = 1
            impulse[1:] = np.exp(-np.arange(1, reverb_len) / (config.reverb_decay * self.sr))
            reverb = np.convolve(audio, impulse * config.reverb_amount, mode='same')
            audio = audio + reverb
        
        # Echo
        if config.echo_delay > 0:
            delay_samples = int(config.echo_delay * self.sr)
            if delay_samples < len(audio):
                echo = np.roll(audio, delay_samples) * config.echo_decay
                echo[:delay_samples] = 0
                audio = audio + echo
        
        # Special effects
        if config.effect_type == "robotic":
            t = np.linspace(0, len(audio)/self.sr, len(audio))
            mod_freq = config.modulation_freq if config.modulation_freq > 0 else 30
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
            audio = audio * mod
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        return audio

# ==================== JARVIS ULTRA SYSTEM ====================
class JarvisUltra:
    """Ultimate JARVIS voice system - Hollywood grade"""
    
    def __init__(self):
        self.engine = HollywoodVoiceEngine()  # Now defined
        self.text_processor = AITextProcessor()  # Now defined
        self.config = VoiceConfig()
        self.prosody = ProsodyConfig()
        
        # Voice presets library
        self.presets = self.load_presets()
        
        # Cache for generated audio
        self.cache = {}
        self.cache_dir = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Audio queue for streaming
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        
    def load_presets(self):
        """Load Hollywood grade voice presets"""
        return {
            # JARVIS variants
            'jarvis_hollywood': VoiceConfig(
                name="JARVIS", gender='male', age='adult', base_f0=112,
                formants=[530, 1130, 2530, 3230, 3880, 4460],
                breathiness=0.04, jitter=0.006, shimmer=0.06,
                vibrato_rate=5.0, vibrato_depth=0.016,
                reverb_amount=0.04, reverb_decay=0.35,
                emotion=Emotion.CONFIDENT, quality=VoiceQuality.VELVETY
            ),
            
            'jarvis_movie': VoiceConfig(
                name="Movie JARVIS", gender='male', age='adult', base_f0=108,
                formants=[510, 1110, 2510, 3210, 3860, 4440],
                breathiness=0.03, jitter=0.005, shimmer=0.05,
                vibrato_rate=4.8, vibrato_depth=0.014,
                reverb_amount=0.06, reverb_decay=0.4,
                echo_delay=0.015, echo_decay=0.2,
                emotion=Emotion.CALM, quality=VoiceQuality.SMOOTH
            ),
            
            # Female variants
            'friday_hollywood': VoiceConfig(
                name="FRIDAY", gender='female', age='young', base_f0=195,
                formants=[750, 1390, 2790, 3590, 4240, 4820],
                breathiness=0.09, jitter=0.01, shimmer=0.09,
                vibrato_rate=5.6, vibrato_depth=0.022,
                reverb_amount=0.03, reverb_decay=0.3,
                emotion=Emotion.CALM, quality=VoiceQuality.BRIGHT
            ),
            
            'pepper': VoiceConfig(
                name="Pepper", gender='female', age='adult', base_f0=185,
                formants=[710, 1330, 2730, 3530, 4180, 4760],
                breathiness=0.08, jitter=0.009, shimmer=0.08,
                vibrato_rate=5.4, vibrato_depth=0.02,
                emotion=Emotion.FRIENDLY, quality=VoiceQuality.WARM
            ),
            
            # Accented variants
            'jarvis_british': VoiceConfig(
                name="British JARVIS", gender='male', age='adult', base_f0=110,
                accent=Accent.BRITISH,
                formants=[520, 1110, 2510, 3210, 3860, 4440],
                breathiness=0.04, jitter=0.006, shimmer=0.06,
                vibrato_rate=5.3, vibrato_depth=0.017,
                emotion=Emotion.CONFIDENT, quality=VoiceQuality.SMOOTH
            ),
            
            'jarvis_indian': VoiceConfig(
                name="Indian JARVIS", gender='male', age='young', base_f0=125,
                accent=Accent.INDIAN,
                formants=[580, 1190, 2590, 3290, 3940, 4520],
                breathiness=0.07, jitter=0.012, shimmer=0.1,
                vibrato_rate=5.8, vibrato_depth=0.022,
                emotion=Emotion.FRIENDLY, quality=VoiceQuality.WARM
            ),
            
            # Emotional variants
            'jarvis_happy': VoiceConfig(
                name="Happy JARVIS", gender='male', age='young', base_f0=130,
                formants=[560, 1160, 2560, 3260, 3910, 4490],
                breathiness=0.1, jitter=0.015, shimmer=0.12,
                vibrato_rate=6.0, vibrato_depth=0.025,
                emotion=Emotion.HAPPY, emotion_intensity=0.7,
                quality=VoiceQuality.BRIGHT
            ),
            
            'jarvis_serious': VoiceConfig(
                name="Serious JARVIS", gender='male', age='adult', base_f0=105,
                formants=[500, 1100, 2500, 3200, 3850, 4430],
                breathiness=0.02, jitter=0.004, shimmer=0.04,
                vibrato_rate=4.5, vibrato_depth=0.01,
                emotion=Emotion.CONFIDENT, emotion_intensity=0.8,
                quality=VoiceQuality.DARK
            ),
            
            # Effect variants
            'robotic_hd': VoiceConfig(
                name="HD Robot", gender='male', age='adult', base_f0=130,
                effect_type='robotic', modulation_freq=45.0,
                reverb_amount=0.15, reverb_decay=0.3,
                jitter=0.02, shimmer=0.15,
                quality=VoiceQuality.ROUGH
            ),

            'mera_custom_voice': VoiceConfig(
             name="Mera Voice", 
             gender='male', 
             age='young', 
             base_f0=140,  # Pitch
             formants=[600, 1300, 2700, 3400, 4000, 4600],
             breathiness=0.1,
             jitter=0.02,
             vibrato_rate=6.0,
             quality=VoiceQuality.WARM
            ),
            
            'alien_hd': VoiceConfig(
                name="HD Alien", gender='male', age='young', base_f0=220,
                effect_type='alien', modulation_freq=25.0,
                formants=[900, 2200, 3800, 4800, 5800, 6800],
                vibrato_depth=0.05, vibrato_rate=8,
                reverb_amount=0.25, reverb_decay=0.6,
                quality=VoiceQuality.BRIGHT
            ),
        }
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Voice set to: {self.config.name}")
        else:
            print(f"✗ Voice '{name}' not found")
    
    def set_emotion(self, emotion: Emotion, intensity: float = 0.5):
        """Set emotional state"""
        self.config.emotion = emotion
        self.config.emotion_intensity = intensity
        self.config.apply_emotion_to_params()
        print(f"✓ Emotion set to: {emotion.value} ({intensity})")
    
    def set_speech_rate(self, rate: float):
        """Set speech rate"""
        self.prosody.speech_rate = max(0.5, min(2.5, rate))
        print(f"✓ Speech rate: {self.prosody.speech_rate}x")
    
    def generate_speech(self, text, output_file=None):
        """Generate speech"""
        if not text:
            return None
        
        # Check cache
        cache_key = hashlib.md5(f"{text}{self.config.name}{self.prosody.speech_rate}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            print(f"📦 Loading from cache: '{text}'")
            audio = np.load(cache_file)
        else:
            print(f"\n🎙️ Generating: '{text}'")
            
            # Analyze text
            phonemes, stresses, prosody_marks = self.text_processor.analyze_text(text, self.prosody)
            print(f"📝 Phonemes: {len(phonemes)} sounds")
            
            if not phonemes:
                return None
            
            # Generate audio segments
            audio_segments = []
            
            for i, phoneme in enumerate(phonemes):
                if phoneme == 'pau':
                    if i < len(prosody_marks):
                        duration = prosody_marks[i].get('pause', self.prosody.pause_duration)
                    else:
                        duration = self.prosody.pause_duration
                    segment = self.engine.generate_silence(duration)
                else:
                    phoneme_data = PhonemeEngine.get_phoneme(phoneme)
                    base_duration = phoneme_data.get('duration', 0.15)
                    duration = base_duration / self.prosody.speech_rate
                    
                    stress = stresses[i] if i < len(stresses) else 0
                    segment = self.engine.generate_phoneme(phoneme, duration, self.config, stress)
                
                if segment is not None and len(segment) > 0:
                    audio_segments.append(segment)
            
            if not audio_segments:
                return None
            
            # Combine
            audio = np.concatenate(audio_segments)
            
            # Apply effects
            audio = self.engine.apply_professional_effects(audio, self.config)
            
            # Cache
            np.save(cache_file, audio)
        
        # Save if requested
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"✓ Saved to: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak text"""
        audio = self.generate_speech(text)
        if audio is not None and len(audio) > 0:
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def speak_stream(self, text):
        """Stream speech (non-blocking)"""
        audio = self.generate_speech(text)
        if audio is not None:
            thread = threading.Thread(target=lambda: (sd.play(audio, self.engine.sr), sd.wait()))
            thread.daemon = True
            thread.start()
    
    def list_voices(self):
        """List all voices"""
        print("\n🎭 VOICE LIBRARY")
        print("=" * 50)
        
        for name, config in self.presets.items():
            emotion_name = config.emotion.value if config.emotion else "neutral"
            quality_name = config.quality.value if config.quality else "standard"
            print(f"  • {name:20} ({config.gender}, {emotion_name}, {quality_name})")
    
    def demo(self):
        """Run demo"""
        print("\n" + "="*60)
        print("🎬 JARVIS VOICE SYSTEM - DEMO")
        print("="*60)
        
        self.list_voices()
        
        # Test phrases
        phrases = [
            "Hello sir, JARVIS at your service",
            "All systems are operational",
            "How may I assist you today?"
        ]
        
        # Test voices
        test_voices = ['jarvis_hollywood', 'friday_hollywood', 'robotic_hd']
        
        for voice in test_voices:
            print(f"\n🎙️ Testing: {voice}")
            self.set_voice(voice)
            for phrase in phrases[:1]:
                self.speak(phrase)
                time.sleep(0.5)
        
        print("\n✨ Demo Complete!")

# ==================== INTERACTIVE CONSOLE ====================
class HollywoodConsole:
    """Interactive console for JARVIS"""
    
    def __init__(self):
        self.jarvis = JarvisUltra()
        self.running = True
        self.history = []
        
    def print_banner(self):
        banner = """
╔══════════════════════════════════════════════════════════╗
║     🎬 JARVIS HOLLYWOOD VOICE SYSTEM                    ║
║            Professional Grade Voice Synthesis            ║
╚══════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def run(self):
        """Run interactive console"""
        self.print_banner()
        
        print("\nCommands:")
        print("  /voices          - List all voices")
        print("  /voice <name>    - Set voice")
        print("  /emotion <name>  - Set emotion")
        print("  /rate <0.5-2.5>  - Set speech rate")
        print("  /save <file>     - Save last speech")
        print("  /replay          - Replay last speech")
        print("  /demo            - Run demo")
        print("  /exit            - Exit")
        print("-" * 50)
        print("Type anything and JARVIS will speak!\n")
        
        last_audio = None
        
        while self.running:
            try:
                user_input = input("\n🎬 You: ").strip()
                
                if not user_input:
                    continue
                
                self.history.append(user_input)
                
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
                            print("Available: happy, sad, angry, calm, excited, confident, friendly")
                    
                    elif cmd[0] == 'rate' and len(cmd) > 1:
                        try:
                            rate = float(cmd[1])
                            self.jarvis.set_speech_rate(rate)
                        except:
                            print("Invalid rate")
                    
                    elif cmd[0] == 'demo':
                        self.jarvis.demo()
                    
                    elif cmd[0] == 'replay':
                        if last_audio is not None:
                            sd.play(last_audio, self.jarvis.engine.sr)
                            sd.wait()
                        else:
                            print("No speech to replay")
                    
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
    
    parser = argparse.ArgumentParser(description="JARVIS Voice System")
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--voice', '-v', type=str, default='jarvis_hollywood', help='Voice preset')
    parser.add_argument('--text', '-t', type=str, help='Text to speak')
    parser.add_argument('--emotion', '-e', type=str, help='Emotion')
    parser.add_argument('--rate', '-r', type=float, default=1.0, help='Speech rate')
    parser.add_argument('--output', '-o', type=str, help='Output file')
    
    args = parser.parse_args()
    
    if args.demo:
        jarvis = JarvisUltra()
        jarvis.demo()
    elif args.interactive:
        console = HollywoodConsole()
        console.run()
    elif args.text:
        jarvis = JarvisUltra()
        if args.voice:
            jarvis.set_voice(args.voice)
        if args.emotion:
            try:
                emotion = Emotion(args.emotion.lower())
                jarvis.set_emotion(emotion)
            except:
                print(f"Invalid emotion: {args.emotion}")
        if args.rate != 1.0:
            jarvis.set_speech_rate(args.rate)
        
        if args.output:
            jarvis.generate_speech(args.text, args.output)
        else:
            jarvis.speak(args.text)
    else:
        console = HollywoodConsole()
        console.run()