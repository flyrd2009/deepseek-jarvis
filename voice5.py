"""
JARVIS Voice System - VOLUME BOOSTED EDITION
Hollywood quality + Full volume + Amplification
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== ENUMS ====================
class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    CONFIDENT = "confident"
    FRIENDLY = "friendly"

class VoiceQuality(Enum):
    STANDARD = "standard"
    WARM = "warm"
    BRIGHT = "bright"
    DARK = "dark"
    BREATHY = "breathy"
    SMOOTH = "smooth"
    VELVETY = "velvety"
    CRISP = "crisp"

# ==================== VOICE CONFIG ====================
@dataclass
class VoiceConfig:
    """Hollywood grade voice configuration"""
    name: str = "JARVIS"
    gender: str = "male"
    age: str = "adult"
    base_f0: float = 115.0
    formants: List[float] = None
    formant_bandwidths: List[float] = None
    formant_strength: float = 1.0
    breathiness: float = 0.06
    jitter: float = 0.008
    shimmer: float = 0.07
    vibrato_rate: float = 5.2
    vibrato_depth: float = 0.018
    tremolo_rate: float = 4.2
    tremolo_depth: float = 0.008
    spectral_tilt: float = -4.5
    harmonic_richness: float = 0.8
    effect_type: str = "natural"
    modulation_freq: float = 0.0
    reverb_amount: float = 0.03
    reverb_decay: float = 0.4
    echo_delay: float = 0.0
    echo_decay: float = 0.0
    emotion: Emotion = Emotion.CONFIDENT
    emotion_intensity: float = 0.6
    quality: VoiceQuality = VoiceQuality.VELVETY
    
    # VOLUME BOOST - New parameters
    master_volume: float = 2.0  # 2x volume boost
    pre_amp: float = 1.5  # Pre-amplification
    output_gain: float = 1.2  # Final gain stage
    
    def __post_init__(self):
        if self.formants is None:
            self.set_formants_for_gender()
        if self.formant_bandwidths is None:
            self.formant_bandwidths = [50, 60, 80, 120, 150]
        self.apply_emotion_to_params()
        self.apply_quality_to_params()
    
    def set_formants_for_gender(self):
        """Professional formant settings"""
        if self.gender == "male":
            if self.age == "adult":
                self.formants = [520, 1120, 2520, 3220, 3870]
            elif self.age == "young":
                self.formants = [620, 1250, 2650, 3350, 4000]
            elif self.age == "old":
                self.formants = [480, 1050, 2450, 3150, 3800]
        elif self.gender == "female":
            if self.age == "adult":
                self.formants = [720, 1350, 2750, 3550, 4200]
            elif self.age == "young":
                self.formants = [780, 1450, 2850, 3650, 4300]
            else:
                self.formants = [950, 1650, 3050, 3850, 4500]
        else:
            self.formants = [620, 1250, 2650, 3350, 4000]
    
    def apply_emotion_to_params(self):
        """Apply emotion modifications"""
        intensity = self.emotion_intensity
        
        emotion_map = {
            Emotion.HAPPY: {'base_f0': 1 + 0.15 * intensity, 'vibrato_rate': 2 * intensity},
            Emotion.SAD: {'base_f0': 1 - 0.12 * intensity, 'vibrato_rate': -1 * intensity},
            Emotion.ANGRY: {'base_f0': 1 + 0.2 * intensity, 'formant_strength': 1 + 0.25 * intensity},
            Emotion.EXCITED: {'base_f0': 1 + 0.25 * intensity, 'vibrato_rate': 3 * intensity},
            Emotion.CALM: {'base_f0': 1 - 0.05 * intensity, 'vibrato_depth': -0.005 * intensity},
            Emotion.FRIENDLY: {'base_f0': 1 + 0.08 * intensity, 'breathiness': 0.1 * intensity}
        }
        
        if self.emotion in emotion_map:
            for param, factor in emotion_map[self.emotion].items():
                if hasattr(self, param):
                    current = getattr(self, param)
                    if isinstance(current, (int, float)):
                        setattr(self, param, current * (1 + factor))
    
    def apply_quality_to_params(self):
        """Apply quality modifications"""
        quality_map = {
            VoiceQuality.VELVETY: {'formants': [f * 0.95 for f in self.formants], 'breathiness': self.breathiness + 0.08},
            VoiceQuality.WARM: {'formants': [f * 0.97 for f in self.formants], 'breathiness': self.breathiness + 0.05},
            VoiceQuality.BRIGHT: {'formants': [f * 1.08 for f in self.formants], 'spectral_tilt': self.spectral_tilt + 4},
            VoiceQuality.DARK: {'formants': [f * 0.92 for f in self.formants], 'spectral_tilt': self.spectral_tilt - 3},
            VoiceQuality.SMOOTH: {'jitter': self.jitter * 0.5, 'shimmer': self.shimmer * 0.5},
        }
        
        if self.quality in quality_map:
            for param, value in quality_map[self.quality].items():
                if param == 'formants':
                    self.formants = value
                else:
                    setattr(self, param, value)

# ==================== PHONEME DATABASE ====================
class PhonemeEngine:
    """Complete phoneme database with AI pronunciation"""
    
    # Core phoneme sounds
    PHONEMES = {
        # Vowels
        'iy': {'type': 'vowel', 'formants': [270, 2300, 3000], 'duration': 0.22},
        'ih': {'type': 'vowel', 'formants': [390, 2000, 2600], 'duration': 0.18},
        'eh': {'type': 'vowel', 'formants': [530, 1850, 2500], 'duration': 0.18},
        'ae': {'type': 'vowel', 'formants': [660, 1700, 2400], 'duration': 0.2},
        'ah': {'type': 'vowel', 'formants': [520, 1190, 2400], 'duration': 0.16},
        'aa': {'type': 'vowel', 'formants': [730, 1090, 2450], 'duration': 0.19},
        'ao': {'type': 'vowel', 'formants': [570, 840, 2410], 'duration': 0.2},
        'uh': {'type': 'vowel', 'formants': [440, 1020, 2250], 'duration': 0.16},
        'uw': {'type': 'vowel', 'formants': [300, 870, 2250], 'duration': 0.21},
        'er': {'type': 'vowel', 'formants': [490, 1350, 1700], 'duration': 0.2},
        'ay': {'type': 'vowel', 'formants': [590, 1850, 2500], 'duration': 0.22},
        'aw': {'type': 'vowel', 'formants': [590, 1300, 2500], 'duration': 0.22},
        'oy': {'type': 'vowel', 'formants': [430, 1300, 2250], 'duration': 0.22},
        'ey': {'type': 'vowel', 'formants': [400, 2000, 2600], 'duration': 0.2},
        'ow': {'type': 'vowel', 'formants': [450, 1000, 2350], 'duration': 0.2},
        
        # Consonants - Plosives
        'p': {'type': 'plosive', 'voiced': False, 'duration': 0.1},
        'b': {'type': 'plosive', 'voiced': True, 'duration': 0.1},
        't': {'type': 'plosive', 'voiced': False, 'duration': 0.1},
        'd': {'type': 'plosive', 'voiced': True, 'duration': 0.1},
        'k': {'type': 'plosive', 'voiced': False, 'duration': 0.12},
        'g': {'type': 'plosive', 'voiced': True, 'duration': 0.12},
        
        # Fricatives
        'f': {'type': 'fricative', 'voiced': False, 'duration': 0.18},
        'v': {'type': 'fricative', 'voiced': True, 'duration': 0.16},
        'th': {'type': 'fricative', 'voiced': False, 'duration': 0.16},
        'dh': {'type': 'fricative', 'voiced': True, 'duration': 0.14},
        's': {'type': 'fricative', 'voiced': False, 'duration': 0.2},
        'z': {'type': 'fricative', 'voiced': True, 'duration': 0.18},
        'sh': {'type': 'fricative', 'voiced': False, 'duration': 0.2},
        'zh': {'type': 'fricative', 'voiced': True, 'duration': 0.18},
        'hh': {'type': 'fricative', 'voiced': False, 'duration': 0.12},
        
        # Affricates
        'ch': {'type': 'affricate', 'voiced': False, 'duration': 0.15},
        'jh': {'type': 'affricate', 'voiced': True, 'duration': 0.15},
        
        # Nasals
        'm': {'type': 'nasal', 'voiced': True, 'duration': 0.15, 'formants': [250, 1200]},
        'n': {'type': 'nasal', 'voiced': True, 'duration': 0.14, 'formants': [250, 1300]},
        'ng': {'type': 'nasal', 'voiced': True, 'duration': 0.15, 'formants': [250, 1100]},
        
        # Liquids
        'l': {'type': 'liquid', 'voiced': True, 'duration': 0.15, 'formants': [400, 1200]},
        'r': {'type': 'liquid', 'voiced': True, 'duration': 0.15, 'formants': [350, 1100]},
        
        # Glides
        'w': {'type': 'glide', 'voiced': True, 'duration': 0.12, 'formants': [300, 800]},
        'y': {'type': 'glide', 'voiced': True, 'duration': 0.12, 'formants': [300, 2200]},
    }
    
    # Letter to phoneme mapping
    LETTER_TO_PHONEME = {
        'a': ['ae'], 'b': ['b'], 'c': ['k'], 'd': ['d'], 'e': ['eh'],
        'f': ['f'], 'g': ['g'], 'h': ['hh'], 'i': ['ih'], 'j': ['jh'],
        'k': ['k'], 'l': ['l'], 'm': ['m'], 'n': ['n'], 'o': ['aa'],
        'p': ['p'], 'q': ['k'], 'r': ['r'], 's': ['s'], 't': ['t'],
        'u': ['ah'], 'v': ['v'], 'w': ['w'], 'x': ['k', 's'], 'y': ['y'],
        'z': ['z'],
    }
    
    # Common words for better pronunciation
    COMMON_WORDS = {
        'the': ['dh', 'ah'],
        'and': ['ae', 'n', 'd'],
        'for': ['f', 'ao', 'r'],
        'you': ['y', 'uw'],
        'are': ['aa', 'r'],
        'is': ['ih', 'z'],
        'it': ['ih', 't'],
        'this': ['dh', 'ih', 's'],
        'that': ['dh', 'ae', 't'],
        'with': ['w', 'ih', 'th'],
        'from': ['f', 'r', 'ah', 'm'],
        'have': ['hh', 'ae', 'v'],
        'will': ['w', 'ih', 'l'],
        'can': ['k', 'ae', 'n'],
        'say': ['s', 'ey'],
        'hello': ['hh', 'ah', 'l', 'ow'],
        'hi': ['hh', 'ay'],
        'good': ['g', 'uh', 'd'],
        'morning': ['m', 'ao', 'r', 'n', 'ih', 'ng'],
        'evening': ['iy', 'v', 'ih', 'n', 'ih', 'ng'],
        'night': ['n', 'ay', 't'],
        'please': ['p', 'l', 'iy', 'z'],
        'thanks': ['th', 'ae', 'ng', 'k', 's'],
        'thank': ['th', 'ae', 'ng', 'k'],
        'welcome': ['w', 'eh', 'l', 'k', 'ah', 'm'],
        'sorry': ['s', 'aa', 'r', 'iy'],
        'excuse': ['eh', 'k', 's', 'k', 'y', 'uw', 'z'],
        'yes': ['y', 'eh', 's'],
        'no': ['n', 'ow'],
        'okay': ['ow', 'k', 'ey'],
        'sure': ['sh', 'uh', 'r'],
        'maybe': ['m', 'ey', 'b', 'iy'],
        'always': ['aa', 'l', 'w', 'ey', 'z'],
        'never': ['n', 'eh', 'v', 'er'],
        'now': ['n', 'aw'],
        'then': ['dh', 'eh', 'n'],
        'here': ['hh', 'iy', 'r'],
        'there': ['dh', 'eh', 'r'],
        'where': ['w', 'eh', 'r'],
        'why': ['w', 'ay'],
        'how': ['hh', 'aw'],
        'what': ['w', 'ah', 't'],
        'when': ['w', 'eh', 'n'],
        'who': ['hh', 'uw'],
        'which': ['w', 'ih', 'ch'],
        'time': ['t', 'ay', 'm'],
        'day': ['d', 'ey'],
        'week': ['w', 'iy', 'k'],
        'month': ['m', 'ah', 'n', 'th'],
        'year': ['y', 'iy', 'r'],
        'today': ['t', 'ah', 'd', 'ey'],
        'tomorrow': ['t', 'ah', 'm', 'aa', 'r', 'ow'],
        'yesterday': ['y', 'eh', 's', 't', 'er', 'd', 'ey'],
        'people': ['p', 'iy', 'p', 'ah', 'l'],
        'person': ['p', 'er', 's', 'ah', 'n'],
        'man': ['m', 'ae', 'n'],
        'woman': ['w', 'uh', 'm', 'ah', 'n'],
        'child': ['ch', 'ay', 'l', 'd'],
        'world': ['w', 'er', 'l', 'd'],
        'earth': ['er', 'th'],
        'sun': ['s', 'ah', 'n'],
        'moon': ['m', 'uw', 'n'],
        'star': ['s', 't', 'aa', 'r'],
        'water': ['w', 'ao', 't', 'er'],
        'fire': ['f', 'ay', 'er'],
        'air': ['eh', 'r'],
        'love': ['l', 'ah', 'v'],
        'hate': ['hh', 'ey', 't'],
        'life': ['l', 'ay', 'f'],
        'death': ['d', 'eh', 'th'],
        'big': ['b', 'ih', 'g'],
        'small': ['s', 'm', 'aa', 'l'],
        'hot': ['hh', 'aa', 't'],
        'cold': ['k', 'ow', 'l', 'd'],
        'new': ['n', 'uw'],
        'old': ['ow', 'l', 'd'],
        'young': ['y', 'ah', 'ng'],
        'happy': ['hh', 'ae', 'p', 'iy'],
        'sad': ['s', 'ae', 'd'],
        'angry': ['ae', 'ng', 'g', 'r', 'iy'],
        'tired': ['t', 'ay', 'er', 'd'],
        'sleep': ['s', 'l', 'iy', 'p'],
        'wake': ['w', 'ey', 'k'],
        'work': ['w', 'er', 'k'],
        'play': ['p', 'l', 'ey'],
        'talk': ['t', 'aa', 'k'],
        'walk': ['w', 'aa', 'k'],
        'run': ['r', 'ah', 'n'],
        'jump': ['jh', 'ah', 'm', 'p'],
        'sit': ['s', 'ih', 't'],
        'stand': ['s', 't', 'ae', 'n', 'd'],
        'give': ['g', 'ih', 'v'],
        'take': ['t', 'ey', 'k'],
        'get': ['g', 'eh', 't'],
        'put': ['p', 'uh', 't'],
        'make': ['m', 'ey', 'k'],
        'do': ['d', 'uw'],
        'be': ['b', 'iy'],
        'see': ['s', 'iy'],
        'hear': ['hh', 'iy', 'r'],
        'feel': ['f', 'iy', 'l'],
        'think': ['th', 'ih', 'ng', 'k'],
        'know': ['n', 'ow'],
        'want': ['w', 'aa', 'n', 't'],
        'need': ['n', 'iy', 'd'],
        'like': ['l', 'ay', 'k'],
        'help': ['hh', 'eh', 'l', 'p'],
        'jarvis': ['jh', 'aa', 'r', 'v', 'ih', 's'],
        'friday': ['f', 'r', 'ay', 'd', 'ey'],
        'pepper': ['p', 'eh', 'p', 'er'],
        'stark': ['s', 't', 'aa', 'r', 'k'],
        'tony': ['t', 'ow', 'n', 'iy'],
        'iron': ['ay', 'er', 'n'],
        'man': ['m', 'ae', 'n'],
        'sir': ['s', 'er'],
        'boss': ['b', 'aa', 's'],
        'mister': ['m', 'ih', 's', 't', 'er'],
        'captain': ['k', 'ae', 'p', 't', 'ih', 'n'],
        'doctor': ['d', 'aa', 'k', 't', 'er'],
        'professor': ['p', 'r', 'ah', 'f', 'eh', 's', 'er'],
    }
    
    @classmethod
    def get_phoneme(cls, symbol):
        """Get phoneme data"""
        return cls.PHONEMES.get(symbol, cls.PHONEMES.get('ah'))
    
    @classmethod
    def word_to_phonemes(cls, word):
        """Convert any word to phonemes"""
        word = word.lower().strip()
        
        # Check common words first
        if word in cls.COMMON_WORDS:
            return cls.COMMON_WORDS[word]
        
        # Simple prediction
        phonemes = []
        for char in word:
            if char in cls.LETTER_TO_PHONEME:
                phonemes.extend(cls.LETTER_TO_PHONEME[char])
            elif char in 'aeiou':
                phonemes.append('ah')
        
        return phonemes if phonemes else ['ah']

# ==================== HOLLYWOOD VOICE ENGINE ====================
class HollywoodVoiceEngine:
    """Professional voice engine with VOLUME BOOST"""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.phonemes = PhonemeEngine()
    
    def generate_glottal_pulse(self, duration, f0, config):
        """Generate glottal pulse with VOLUME BOOST"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples, False)
        
        # Generate pulse with harmonics - BOOSTED AMPLITUDE
        pulse = np.zeros(n_samples)
        for h in range(1, 5):
            amp = 1.0 / h
            pulse += amp * np.sin(2 * np.pi * h * f0 * t)
        
        # BOOST: Increase base amplitude
        pulse = pulse * 1.5
        
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
        
        # BOOST: Pre-amplification
        pulse = pulse * config.pre_amp
        
        return pulse
    
    def apply_vocal_tract(self, source, formants, bandwidths, config):
        """Apply vocal tract filtering with gain staging"""
        if len(source) == 0:
            return source
        
        result = source.copy()
        
        # Apply formants
        for f, bw in zip(formants[:3], bandwidths[:3]):
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
        
        # BOOST: Maintain gain through formants
        result = result * config.formant_strength * 1.2
        
        return result
    
    def generate_phoneme(self, phoneme, duration, config):
        """Generate individual phoneme with VOLUME BOOST"""
        phoneme_data = self.phonemes.get_phoneme(phoneme)
        if not phoneme_data:
            return None
        
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return None
        
        # BOOSTED amplitudes for all sound types
        if phoneme_data['type'] == 'vowel':
            voice_formants = config.formants[:3]
            target_formants = phoneme_data['formants']
            
            blended = []
            for i in range(3):
                blended.append(target_formants[i] * 0.7 + voice_formants[i] * 0.3)
            
            source = self.generate_glottal_pulse(duration, config.base_f0, config)
            sound = self.apply_vocal_tract(source, blended, config.formant_bandwidths, config)
            
        else:
            # Consonants - BOOSTED noise floor
            if phoneme_data['type'] == 'plosive':
                burst_len = min(int(0.01 * self.sr), n_samples)
                sound = np.random.randn(n_samples) * 0.3  # Increased from 0.1
                sound[:burst_len] *= 8  # Increased from 5
            elif phoneme_data['type'] == 'fricative':
                sound = np.random.randn(n_samples) * 0.4  # Increased from 0.2
            elif phoneme_data['type'] == 'nasal':
                source = self.generate_glottal_pulse(duration, config.base_f0 * 0.8, config)
                sound = source * 0.8  # Increased from 0.5
            else:
                source = self.generate_glottal_pulse(duration, config.base_f0, config)
                sound = source * 0.5  # Increased from 0.3
        
        # Apply envelope
        envelope = np.ones(n_samples)
        attack = int(0.005 * self.sr)
        release = int(0.01 * self.sr)
        
        if attack < n_samples:
            envelope[:attack] = np.linspace(0, 1, attack)
        if release < n_samples:
            envelope[-release:] = np.linspace(1, 0, release)
        
        if len(sound) > len(envelope):
            sound = sound[:len(envelope)]
        elif len(sound) < len(envelope):
            sound = np.pad(sound, (0, len(envelope) - len(sound)))
        
        sound = sound * envelope
        
        # BOOST: Apply master volume and output gain
        sound = sound * config.master_volume * config.output_gain
        
        # Limit to prevent distortion, but keep volume high
        max_val = np.max(np.abs(sound))
        if max_val > 1.0:
            sound = sound / max_val * 0.98  # Soft limit at 0.98
        
        return sound
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)
    
    def apply_effects(self, audio, config):
        """Apply audio effects with gain staging"""
        if len(audio) == 0:
            return audio
        
        # Store original for gain staging
        original_rms = np.sqrt(np.mean(audio**2))
        
        # Reverb
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
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * config.modulation_freq * t)
            audio = audio * mod
        
        # BOOST: Maintain volume after effects
        new_rms = np.sqrt(np.mean(audio**2))
        if new_rms > 0:
            gain_compensation = original_rms / new_rms
            audio = audio * gain_compensation * 1.1  # Slight overall boost
        
        # Final limiting - HOT but clean
        if np.max(np.abs(audio)) > 0.99:
            audio = audio / np.max(np.abs(audio)) * 0.98
        
        return audio

# ==================== JARVIS VOICE SYSTEM ====================
class JarvisVoiceSystem:
    """Complete voice system - HOLLYWOOD QUALITY + FULL VOLUME"""
    
    def __init__(self):
        self.engine = HollywoodVoiceEngine()
        self.config = VoiceConfig()
        self.phonemes = PhonemeEngine()
        
        # Voice presets with VOLUME BOOST
        self.presets = {
            'jarvis': VoiceConfig(
                name="JARVIS", gender='male', base_f0=112,
                formants=[520, 1120, 2520, 3220, 3870],
                breathiness=0.04, quality=VoiceQuality.VELVETY,
                master_volume=2.2, pre_amp=1.6, output_gain=1.3  # BOOSTED
            ),
            'friday': VoiceConfig(
                name="FRIDAY", gender='female', base_f0=195,
                formants=[720, 1350, 2750, 3550, 4200],
                breathiness=0.08, quality=VoiceQuality.BRIGHT,
                master_volume=2.0, pre_amp=1.5, output_gain=1.2
            ),
            'pepper': VoiceConfig(
                name="Pepper", gender='female', base_f0=185,
                formants=[710, 1330, 2730, 3530, 4180],
                breathiness=0.08, quality=VoiceQuality.WARM,
                master_volume=2.1, pre_amp=1.5, output_gain=1.2
            ),
            'robotic': VoiceConfig(
                name="Robotic", gender='male', base_f0=130,
                effect_type='robotic', modulation_freq=30,
                jitter=0.02, quality=VoiceQuality.STANDARD,
                master_volume=2.5, pre_amp=1.8, output_gain=1.4  # EXTRA BOOST for robotic
            ),
        }
        
        # Cache
        self.cache_dir = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Voice set to: {self.config.name} (Volume: {self.config.master_volume}x)")
        else:
            print(f"✗ Voice '{name}' not found")
    
    def set_emotion(self, emotion: Emotion, intensity: float = 0.5):
        """Set emotion"""
        self.config.emotion = emotion
        self.config.emotion_intensity = intensity
        self.config.apply_emotion_to_params()
        print(f"✓ Emotion: {emotion.value} ({intensity})")
    
    def set_volume(self, level: float):
        """Manually set volume level (0.5 to 4.0)"""
        self.config.master_volume = max(0.5, min(4.0, level))
        print(f"✓ Volume set to: {self.config.master_volume}x")
    
    def text_to_phonemes(self, text):
        """Convert ANY text to phonemes"""
        text = text.lower().strip()
        words = text.split()
        
        all_phonemes = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if not clean_word:
                continue
            
            word_phonemes = self.phonemes.word_to_phonemes(clean_word)
            all_phonemes.extend(word_phonemes)
            all_phonemes.append('pau')
        
        return all_phonemes
    
    def generate_speech(self, text, output_file=None):
        """Generate speech with FULL VOLUME"""
        if not text:
            print("No text provided")
            return None
        
        print(f"\n🎙️ Generating: '{text}'")
        print(f"🔊 Volume: {self.config.master_volume}x")
        
        # Convert to phonemes
        phonemes = self.text_to_phonemes(text)
        print(f"📝 Phonemes: {len(phonemes)} sounds")
        
        if not phonemes:
            return None
        
        # Check cache
        cache_key = hashlib.md5(f"{text}{self.config.name}{self.config.master_volume}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            print("📦 Loading from cache")
            audio = np.load(cache_file)
        else:
            # Generate audio
            audio_segments = []
            
            for phoneme in phonemes:
                if phoneme == 'pau':
                    segment = self.engine.generate_silence(0.1)
                else:
                    phoneme_data = self.phonemes.get_phoneme(phoneme)
                    if phoneme_data:
                        duration = phoneme_data.get('duration', 0.15)
                        segment = self.engine.generate_phoneme(phoneme, duration, self.config)
                    else:
                        continue
                
                if segment is not None and len(segment) > 0:
                    audio_segments.append(segment)
            
            if not audio_segments:
                return None
            
            audio = np.concatenate(audio_segments)
            
            # Apply effects
            audio = self.engine.apply_effects(audio, self.config)
            
            # Final VOLUME BOOST (extra safety)
            if np.max(np.abs(audio)) < 0.95:
                audio = audio * 1.1
            
            # Cache
            np.save(cache_file, audio)
        
        # Save if requested
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"✓ Saved to: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak text with FULL VOLUME"""
        audio = self.generate_speech(text)
        if audio is not None:
            # BOOST: Extra volume on playback
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def list_voices(self):
        """List all voices"""
        print("\n🎭 Available Voices:")
        for name, config in self.presets.items():
            print(f"  • {name:10} ({config.gender}, Volume: {config.master_volume}x)")

# ==================== INTERACTIVE CONSOLE ====================
class VoiceConsole:
    """Interactive console with volume control"""
    
    def __init__(self):
        self.jarvis = JarvisVoiceSystem()
        self.running = True
    
    def run(self):
        """Run console"""
        print("\n" + "="*60)
        print("🎙️ JARVIS HOLLYWOOD VOICE - VOLUME BOOSTED")
        print("   Any word - Full Volume Output")
        print("="*60)
        print("\nCommands:")
        print("  /voices    - List voices")
        print("  /voice     - Change voice (jarvis/friday/pepper/robotic)")
        print("  /emotion   - Set emotion (happy/sad/angry/calm)")
        print("  /volume    - Set volume (0.5 to 4.0)")
        print("  /exit      - Exit")
        print("-" * 40)
        print("Type anything - JARVIS will speak LOUD and CLEAR!")
        
        while self.running:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    cmd = user_input[1:].split()
                    
                    if cmd[0] == 'exit':
                        self.running = False
                    elif cmd[0] == 'voices':
                        self.jarvis.list_voices()
                    elif cmd[0] == 'voice' and len(cmd) > 1:
                        self.jarvis.set_voice(cmd[1])
                    elif cmd[0] == 'emotion' and len(cmd) > 1:
                        try:
                            emotion = Emotion(cmd[1].lower())
                            self.jarvis.set_emotion(emotion)
                        except:
                            print("Emotions: happy, sad, angry, calm")
                    elif cmd[0] == 'volume' and len(cmd) > 1:
                        try:
                            vol = float(cmd[1])
                            self.jarvis.set_volume(vol)
                        except:
                            print("Use: /volume 2.0")
                    else:
                        print("Unknown command")
                else:
                    self.jarvis.speak(user_input)
                    
            except KeyboardInterrupt:
                self.running = False
                print("\nGoodbye!")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--voice', '-v', default='jarvis')
    parser.add_argument('--emotion', '-e')
    parser.add_argument('--volume', '-vol', type=float, default=2.0)
    parser.add_argument('--text', '-t')
    parser.add_argument('--output', '-o')
    
    args = parser.parse_args()
    
    if args.interactive:
        console = VoiceConsole()
        console.run()
    elif args.text:
        jarvis = JarvisVoiceSystem()
        jarvis.set_voice(args.voice)
        jarvis.set_volume(args.volume)
        if args.emotion:
            try:
                emotion = Emotion(args.emotion.lower())
                jarvis.set_emotion(emotion)
            except:
                pass
        
        print(f"🔊 Volume: {args.volume}x")
        if args.output:
            jarvis.generate_speech(args.text, args.output)
        else:
            jarvis.speak(args.text)
    else:
        console = VoiceConsole()
        console.run()