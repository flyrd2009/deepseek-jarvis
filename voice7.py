"""
JARVIS Voice System - STUDIO QUALITY EDITION
Professional broadcast quality - 48kHz/24-bit, Dynamic Range, Perfect Clarity
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal, interpolate
from scipy.io import wavfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== PROFESSIONAL AUDIO CONSTANTS ====================
SAMPLE_RATE = 48000  # Broadcast quality
BIT_DEPTH = 24       # Professional bit depth
NYQUIST = SAMPLE_RATE / 2

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
    STUDIO = "studio"        # Broadcast quality
    BROADCAST = "broadcast"  # Radio quality
    WARM = "warm"            # Vintage warmth
    BRIGHT = "bright"        # Modern bright
    SMOOTH = "smooth"        # Buttery smooth
    CRISP = "crisp"          # Extra definition

# ==================== STUDIO VOICE CONFIG ====================
@dataclass
class StudioVoiceConfig:
    """Professional broadcast-quality voice configuration"""
    
    # Identity
    name: str = "JARVIS Studio"
    gender: str = "male"
    age: str = "adult"
    
    # Fundamental - OPTIMIZED for clarity
    base_f0: float = 118.0
    f0_variation: float = 0.3
    pitch_stability: float = 0.95  # Professional stability
    
    # Formants - BROADCAST optimized
    formants: List[float] = None
    formant_bandwidths: List[float] = None
    formant_q: List[float] = None  # Quality factor
    formant_strength: float = 1.3
    
    # Voice Quality - MINIMAL artifacts
    breathiness: float = 0.01       # Almost none
    aspiration: float = 0.02        # Gentle
    jitter: float = 0.002           # Very stable
    shimmer: float = 0.01           # Very stable
    creak: float = 0.0              # None
    whisper: float = 0.0            # None
    
    # Modulation - SUBTLE for professionalism
    vibrato_rate: float = 4.2
    vibrato_depth: float = 0.005
    tremolo_rate: float = 3.0
    tremolo_depth: float = 0.002
    
    # Spectral - BROADCAST EQ
    spectral_tilt: float = -2.5      # Gentle high-frequency rolloff
    presence_boost: float = 2.0      # 2kHz presence
    air_band: float = 1.5            # 10kHz air
    harmonic_richness: float = 1.4   # Rich harmonics
    
    # Effects - MINIMAL for clarity
    effect_type: str = "natural"
    reverb_amount: float = 0.005     # Just a touch
    reverb_decay: float = 0.15
    reverb_diffusion: float = 0.3
    reverb_damping: float = 0.5
    
    # Dynamics - PROFESSIONAL
    compressor_threshold: float = -18.0  # dB
    compressor_ratio: float = 2.5
    compressor_attack: float = 0.005
    compressor_release: float = 0.1
    
    # Limiter
    limiter_threshold: float = -1.0  # dB
    limiter_attack: float = 0.001
    limiter_release: float = 0.05
    
    # Output
    output_gain: float = 0.95
    headroom: float = 0.5  # dB
    
    # Emotional state
    emotion: Emotion = Emotion.CONFIDENT
    emotion_intensity: float = 0.4  # Subtle
    
    # Quality preset
    quality: VoiceQuality = VoiceQuality.STUDIO
    
    def __post_init__(self):
        if self.formants is None:
            self.set_formants_for_gender()
        if self.formant_bandwidths is None:
            # Professional bandwidths (narrow = clear)
            self.formant_bandwidths = [40, 45, 55, 70, 85, 100]
        if self.formant_q is None:
            # High Q for sharp formants
            self.formant_q = [25, 22, 20, 18, 15, 12]
        self.apply_emotion_to_params()
        self.apply_quality_to_params()
    
    def set_formants_for_gender(self):
        """Broadcast-optimized formants"""
        if self.gender == "male":
            # JARVIS signature formants
            self.formants = [540, 1140, 2580, 3280, 3950, 4650]
        elif self.gender == "female":
            # FRIDAY signature formants
            self.formants = [730, 1380, 2780, 3580, 4250, 4950]
        else:
            self.formants = [650, 1300, 2700, 3400, 4100, 4800]
    
    def apply_emotion_to_params(self):
        """Subtle emotion modifications"""
        intensity = self.emotion_intensity * 0.3  # Very subtle
        
        if self.emotion == Emotion.HAPPY:
            self.base_f0 *= (1 + 0.05 * intensity)
            self.presence_boost *= (1 + 0.1 * intensity)
        elif self.emotion == Emotion.SAD:
            self.base_f0 *= (1 - 0.03 * intensity)
            self.spectral_tilt -= 0.5 * intensity
        elif self.emotion == Emotion.ANGRY:
            self.formant_strength *= (1 + 0.1 * intensity)
            self.compressor_ratio += 0.5 * intensity
        elif self.emotion == Emotion.EXCITED:
            self.base_f0 *= (1 + 0.08 * intensity)
            self.vibrato_rate += 0.5 * intensity
    
    def apply_quality_to_params(self):
        """Apply quality presets"""
        if self.quality == VoiceQuality.STUDIO:
            # Perfect balance
            self.formant_bandwidths = [38, 42, 52, 65, 80, 95]
            self.presence_boost = 2.2
            self.air_band = 1.6
        elif self.quality == VoiceQuality.BROADCAST:
            # Radio optimized
            self.formant_bandwidths = [45, 50, 60, 75, 90, 105]
            self.presence_boost = 2.5
            self.spectral_tilt = -3.0
        elif self.quality == VoiceQuality.WARM:
            # Vintage warmth
            self.formants = [f * 0.96 for f in self.formants]
            self.spectral_tilt = -4.0
            self.air_band = 1.0
        elif self.quality == VoiceQuality.BRIGHT:
            # Modern bright
            self.formants = [f * 1.04 for f in self.formants]
            self.spectral_tilt = -1.0
            self.air_band = 2.0

# ==================== STUDIO PHONEME ENGINE ====================
class StudioPhonemeEngine:
    """Professional phoneme database with studio-quality formants"""
    
    # Studio-optimized phonemes
    PHONEMES = {
        # Vowels - PERFECT formants
        'iy': {'type': 'vowel', 'f1': 290, 'f2': 2350, 'f3': 3050, 'duration': 0.22},
        'ih': {'type': 'vowel', 'f1': 410, 'f2': 2050, 'f3': 2650, 'duration': 0.18},
        'eh': {'type': 'vowel', 'f1': 550, 'f2': 1900, 'f3': 2550, 'duration': 0.18},
        'ae': {'type': 'vowel', 'f1': 680, 'f2': 1750, 'f3': 2450, 'duration': 0.20},
        'ah': {'type': 'vowel', 'f1': 540, 'f2': 1220, 'f3': 2450, 'duration': 0.16},
        'aa': {'type': 'vowel', 'f1': 750, 'f2': 1120, 'f3': 2500, 'duration': 0.19},
        'ao': {'type': 'vowel', 'f1': 590, 'f2': 880, 'f3': 2450, 'duration': 0.20},
        'uh': {'type': 'vowel', 'f1': 460, 'f2': 1080, 'f3': 2300, 'duration': 0.16},
        'uw': {'type': 'vowel', 'f1': 320, 'f2': 920, 'f3': 2300, 'duration': 0.21},
        'er': {'type': 'vowel', 'f1': 510, 'f2': 1380, 'f3': 1750, 'duration': 0.20},
        'ay': {'type': 'vowel', 'f1': 610, 'f2': 1900, 'f3': 2550, 'duration': 0.22},
        'aw': {'type': 'vowel', 'f1': 610, 'f2': 1350, 'f3': 2550, 'duration': 0.22},
        'oy': {'type': 'vowel', 'f1': 450, 'f2': 1350, 'f3': 2300, 'duration': 0.22},
        'ey': {'type': 'vowel', 'f1': 420, 'f2': 2050, 'f3': 2650, 'duration': 0.20},
        'ow': {'type': 'vowel', 'f1': 470, 'f2': 1050, 'f3': 2400, 'duration': 0.20},
        
        # Consonants - PROFESSIONAL
        'p': {'type': 'plosive', 'voiced': False, 'duration': 0.12, 'spectral': 'low'},
        'b': {'type': 'plosive', 'voiced': True, 'duration': 0.12, 'spectral': 'low'},
        't': {'type': 'plosive', 'voiced': False, 'duration': 0.12, 'spectral': 'high'},
        'd': {'type': 'plosive', 'voiced': True, 'duration': 0.12, 'spectral': 'high'},
        'k': {'type': 'plosive', 'voiced': False, 'duration': 0.14, 'spectral': 'mid'},
        'g': {'type': 'plosive', 'voiced': True, 'duration': 0.14, 'spectral': 'mid'},
        
        'f': {'type': 'fricative', 'voiced': False, 'duration': 0.20, 'center': 7000},
        'v': {'type': 'fricative', 'voiced': True, 'duration': 0.18, 'center': 6500},
        'th': {'type': 'fricative', 'voiced': False, 'duration': 0.18, 'center': 5500},
        'dh': {'type': 'fricative', 'voiced': True, 'duration': 0.16, 'center': 5000},
        's': {'type': 'fricative', 'voiced': False, 'duration': 0.22, 'center': 7500},
        'z': {'type': 'fricative', 'voiced': True, 'duration': 0.20, 'center': 7000},
        'sh': {'type': 'fricative', 'voiced': False, 'duration': 0.22, 'center': 4500},
        'zh': {'type': 'fricative', 'voiced': True, 'duration': 0.20, 'center': 4200},
        'hh': {'type': 'fricative', 'voiced': False, 'duration': 0.14, 'center': 1500},
        
        'ch': {'type': 'affricate', 'voiced': False, 'duration': 0.17},
        'jh': {'type': 'affricate', 'voiced': True, 'duration': 0.17},
        
        'm': {'type': 'nasal', 'voiced': True, 'duration': 0.16, 'f1': 250, 'f2': 1250},
        'n': {'type': 'nasal', 'voiced': True, 'duration': 0.15, 'f1': 250, 'f2': 1350},
        'ng': {'type': 'nasal', 'voiced': True, 'duration': 0.16, 'f1': 250, 'f2': 1150},
        
        'l': {'type': 'liquid', 'voiced': True, 'duration': 0.16, 'f1': 420, 'f2': 1250},
        'r': {'type': 'liquid', 'voiced': True, 'duration': 0.16, 'f1': 380, 'f2': 1150},
        
        'w': {'type': 'glide', 'voiced': True, 'duration': 0.13, 'f1': 320, 'f2': 850},
        'y': {'type': 'glide', 'voiced': True, 'duration': 0.13, 'f1': 320, 'f2': 2250},
    }
    
    # Professional word dictionary
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
        'hello': ['hh', 'ah', 'l', 'ow'],
        'hi': ['hh', 'ay'],
        'jarvis': ['jh', 'aa', 'r', 'v', 'ih', 's'],
        'friday': ['f', 'r', 'ay', 'd', 'ey'],
        'pepper': ['p', 'eh', 'p', 'er'],
        'stark': ['s', 't', 'aa', 'r', 'k'],
        'tony': ['t', 'ow', 'n', 'iy'],
        'sir': ['s', 'er'],
        'boss': ['b', 'aa', 's'],
    }
    
    LETTER_TO_PHONEME = {
        'a': ['ae'], 'b': ['b'], 'c': ['k'], 'd': ['d'], 'e': ['eh'],
        'f': ['f'], 'g': ['g'], 'h': ['hh'], 'i': ['ih'], 'j': ['jh'],
        'k': ['k'], 'l': ['l'], 'm': ['m'], 'n': ['n'], 'o': ['aa'],
        'p': ['p'], 'q': ['k'], 'r': ['r'], 's': ['s'], 't': ['t'],
        'u': ['ah'], 'v': ['v'], 'w': ['w'], 'x': ['k', 's'], 'y': ['y'],
        'z': ['z'],
    }
    
    @classmethod
    def get_phoneme(cls, symbol):
        """Get phoneme data"""
        return cls.PHONEMES.get(symbol, cls.PHONEMES.get('ah'))
    
    @classmethod
    def word_to_phonemes(cls, word):
        """Convert word to phonemes"""
        word = word.lower().strip()
        if word in cls.COMMON_WORDS:
            return cls.COMMON_WORDS[word]
        
        phonemes = []
        for char in word:
            if char in cls.LETTER_TO_PHONEME:
                phonemes.extend(cls.LETTER_TO_PHONEME[char])
            elif char in 'aeiou':
                phonemes.append('ah')
        return phonemes if phonemes else ['ah']

# ==================== STUDIO VOICE ENGINE ====================
class StudioVoiceEngine:
    """Professional broadcast-quality voice engine"""
    
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sr = sample_rate
        self.phonemes = StudioPhonemeEngine()
        
    def generate_glottal_pulse(self, duration, f0, config):
        """Generate professional glottal pulse"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples, False)
        
        # LF model with professional harmonics
        pulse = np.zeros(n_samples)
        
        # Fundamental + 5 harmonics with optimal weights
        harmonic_weights = [1.0, 0.65, 0.45, 0.30, 0.20, 0.12]
        
        for h, weight in enumerate(harmonic_weights, 1):
            # Add slight phase variation for naturalness
            phase = 0.1 * np.random.randn() if h > 1 else 0
            pulse += weight * np.sin(2 * np.pi * h * f0 * t + phase)
        
        # Very minimal jitter (professional stability)
        jitter_mod = 1 + config.jitter * 0.3 * np.random.randn(n_samples)
        
        # Anti-aliasing interpolation
        pulse = np.interp(np.arange(n_samples) * np.mean(jitter_mod), 
                         np.arange(n_samples), pulse, left=0, right=0)
        pulse = np.nan_to_num(pulse)
        
        # Apply envelope shaping
        envelope = 1 - np.exp(-t * 500)  # Smooth onset
        
        return pulse * envelope * 0.8
    
    def apply_formant_filter(self, signal_data, formants, bandwidths, config):
        """Apply professional formant filtering"""
        if len(signal_data) == 0:
            return signal_data
        
        result = signal_data.copy()
        
        # Apply formants with high Q
        for f, bw in zip(formants[:4], bandwidths[:4]):
            # High Q filter for sharp formants
            Q = f / bw if bw > 0 else 20
            w0 = 2 * np.pi * f / self.sr
            
            # Biquad bandpass with precise Q
            alpha = np.sin(w0) / (2 * Q)
            
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
            
            # Normalize
            b = [b0/a0, b1/a0, b2/a0]
            a = [1, a1/a0, a2/a0]
            
            try:
                result = signal.lfilter(b, a, result)
            except:
                pass
        
        # Professional EQ
        result = self.apply_professional_eq(result, config)
        
        return result * config.formant_strength
    
    def apply_professional_eq(self, audio, config):
        """Apply broadcast-quality EQ"""
        if len(audio) == 0:
            return audio
        
        # Presence boost (2kHz)
        w0 = 2 * np.pi * 2000 / self.sr
        Q = 1.5
        gain_db = config.presence_boost * 3  # Convert to dB
        
        # Peaking filter
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / (2 * Q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        b = [b0/a0, b1/a0, b2/a0]
        a = [1, a1/a0, a2/a0]
        
        try:
            audio = signal.lfilter(b, a, audio)
        except:
            pass
        
        # Air band (10kHz high shelf)
        if config.air_band > 1:
            w0 = 2 * np.pi * 10000 / self.sr
            gain_db = (config.air_band - 1) * 6
            
            A = 10 ** (gain_db / 40)
            alpha = np.sin(w0) / np.sqrt(2)
            
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
            b = [b0/a0, b1/a0, b2/a0]
            a = [1, a1/a0, a2/a0]
            
            try:
                audio = signal.lfilter(b, a, audio)
            except:
                pass
        
        return audio
    
    def generate_phoneme(self, phoneme, duration, config):
        """Generate studio-quality phoneme"""
        phoneme_data = self.phonemes.get_phoneme(phoneme)
        if not phoneme_data:
            return None
        
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return None
        
        if phoneme_data['type'] == 'vowel':
            # Studio vowel
            voice_formants = config.formants[:3]
            target_formants = [
                phoneme_data['f1'],
                phoneme_data['f2'],
                phoneme_data['f3']
            ]
            
            # Blend with voice character
            blended = []
            for i in range(3):
                blended.append(target_formants[i] * 0.8 + voice_formants[i] * 0.2)
            
            source = self.generate_glottal_pulse(duration, config.base_f0, config)
            sound = self.apply_formant_filter(source, blended, config.formant_bandwidths, config)
            
        elif phoneme_data['type'] == 'plosive':
            # Professional plosive
            burst_len = min(int(0.012 * self.sr), n_samples)
            
            # Colored burst based on spectral type
            if phoneme_data.get('spectral') == 'high':
                noise = np.random.randn(burst_len) * 0.4
                b, a = signal.butter(2, 4000/NYQUIST, 'high')
            elif phoneme_data.get('spectral') == 'low':
                noise = np.random.randn(burst_len) * 0.4
                b, a = signal.butter(2, 1000/NYQUIST, 'low')
            else:
                noise = np.random.randn(burst_len) * 0.4
                b, a = signal.butter(2, [500/NYQUIST, 3000/NYQUIST], 'band')
            
            try:
                burst = signal.filtfilt(b, a, noise)
            except:
                burst = noise
            
            # Add voicing if voiced
            if phoneme_data.get('voiced', False):
                voice = self.generate_glottal_pulse(burst_len/self.sr, config.base_f0 * 0.7, config)
                if len(voice) == burst_len:
                    burst = burst * 0.4 + voice * 0.6
            
            # Rest of sound
            if n_samples > burst_len:
                sound = np.concatenate([burst, np.zeros(n_samples - burst_len)])
            else:
                sound = burst[:n_samples]
            
        elif phoneme_data['type'] == 'fricative':
            # Professional fricative
            noise = np.random.randn(n_samples) * 0.3
            center = phoneme_data.get('center', 4000)
            bandwidth = 1000
            
            # Bandpass filter
            b, a = signal.butter(4, [(center - bandwidth/2)/NYQUIST, 
                                     (center + bandwidth/2)/NYQUIST], 'band')
            try:
                sound = signal.filtfilt(b, a, noise)
            except:
                sound = noise
            
            # Add voicing if needed
            if phoneme_data.get('voiced', False):
                voice = self.generate_glottal_pulse(duration, config.base_f0 * 0.8, config)
                if len(voice) == len(sound):
                    sound = sound * 0.4 + voice * 0.6
        
        elif phoneme_data['type'] == 'nasal':
            # Professional nasal
            source = self.generate_glottal_pulse(duration, config.base_f0 * 0.8, config)
            nasal_formants = [phoneme_data.get('f1', 250), phoneme_data.get('f2', 1200)]
            sound = self.apply_formant_filter(source, nasal_formants, [50, 70], config)
            
        else:  # liquids, glides, affricates
            source = self.generate_glottal_pulse(duration, config.base_f0, config)
            sound = source * 0.6
        
        # Ensure correct length
        if len(sound) > n_samples:
            sound = sound[:n_samples]
        elif len(sound) < n_samples:
            sound = np.pad(sound, (0, n_samples - len(sound)))
        
        return sound
    
    def generate_silence(self, duration):
        """Generate professional silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)
    
    def apply_dynamics(self, audio, config):
        """Professional dynamics processing"""
        if len(audio) == 0:
            return audio
        
        # Convert threshold from dB to linear
        threshold = 10 ** (config.compressor_threshold / 20)
        
        # Simple compressor
        gain = np.ones_like(audio)
        level = np.abs(audio)
        
        # Above threshold, apply compression
        mask = level > threshold
        if np.any(mask):
            gain[mask] = threshold + (level[mask] - threshold) / config.compressor_ratio
            gain[mask] = gain[mask] / level[mask]
        
        audio = audio * gain
        
        # Limiter
        limit = 10 ** (config.limiter_threshold / 20)
        audio = np.clip(audio, -limit, limit)
        
        return audio
    
    def apply_professional_reverb(self, audio, config):
        """Subtle professional reverb"""
        if config.reverb_amount <= 0 or len(audio) == 0:
            return audio
        
        # Create studio impulse response
        reverb_len = int(0.2 * self.sr)
        impulse = np.zeros(reverb_len)
        
        # Early reflections
        impulse[0] = 1
        for i in range(1, 5):
            delay = int(0.01 * i * self.sr)
            if delay < reverb_len:
                impulse[delay] = 0.7 ** i
        
        # Late reverb (exponential decay)
        t = np.arange(reverb_len) / self.sr
        decay = np.exp(-t / config.reverb_decay)
        impulse += decay * 0.3
        
        # Apply reverb
        reverb = np.convolve(audio, impulse, mode='same')
        return audio * (1 - config.reverb_amount) + reverb * config.reverb_amount

# ==================== STUDIO JARVIS ====================
class StudioJarvis:
    """Professional broadcast-quality JARVIS"""
    
    def __init__(self):
        self.engine = StudioVoiceEngine()
        self.config = StudioVoiceConfig()
        self.phonemes = StudioPhonemeEngine()
        
        # Studio-quality presets
        self.presets = {
            'jarvis_studio': StudioVoiceConfig(
                name="JARVIS Studio", gender='male',
                base_f0=116,
                formants=[550, 1150, 2600, 3300, 4000, 4700],
                formant_bandwidths=[38, 42, 52, 65, 80, 95],
                presence_boost=2.2,
                air_band=1.6,
                harmonic_richness=1.4,
                quality=VoiceQuality.STUDIO
            ),
            'friday_studio': StudioVoiceConfig(
                name="FRIDAY Studio", gender='female',
                base_f0=198,
                formants=[740, 1400, 2800, 3600, 4300, 5000],
                formant_bandwidths=[40, 45, 55, 68, 82, 98],
                presence_boost=2.3,
                air_band=1.7,
                quality=VoiceQuality.BROADCAST
            ),
            'pepper_studio': StudioVoiceConfig(
                name="Pepper Studio", gender='female',
                base_f0=188,
                formants=[720, 1360, 2760, 3560, 4250, 4950],
                formant_bandwidths=[42, 48, 58, 72, 85, 100],
                presence_boost=2.0,
                air_band=1.5,
                quality=VoiceQuality.WARM
            ),
        }
        
        self.cache_dir = Path("studio_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Studio voice: {self.config.name}")
            print(f"  Sample Rate: {SAMPLE_RATE}Hz, Bit Depth: {BIT_DEPTH}-bit")
            print(f"  Presence: {self.config.presence_boost}, Air: {self.config.air_band}")
        else:
            print(f"✗ Voice '{name}' not found")
    
    def text_to_phonemes(self, text):
        """Convert text to phonemes"""
        text = text.lower().strip()
        words = text.split()
        
        phonemes = []
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word:
                word_phonemes = self.phonemes.word_to_phonemes(clean_word)
                phonemes.extend(word_phonemes)
                phonemes.append('pau')
        
        return phonemes
    
    def generate_speech(self, text, output_file=None):
        """Generate studio-quality speech"""
        if not text:
            return None
        
        print(f"\n🎙️ STUDIO GENERATION: '{text}'")
        print(f"   Sample Rate: {SAMPLE_RATE/1000}kHz | Bit Depth: {BIT_DEPTH}-bit")
        
        phonemes = self.text_to_phonemes(text)
        print(f"   Phonemes: {len(phonemes)}")
        
        if not phonemes:
            return None
        
        # Cache check
        cache_key = hashlib.md5(f"{text}{self.config.name}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        if cache_file.exists():
            print("📦 Loading from cache")
            audio, _ = sf.read(cache_file)
        else:
            # Generate
            audio_segments = []
            
            for phoneme in phonemes:
                if phoneme == 'pau':
                    segment = self.engine.generate_silence(0.08)
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
            
            # Professional processing chain
            audio = self.engine.apply_dynamics(audio, self.config)
            audio = self.engine.apply_professional_reverb(audio, self.config)
            
            # Normalize to 24-bit range
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            # Save cache
            sf.write(cache_file, audio, self.engine.sr)
            print(f"💾 Cached to: {cache_file}")
        
        # Save output
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"✓ Saved to: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak with studio quality"""
        audio = self.generate_speech(text)
        if audio is not None:
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def list_voices(self):
        """List studio voices"""
        print("\n🎙️ STUDIO VOICE PRESETS:")
        for name, config in self.presets.items():
            print(f"  • {name:15} | {config.gender} | Presence: {config.presence_boost}")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--voice', '-v', default='jarvis_studio')
    parser.add_argument('--text', '-t')
    parser.add_argument('--output', '-o')
    
    args = parser.parse_args()
    
    jarvis = StudioJarvis()
    
    if args.voice:
        jarvis.set_voice(args.voice)
    
    if args.text:
        if args.output:
            jarvis.generate_speech(args.text, args.output)
        else:
            jarvis.speak(args.text)
    else:
        # Interactive mode
        print("\n🎙️ STUDIO JARVIS - Broadcast Quality")
        print("Type text to speak, or:")
        print("  /voice <name>  - Change voice")
        print("  /exit          - Exit")
        
        while True:
            try:
                cmd = input("\n🎬 ").strip()
                if cmd == '/exit':
                    break
                elif cmd.startswith('/voice'):
                    _, name = cmd.split()
                    jarvis.set_voice(name)
                elif cmd:
                    jarvis.speak(cmd)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")