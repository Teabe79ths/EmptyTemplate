
import os
import logging
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAnalyzer:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 10  # nagrywanie przez 10 sekund
        
    def record_audio(self):
        """Nagrywa głos użytkownika"""
        try:
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1
            )
            sd.wait()
            return recording
            
        except Exception as e:
            logger.error(f"Błąd podczas nagrywania: {str(e)}")
            return None

    def analyze_emotion(self, audio_data):
        """Analizuje emocje w głosie"""
        try:
            # Ekstrakcja cech audio
            mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=self.sample_rate)
            
            # Przykładowa klasyfikacja (można rozszerzyć o bardziej zaawansowany model)
            emotional_features = {
                "tone_variation": np.std(mfccs),
                "energy": np.mean(np.abs(audio_data)),
                "pitch": np.mean(librosa.pitch.piptrack(y=audio_data.flatten(), sr=self.sample_rate)[0])
            }
            
            # Podstawowa interpretacja
            emotion = self._interpret_features(emotional_features)
            return emotion
            
        except Exception as e:
            logger.error(f"Błąd podczas analizy emocji: {str(e)}")
            return {"error": str(e)}

    def _interpret_features(self, features):
        """Interpretuje cechy głosu"""
        emotion_state = {
            "primary_emotion": "neutral",
            "confidence": 0.0,
            "voice_characteristics": []
        }
        
        # Przykładowa logika interpretacji
        if features["energy"] > 0.7:
            emotion_state["primary_emotion"] = "excited/angry"
            emotion_state["confidence"] = 0.8
        elif features["tone_variation"] > 0.5:
            emotion_state["primary_emotion"] = "emotional"
            emotion_state["confidence"] = 0.6
        else:
            emotion_state["primary_emotion"] = "calm"
            emotion_state["confidence"] = 0.7
            
        return emotion_state
