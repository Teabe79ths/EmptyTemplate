
import nltk
import re
from collections import defaultdict
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Słownik emocji i powiązanych słów (w języku polskim)
EMOTION_WORDS = {
    'radość': ['szczęśliwy', 'radosny', 'wesoły', 'zadowolony', 'uśmiechnięty', 'entuzjastyczny'],
    'smutek': ['smutny', 'przygnębiony', 'zmartwiony', 'zrozpaczony', 'załamany'],
    'złość': ['zły', 'wściekły', 'zirytowany', 'poirytowany', 'rozzłoszczony'],
    'strach': ['przestraszony', 'przerażony', 'zaniepokojony', 'wystraszony'],
    'zaskoczenie': ['zaskoczony', 'zdziwiony', 'zdumiony', 'oszołomiony'],
    'spokój': ['spokojny', 'zrelaksowany', 'wyciszony', 'opanowany']
}

class TextEmotionAnalyzer:
    def __init__(self):
        self.emotion_words = EMOTION_WORDS
        
    def analyze_text(self, text):
        """Analizuje tekst pod kątem wyrażanych emocji"""
        if not text:
            return {
                'dominant_emotion': None,
                'emotion_scores': {},
                'emotion_intensity': 0.0
            }
            
        # Normalizacja tekstu
        text = text.lower()
        
        # Liczenie wystąpień słów związanych z emocjami
        emotion_counts = defaultdict(int)
        words = text.split()
        
        for word in words:
            for emotion, emotion_words in self.emotion_words.items():
                if word in emotion_words:
                    emotion_counts[emotion] += 1
                    
        # Obliczanie wyników
        total_emotional_words = sum(emotion_counts.values())
        emotion_scores = {}
        
        for emotion in self.emotion_words.keys():
            count = emotion_counts[emotion]
            score = count / len(words) if len(words) > 0 else 0
            emotion_scores[emotion] = round(score * 100, 2)
            
        # Określanie dominującej emocji
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else None
        
        # Obliczanie intensywności emocjonalnej
        emotion_intensity = (total_emotional_words / len(words)) * 100 if len(words) > 0 else 0
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_scores': emotion_scores,
            'emotion_intensity': round(emotion_intensity, 2)
        }
        
    def get_emotional_phrases(self, text):
        """Wyodrębnia frazy zawierające wyrażenia emocjonalne"""
        emotional_phrases = []
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            for emotion, words in self.emotion_words.items():
                if any(word in sentence.lower() for word in words):
                    emotional_phrases.append({
                        'phrase': sentence,
                        'emotion': emotion
                    })
                    
        return emotional_phrases

    def analyze_emotional_changes(self, texts):
        """Analizuje zmiany emocjonalne w serii tekstów"""
        analyses = [self.analyze_text(text) for text in texts]
        
        emotional_progression = {
            'emotion_changes': [],
            'overall_trend': None
        }
        
        # Śledź zmiany dominujących emocji
        prev_emotion = None
        for analysis in analyses:
            current_emotion = analysis['dominant_emotion']
            if prev_emotion and current_emotion and prev_emotion != current_emotion:
                emotional_progression['emotion_changes'].append(
                    f"Zmiana z {prev_emotion} na {current_emotion}"
                )
            prev_emotion = current_emotion
            
        # Określ ogólny trend
        if len(analyses) >= 2:
            first_intensity = analyses[0]['emotion_intensity']
            last_intensity = analyses[-1]['emotion_intensity']
            
            if last_intensity > first_intensity:
                emotional_progression['overall_trend'] = 'wzrastający'
            elif last_intensity < first_intensity:
                emotional_progression['overall_trend'] = 'malejący'
            else:
                emotional_progression['overall_trend'] = 'stabilny'
                
        return emotional_progression
