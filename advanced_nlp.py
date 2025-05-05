"""
Moduł zaawansowanej analizy NLP do generowania bardziej kontekstowych
i spersonalizowanych pytań terapeutycznych.

Ten moduł zapewnia dostęp do zaawansowanych modeli NLP, które mogą
lepiej zrozumieć kontekst rozmowy i odpowiednio generować pytania.
"""

import os
import logging
import json
from google.cloud import aiplatform
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import random

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja Google AI
HAS_GOOGLE_AI = False
google_ai_client = None

try:
    aiplatform.init()
    google_ai_client = aiplatform.PredictionService()
    HAS_GOOGLE_AI = True
    logger.info("Zainicjalizowano klienta Google AI")
except Exception as e:
    logger.warning(f"Google AI jest niedostępne: {str(e)}")

# Cache na wyniki analizy
analysis_cache = {}

class NLPModels:
    GOOGLE_TEXT = "text-bison@002"
    GOOGLE_CHAT = "chat-bison@002"


def analyze_text(text, cache_key=None):
    """
    Analizuje tekst używając Google AI
    """
    if cache_key and cache_key in analysis_cache:
        return analysis_cache[cache_key]

    if not HAS_GOOGLE_AI:
        return {
            "analysis": "System analizy chwilowo niedostępny.",
            "sentiment": "neutral",
            "suggestions": ["Spróbuj ponownie później."]
        }

    try:
        response = google_ai_client.predict({
            "text": text,
            "task": "text_analysis",
            "model": NLPModels.GOOGLE_TEXT
        })

        analysis = {
            "analysis": response.predictions[0],
            "timestamp": datetime.now().isoformat()
        }

        if cache_key:
            analysis_cache[cache_key] = analysis

        return analysis

    except Exception as e:
        logger.error(f"Błąd podczas analizy Google AI: {str(e)}")
        return {
            "analysis": "Wystąpił błąd podczas analizy.",
            "sentiment": "neutral",
            "suggestions": ["Spróbuj ponownie później."]
        }


def analyze_emotional_state(context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Przeprowadza pogłębioną analizę stanu emocjonalnego użytkownika z uwzględnieniem:
    - Dominujących emocji i ich intensywności
    - Wzorców emocjonalnych w czasie
    - Czynników wyzwalających określone reakcje
    - Strategii radzenia sobie z emocjami
    """

    if not context or len(context) < 2:
        return {
            "dominant_emotions": [],
            "emotional_state": "neutral",
            "suggested_focus_areas": ["samoświadomość", "refleksja"]
        }

    # Przygotowanie tekstu do analizy
    conversation_text = ""
    for item in context[-5:]:  # Użyj tylko ostatnich 5 wpisów
        if item.get("question") and item.get("response"):
            conversation_text += f"Pytanie: {item['question']}\n"
            conversation_text += f"Odpowiedź: {item['response']}\n\n"

    # Cache'owanie na podstawie zawartości rozmowy
    cache_key = hash(conversation_text)
    if cache_key in analysis_cache:
        logger.info("Używam zbuforowanej analizy emocjonalnej")
        return analysis_cache[cache_key]

    # Domyślna analiza na wypadek błędów
    default_analysis = {
        "dominant_emotions": ["refleksyjność"],
        "emotional_state": "neutral",
        "suggested_focus_areas": ["samoświadomość", "refleksja"]
    }

    # Spróbuj użyć Google AI do analizy
    analysis = analyze_text(conversation_text, cache_key)
    
    # Extract relevant information from Google AI analysis (adapt as needed)
    dominant_emotions = analysis.get("analysis", {}).get("dominant_emotions", []) or ["neutral"]
    emotional_state = analysis.get("analysis", {}).get("sentiment", "neutral")
    suggested_focus_areas = analysis.get("analysis", {}).get("suggestions", ["samoświadomość"])


    final_analysis = {
        "dominant_emotions": dominant_emotions,
        "emotional_state": emotional_state,
        "suggested_focus_areas": suggested_focus_areas
    }


    return final_analysis


def generate_therapeutic_response(conversation_history):
    """
    Generuje odpowiedź terapeutyczną używając Google AI
    """
    if not HAS_GOOGLE_AI:
        return "Przepraszamy, system odpowiedzi jest chwilowo niedostępny."

    try:
        prompt = f"Historia rozmowy:\n{conversation_history}\n\nWygeneruj empatyczną i wspierającą odpowiedź terapeutyczną:"

        response = google_ai_client.predict({
            "text": prompt,
            "model": NLPModels.GOOGLE_CHAT
        })

        return response.predictions[0]

    except Exception as e:
        logger.error(f"Błąd podczas generowania odpowiedzi: {str(e)}")
        return "Przepraszamy, wystąpił błąd podczas generowania odpowiedzi."


def generate_advanced_question(context: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Generuje zaawansowane pytanie terapeutyczne z wykorzystaniem najnowszych modeli NLP.
    """
    # Jeśli nie ma kontekstu, wygeneruj pytanie inicjujące
    if not context or len(context) == 0:
        if random.random() < 0.7 and HAS_GOOGLE_AI:
            initial_question = _generate_initial_question()
            return initial_question, {"model": "advanced", "context_used": False}
        else:
            # Losowe pytanie z domyślnych
            return random.choice(DEFAULT_QUESTIONS), {"model": "default", "context_used": False}

    # Analizuj stan emocjonalny na podstawie kontekstu
    emotional_analysis = analyze_emotional_state(context)

    # Przygotowanie kontekstu rozmowy
    conversation_text = ""
    for item in context[-5:]:  # Użyj tylko ostatnich 5 wpisów
        if item.get("question") and item.get("response"):
            conversation_text += f"Pytanie: {item['question']}\n"
            conversation_text += f"Odpowiedź: {item['response']}\n"
            conversation_text += f"Data: {item['date']}\n\n"

    # Cache'owanie na podstawie zawartości rozmowy
    cache_key = hash(conversation_text)
    if cache_key in analysis_cache:
        logger.info("Używam zbuforowanego pytania")
        cached_question = analysis_cache[cache_key]
        return cached_question, {"model": "cached", "context_used": True, "emotional_analysis": emotional_analysis}

    # Dostępne stany emocjonalne i sugerowane typy pytań
    question_strategies = {
        "positive": "Zadaj pytanie, które zachęci do refleksji nad pozytywnymi aspektami życia lub doświadczeniami.",
        "negative": "Zadaj empatyczne pytanie, które pomoże w analizie trudnych emocji, ale z perspektywą konstruktywnego rozwiązania.",
        "mixed": "Zadaj pytanie, które pozwoli na zrównoważenie sprzecznych emocji i znalezienie harmonii.",
        "neutral": "Zadaj pytanie, które zgłębi tematy ważne dla osobistego rozwoju i samoświadomości."
    }

    # Domyślna strategia
    emotion_strategy = question_strategies.get(
        emotional_analysis.get("emotional_state", "neutral"),
        question_strategies["neutral"]
    )

    # Obszary sugerowane do skupienia się
    focus_areas = emotional_analysis.get("suggested_focus_areas", ["samoświadomość"])
    focus_areas_text = ", ".join(focus_areas)

    generated_question = _generate_contextual_question(
        conversation_text,
        emotion_strategy,
        focus_areas_text
    )

    # Dodaj do cache
    if generated_question:
        analysis_cache[cache_key] = generated_question

    return generated_question or random.choice(DEFAULT_QUESTIONS), {
        "model": "advanced",
        "context_used": True,
        "emotional_analysis": emotional_analysis
    }


def _generate_initial_question() -> str:
    """Generuje pierwsze pytanie terapeutyczne bez kontekstu wcześniejszej rozmowy."""

    prompt = """
    Wygeneruj jedno głębokie, refleksyjne pytanie terapeutyczne w języku polskim, 
    które mogłoby rozpocząć rozmowę z nowym użytkownikiem aplikacji wsparcia psychologicznego.

    Pytanie powinno być empatyczne, otwarte i zachęcające do głębszej refleksji nad sobą
    i swoim samopoczuciem.

    Unikaj pytań zamkniętych i powierzchownych. Pytanie powinno być napisane w drugiej
    osobie liczby pojedynczej (Ty).

    Odpowiedz tylko samym pytaniem, bez dodatkowego tekstu.
    """

    if HAS_GOOGLE_AI and google_ai_client:
        try:
            response = google_ai_client.predict({
                "text": prompt,
                "model": NLPModels.GOOGLE_CHAT
            })
            return response.predictions[0]

        except Exception as e:
            logger.error(f"Błąd podczas generowania pytania inicjującego z Google AI: {str(e)}")

    # Strategia 3: Użyj domyślnego pytania, jeśli wszystko zawiedzie
    return random.choice(DEFAULT_QUESTIONS)


def _generate_contextual_question(conversation_text: str, emotion_strategy: str, focus_areas: str) -> Optional[str]:
    """
    Generuje kontekstowe pytanie na podstawie analizy rozmowy i stanu emocjonalnego.
    """

    system_prompt = f"""
    Jesteś doświadczonym polskim psychoterapeutą prowadzącym terapeutyczną rozmowę.
    Twoim zadaniem jest wygenerowanie pojedynczego, głębokiego pytania w języku polskim,
    które będzie kontynuacją rozmowy z pacjentem.

    Na podstawie dostarczonego fragmentu rozmowy, stwórz pytanie, które:
    1. Bezpośrednio odnosi się do tematów poruszonych przez pacjenta
    2. {emotion_strategy}
    3. Skupia się na jednym lub więcej z następujących obszarów: {focus_areas}
    4. Jest sformułowane w sposób otwarty (nie może być odpowiedzią tak/nie)
    5. Nie zawiera osądów ani założeń
    6. Jest empatyczne i pełne zrozumienia

    Wygeneruj wyłącznie jedno pytanie, bez wprowadzenia ani wyjaśnień.
    """

    if HAS_GOOGLE_AI and google_ai_client:
        try:
            response = google_ai_client.predict({
                "text": system_prompt + "\n\n" + conversation_text,
                "model": NLPModels.GOOGLE_CHAT
            })
            return response.predictions[0]

        except Exception as e:
            logger.error(f"Błąd podczas generowania kontekstowego pytania z Google AI: {str(e)}")

    # Jeśli wszystko zawiedzie, zwróć None (caller powinien użyć domyślnego pytania)
    return None


# Domyślne pytania, gdy API zawiedzie lub brak kontekstu
DEFAULT_QUESTIONS = [
    "Co sprawiło Ci największą satysfakcję w ostatnim tygodniu?",
    "Jak opisałbyś/opisałabyś swój nastrój w ostatnich dniach?",
    "Jakie emocje towarzyszą Ci najczęściej w ciągu dnia?",
    "Co przede wszystkim motywuje Cię do działania?",
    "Jakie wartości są dla Ciebie najważniejsze w życiu?"
]