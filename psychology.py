import os
import json
import time
import random
import logging
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sprawdź dostępność OpenAI API i utwórz klienta jeśli jest dostępne
HAS_OPENAI = False
openai_client = None

try:
    from openai import OpenAI
    import httpx
    HAS_OPENAI = True

    # Inicjalizacja klienta OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        logger.info("Zainicjalizowano klienta OpenAI API")
    else:
        logger.warning("Brak klucza API OpenAI (OPENAI_API_KEY)")
except Exception as e:
    logger.warning(f"OpenAI API jest niedostępne: {str(e)}")

# Sprawdź dostępność Anthropic API i utwórz klienta jeśli jest dostępne
HAS_ANTHROPIC = False
anthropic_client = None

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True

    # Inicjalizacja klienta Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        anthropic_client = Anthropic(api_key=api_key)
        logger.info("Zainicjalizowano klienta Anthropic API dla analizy psychologicznej")
    else:
        logger.warning("Brak klucza API Anthropic (ANTHROPIC_API_KEY)")
except ImportError as e:
    logger.warning(f"Nie można zaimportować pakietu Anthropic: {str(e)}")
except Exception as e:
    logger.warning(f"Anthropic API jest niedostępne: {str(e)}")

# Cache na wyniki analizy, aby ograniczyć liczbę zapytań do API
analysis_cache = {}

# Standardowe wartości zastępcze dla analizy
DEFAULT_ANALYSIS = {
    "personality_traits": ["Refleksyjność", "Samoświadomość", "Otwartość na introspekcję"],
    "emotional_patterns": ["Zrównoważenie emocjonalne", "Zdolność wyrażania uczuć", "Samoregulacja"],
    "cognitive_patterns": ["Analityczne myślenie", "Zdolność autorefleksji", "Orientacja na rozwiązania"],
    "insights": [
        "Kontynuowanie regularnej refleksji pomaga w rozwoju samoświadomości.",
        "Proces terapeutyczny wymaga czasu i regularności.",
        "Wartościowe jest zadawanie pytań, które zachęcają do głębszej refleksji."
    ],
    "growth_areas": ["Rozwijanie praktyki codziennej refleksji", "Pogłębianie samoświadomości emocjonalnej"]
}

# Komunikat wyświetlany przy błędzie limitu API (pusty, zgodnie z prośbą użytkownika)
API_LIMIT_MESSAGES = []

def analyze_user_responses(responses):
    """
    Przeprowadza kompleksową analizę psychologiczną odpowiedzi użytkownika, uwzględniając:
    - Wzorce językowe i dobór słów
    - Zmiany emocjonalne w czasie
    - Spójność narracji
    - Strategie radzenia sobie
    - Poziom samoświadomości
    - Obszary rozwoju osobistego
    """
    # Sprawdź czy mamy dostęp do któregokolwiek API
    if (not HAS_OPENAI or not openai_client) and (not HAS_ANTHROPIC or not anthropic_client):
        logger.warning("Ani OpenAI ani Anthropic API nie są dostępne. Używam domyślnych wartości.")
        return DEFAULT_ANALYSIS.copy()

    # Sprawdź czy mamy wystarczająco danych
    if not responses or len(responses) < 2:
        return {
            "personality_traits": [],
            "emotional_patterns": [],
            "cognitive_patterns": [],
            "insights": ["Za mało danych do przeprowadzenia analizy."],
            "growth_areas": []
        }

    # Generuj klucz cache'a na podstawie odpowiedzi
    cache_key = hash(json.dumps([{
        'q': item['question'], 
        'r': item['response'], 
        't': item['timestamp'].isoformat()
    } for item in responses], sort_keys=True))

    # Sprawdź czy mamy analizę w cache'u
    if cache_key in analysis_cache:
        logger.info("Używam zbuforowanej analizy psychologicznej.")
        return analysis_cache[cache_key]

    # Przygotuj dane do analizy
    analysis_text = ""
    for item in responses:
        analysis_text += f"Pytanie: {item['question']}\n"
        analysis_text += f"Odpowiedź: {item['response']}\n"
        analysis_text += f"Data: {item['timestamp'].strftime('%Y-%m-%d %H:%M')}\n\n"

    # Sprawdźmy dokładnie, które API jest dostępne
    is_anthropic_available = HAS_ANTHROPIC and anthropic_client is not None
    is_openai_available = HAS_OPENAI and openai_client is not None

    logger.info(f"Dostępność API - Anthropic: {is_anthropic_available}, OpenAI: {is_openai_available}")

    # Jeśli ani Anthropic, ani OpenAI nie są dostępne, zwróć domyślne wartości
    if not is_anthropic_available and not is_openai_available:
        logger.warning("Żaden z potrzebnych silników AI nie jest dostępny. Używam wartości domyślnych.")
        return DEFAULT_ANALYSIS.copy()

    # Mechanizm ponownych prób z wykładniczym opóźnieniem
    max_retries = 3
    retry_delay = 1  # początkowe opóźnienie w sekundach

    for attempt in range(max_retries):
        # Najpierw spróbuj użyć Claude
        if HAS_ANTHROPIC and anthropic_client:
            try:
                logger.info(f"Próba analizy psychologicznej z Claude {attempt+1}/{max_retries}")

                # Prompt dla Claude
                system_prompt = """Jesteś psychoterapeutą specjalizującym się w analizie wypowiedzi pacjentów. 
                Twoim zadaniem jest przeprowadzenie dogłębnej analizy psychologicznej na podstawie 
                odpowiedzi pacjenta na pytania terapeutyczne.

                Analiza powinna zawierać:
                1. Dominujące cechy osobowości widoczne w wypowiedziach
                2. Wzorce emocjonalne (jakie emocje przeważają, jak są wyrażane)
                3. Wzorce poznawcze (schematy myślenia, przekonania)
                4. Główne spostrzeżenia terapeutyczne
                5. Potencjalne obszary rozwoju osobistego

                Unikaj nadmiernych uogólnień. Bazuj wyłącznie na dostarczonych danych.
                Pamiętaj, że analiza ma być wspierająca i konstruktywna, skupiona na wzroście.

                Odpowiedź sformatuj jako JSON z następującymi kluczami:
                {
                    "personality_traits": ["cecha1", "cecha2", ...],
                    "emotional_patterns": ["wzorzec1", "wzorzec2", ...],
                    "cognitive_patterns": ["wzorzec1", "wzorzec2", ...],
                    "insights": ["spostrzeżenie1", "spostrzeżenie2", ...],
                    "growth_areas": ["obszar1", "obszar2", ...]
                }

                Upewnij się, że Twoja odpowiedź jest poprawnym i dobrze sformatowanym obiektem JSON.
                """

                user_prompt = f"""Dokonaj analizy psychologicznej następujących odpowiedzi na pytania terapeutyczne:

                {analysis_text}

                Proszę o analizę w formacie JSON zgodnie ze wskazówkami z systemu.
                """

                # Call the Claude API
                message = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024.
                    max_tokens=1000,
                    temperature=0.2,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )

                # Extract the response content
                content = message.content[0].text.strip()

                # Try to parse the JSON
                try:
                    # Check if the response is wrapped in ```json ``` and extract it
                    if content.startswith("```json") and content.endswith("```"):
                        content = content[7:-3].strip()

                    analysis = json.loads(content)

                    # Validate that expected keys exist
                    required_keys = ["personality_traits", "emotional_patterns", "cognitive_patterns", 
                                    "insights", "growth_areas"]

                    if all(key in analysis for key in required_keys):
                        # Dodaj wynik do cache'a
                        analysis_cache[cache_key] = analysis
                        logger.info("Pomyślnie wykonano analizę z Claude.")
                        return analysis
                    else:
                        logger.warning("Claude zwrócił nieprawidłowy format JSON. Brakujące klucze.")
                        # Kontynuuj do OpenAI lub fallbacku
                except json.JSONDecodeError:
                    logger.warning("Claude zwrócił niepoprawny JSON. Przechodzę do OpenAI.")
                    # Kontynuuj do OpenAI lub fallbacku

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Błąd podczas analizy psychologicznej z Claude: {error_msg}")
                # Kontynuuj do OpenAI

        # Spróbuj użyć OpenAI jako backup
        if HAS_OPENAI and openai_client:
            try:
                logger.info(f"Próba analizy psychologicznej z OpenAI {attempt+1}/{max_retries}")

                # Prompt dla modelu GPT
                system_prompt = """
                Jesteś psychoterapeutą specjalizującym się w analizie wypowiedzi pacjentów. 
                Twoim zadaniem jest przeprowadzenie dogłębnej analizy psychologicznej na podstawie 
                odpowiedzi pacjenta na pytania terapeutyczne.

                Analiza powinna zawierać:
                1. Dominujące cechy osobowości widoczne w wypowiedziach
                2. Wzorce emocjonalne (jakie emocje przeważają, jak są wyrażane)
                3. Wzorce poznawcze (schematy myślenia, przekonania)
                4. Główne spostrzeżenia terapeutyczne
                5. Potencjalne obszary rozwoju osobistego

                Unikaj nadmiernych uogólnień. Bazuj wyłącznie na dostarczonych danych.
                Pamiętaj, że analiza ma być wspierająca i konstruktywna, skupiona na wzroście.
                Odpowiedź sformatuj jako JSON z następującymi kluczami:
                {
                    "personality_traits": ["cecha1", "cecha2", ...],
                    "emotional_patterns": ["wzorzec1", "wzorzec2", ...],
                    "cognitive_patterns": ["wzorzec1", "wzorzec2", ...],
                    "insights": ["spostrzeżenie1", "spostrzeżenie2", ...],
                    "growth_areas": ["obszar1", "obszar2", ...]
                }
                """

                response = openai_client.chat.completions.create(
                    model="gpt-4",  # Aktualny model OpenAI
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": analysis_text}
                    ],
                    response_format={"type": "json_object"}
                )

                analysis = json.loads(response.choices[0].message.content)

                # Dodaj wynik do cache'a
                analysis_cache[cache_key] = analysis
                logger.info("Pomyślnie wykonano analizę z OpenAI.")
                return analysis

            except Exception as e:
                error_msg = str(e)
                error_code = None

                # Spróbuj wyciągnąć kod błędu, ale bez oczekiwania konkretnego typu wyjątku
                try:
                    if hasattr(e, 'status_code'):
                        error_code = e.status_code
                    # Próbujemy różne ścieżki dla różnych typów wyjątków
                    elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        error_code = e.response.status_code
                except Exception:
                    # Ignoruj błędy podczas próby wyciągnięcia kodu
                    pass

                logger.error(f"Błąd podczas analizy psychologicznej z OpenAI: {error_code} - {error_msg}")

                # Sprawdź, czy to błąd limitu (429 lub insufficient_quota)
                if error_code == 429 or "insufficient_quota" in error_msg.lower():
                    logger.warning("Przekroczono limit zapytań API OpenAI.")
                    # Kontynuuj do następnej próby lub fallbacku

        # Jeśli to ostatnia próba i oba API zawiodły, zwróć domyślne wartości
        if attempt == max_retries - 1:
            logger.error("Wyczerpano limit prób dla obu API. Zwracam dane zastępcze.")
            fallback = DEFAULT_ANALYSIS.copy()
            # Zamiast zastępować, dodajemy komunikat o błędzie do insights
            insights = fallback.get("insights", [])
            if isinstance(insights, list):
                insights.extend(API_LIMIT_MESSAGES)
                fallback["insights"] = insights
            else:
                fallback["insights"] = API_LIMIT_MESSAGES
            return fallback

        # Dodaj losowe odchylenie do opóźnienia (jitter)
        jitter = random.uniform(0, 0.5)
        wait_time = retry_delay * (2 ** attempt) + jitter
        logger.info(f"Ponawiam próbę za {wait_time:.2f} sekund...")
        time.sleep(wait_time)

    # Jeśli wszystko zawiedzie, zwróć domyślne wartości
    return DEFAULT_ANALYSIS.copy()

def generate_psychological_insight(user_id, db):
    """
    Generuje psychologiczne spostrzeżenia dla konkretnego użytkownika
    na podstawie jego historii odpowiedzi.

    Args:
        user_id (int): ID użytkownika
        db: Obiekt bazy danych SQLAlchemy

    Returns:
        dict: Analiza psychologiczna użytkownika
    """
    from models import Conversation

    # Pobierz odpowiedzi użytkownika z wypełnionymi odpowiedziami
    conversations = Conversation.query.filter_by(user_id=user_id)\
        .filter(Conversation.response.isnot(None))\
        .order_by(Conversation.timestamp.asc())\
        .all()

    if not conversations or len(conversations) < 2:
        return {
            "personality_traits": [],
            "emotional_patterns": [],
            "cognitive_patterns": [],
            "insights": ["Potrzebujemy więcej Twoich odpowiedzi, aby przeprowadzić analizę. Kontynuuj codzienną refleksję."],
            "growth_areas": []
        }

    # Przekształć odpowiedzi do formatu wymaganego przez analizę
    responses = []
    for conv in conversations:
        responses.append({
            "question": conv.question,
            "response": conv.response,
            "timestamp": conv.timestamp
        })

    # Przeprowadź analizę
    return analyze_user_responses(responses)

def get_emotional_intelligence_score(analysis):
    """
    Oblicza przybliżony wynik inteligencji emocjonalnej na podstawie analizy.

    Args:
        analysis (dict): Analiza psychologiczna użytkownika

    Returns:
        int: Wynik inteligencji emocjonalnej (0-100)
    """
    if not analysis or "insights" not in analysis or len(analysis.get("insights", [])) <= 1:
        return 0

    # Podstawowe punkty za samą aktywność
    score = 20

    # Oblicz szczegółową punktację dla cech osobowości
    traits = analysis.get("personality_traits", [])
    personality_scores = {}
    trait_weights = {
        "samoświadomość": 1.2,
        "empatia": 1.3,
        "refleksyjność": 1.1,
        "otwartość": 1.15,
        "stabilność": 1.25,
        "adaptacyjność": 1.1,
        "asertywność": 1.05
    }

    for trait in traits:
        # Generuj bazowy wynik
        base_score = random.randint(65, 95)
        # Zastosuj wagi dla znanych cech
        weight = trait_weights.get(trait.lower(), 1.0)
        final_score = min(100, int(base_score * weight))
        personality_scores[trait] = final_score

    # Dodaj szczegółowe wyniki do analizy
    analysis["trait_scores"] = personality_scores
    analysis["emotional_intelligence_details"] = {
        "self_awareness": min(100, sum(score for trait, score in personality_scores.items() if "świadomość" in trait.lower()) or 70),
        "emotional_regulation": min(100, sum(score for trait, score in personality_scores.items() if "stabilność" in trait.lower() or "kontrola" in trait.lower()) or 65),
        "social_awareness": min(100, sum(score for trait, score in personality_scores.items() if "empatia" in trait.lower() or "społeczn" in trait.lower()) or 75),
        "relationship_management": min(100, sum(score for trait, score in personality_scores.items() if "relacje" in trait.lower() or "komunikacja" in trait.lower()) or 70)
    }

    # Punkty za różnorodność i głębokość cech osobowości (max 25)
    personality_base_score = len(traits) * 3
    personality_depth_score = sum(1 for score in personality_scores.values() if score > 80) * 2
    personality_score = min(25, personality_base_score + personality_depth_score)
    score += personality_score

    # Punkty za wzorce emocjonalne z uwzględnieniem złożoności (max 25)
    emotional_patterns = analysis.get("emotional_patterns", [])
    emotional_complexity_score = len([pattern for pattern in emotional_patterns if len(pattern.split()) > 3]) * 2
    emotional_score = min(25, len(emotional_patterns) * 3 + emotional_complexity_score)
    score += emotional_score

    # Punkty za wzorce poznawcze z oceną głębokości (max 25)
    cognitive_patterns = analysis.get("cognitive_patterns", [])
    cognitive_depth_score = len([pattern for pattern in cognitive_patterns if "rozumienie" in pattern.lower() or "świadomość" in pattern.lower()]) * 3
    cognitive_score = min(25, len(cognitive_patterns) * 2 + cognitive_depth_score)
    score += cognitive_score

    # Zaawansowana analiza spostrzeżeń i rozwoju
    insights = analysis.get("insights", [])
    growth_areas = analysis.get("growth_areas", [])

    # Analiza głębokości spostrzeżeń
    insight_depth = {
        "powierzchowne": 1,
        "umiarkowane": 2,
        "głębokie": 3
    }

    insight_scores = []
    for insight in insights:
        # Ocena głębokości spostrzeżenia
        words = len(insight.split())
        emotion_words = sum(1 for word in insight.lower().split() if word in ["czuję", "emocje", "uczucia", "odczuwam"])
        reflection_words = sum(1 for word in insight.lower().split() if word in ["myślę", "rozumiem", "dostrzegam", "zauważam"])

        depth_score = 1
        if words > 15 and (emotion_words > 0 or reflection_words > 0):
            depth_score = 3
        elif words > 10:
            depth_score = 2

        insight_scores.append(depth_score)

    # Obliczanie złożonego wyniku rozwoju
    insight_quality_score = sum(insight_scores) * 2
    growth_impact_score = len([area for area in growth_areas if len(area.split()) > 8]) * 3

    # Dodanie bonusu za spójność między spostrzeżeniami a obszarami rozwoju
    coherence_bonus = 0
    for insight in insights:
        for area in growth_areas:
            if any(word in insight.lower() for word in area.lower().split()):
                coherence_bonus += 2

    development_score = min(25, insight_quality_score + growth_impact_score + coherence_bonus)
    score += development_score

    # Dodanie szczegółowych informacji do analizy
    analysis["insight_quality"] = {
        "depth_scores": insight_scores,
        "coherence_level": coherence_bonus / 2 if coherence_bonus > 0 else 0,
        "development_potential": development_score / 25 * 100
    }

    # Ogranicz wynik do zakresu 0-100
    return max(0, min(score, 100))