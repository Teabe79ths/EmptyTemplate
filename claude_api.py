import os
import json
import logging
import time
import random
import anthropic
from typing import List, Dict, Any, Optional

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja klienta Claude
HAS_ANTHROPIC = False
client = None

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
    
    # Inicjalizacja klienta Anthropic Claude
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        client = Anthropic(api_key=api_key)
        logger.info("Zainicjalizowano klienta Anthropic Claude API")
    else:
        logger.warning("Brak klucza API Anthropic (ANTHROPIC_API_KEY)")
except ImportError as e:
    logger.warning(f"Nie można zaimportować pakietu Anthropic: {str(e)}")
except Exception as e:
    logger.warning(f"Anthropic API jest niedostępne: {str(e)}")

# Cache na wyniki analizy, aby ograniczyć liczbę zapytań do API
question_cache = {}

# Standardowe wartości zastępcze dla analizy niedzialajacegp API
DEFAULT_QUESTION = "Jakie emocje towarzyszą Ci najczęściej w ciągu dnia? Potrafisz je nazwać?"

# Komunikat wyświetlany przy błędzie limitu API
API_LIMIT_MESSAGES = [
    "Wystąpił błąd podczas generowania pytania z powodu ograniczeń API.",
    "Wykorzystano dzienny limit zapytań do usługi Claude.",
    "Spróbuj ponownie później lub zapoznaj się z alternatywnie generowanymi pytaniami."
]

def generate_claude_question(context: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generuje terapeutyczne pytanie wykorzystując model Claude, które jest dopasowane 
    do kontekstu wcześniejszych odpowiedzi użytkownika.
    
    Args:
        context (list, optional): Lista poprzednich elementów konwersacji.
                                Każdy element to słownik z kluczami 'question', 'response' i 'date'.
    
    Returns:
        str: Terapeutyczne pytanie w języku polskim.
    """
    # Jeśli Claude nie jest dostępny, użyj algorytmu zastępczego
    if not HAS_ANTHROPIC or client is None:
        logger.warning("Anthropic Claude API jest niedostępne. Używam domyślnego mechanizmu generowania pytań.")
        from therapy import generate_question
        return generate_question(context)
    
    # Jeśli nie mamy kontekstu, generujemy pytanie inicjalne
    if not context or len(context) == 0:
        try:
            # Generujemy pierwsze pytanie inicjujące rozmowę
            prompt = """
            Wygeneruj jedno głębokie, refleksyjne pytanie terapeutyczne w języku polskim, które mogłoby rozpocząć rozmowę z nowym użytkownikiem aplikacji wsparcia psychologicznego. 
            Pytanie powinno być empatyczne, otwarte i zachęcające do głębszej refleksji nad sobą i swoim samopoczuciem.
            Unikaj pytań zamkniętych i powierzchownych. Pytanie powinno być napisane w drugiej osobie liczby pojedynczej (Ty).
            
            Odpowiedz tylko samym pytaniem, bez dodatkowego tekstu.
            """
            
            # Cache key for the initial question
            cache_key = "initial_question"
            
            # Check if we have a cached response
            if cache_key in question_cache:
                return question_cache[cache_key]
            
            # Call the Claude API
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024.
                max_tokens=150,
                temperature=0.7,
                system="Jesteś empatycznym polskim psychoterapeutą specjalizującym się w terapii poznawczo-behawioralnej i refleksyjnym podejściu do problemów życiowych. Twoje pytania są głębokie, wnikliwe i zachęcają do autorefleksji.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            question = message.content[0].text.strip()
            
            # Cache the response
            question_cache[cache_key] = question
            
            return question
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania pierwszego pytania z Claude: {str(e)}")
            # Fallback to a default first question
            from therapy import DEFAULT_FIRST_QUESTIONS
            return random.choice(DEFAULT_FIRST_QUESTIONS)
    
    # Prepare conversation context for Claude
    conversation_history = ""
    for i, entry in enumerate(context[-5:]):  # Use only last 5 entries for context
        if entry.get("question") and entry.get("response"):
            conversation_history += f"Pytanie: {entry['question']}\n"
            conversation_history += f"Odpowiedź użytkownika: {entry['response']}\n\n"
    
    # Generate a hash of the conversation history for caching
    cache_key = hash(conversation_history)
    
    # Check if we have a cached response for this conversation
    if cache_key in question_cache:
        return question_cache[cache_key]
    
    # Define the prompt for Claude
    prompt = f"""
    Oto fragment rozmowy terapeutycznej w języku polskim. Przeanalizuj kontekst i wygeneruj jedno kolejne, pogłębiające pytanie dla użytkownika:

    {conversation_history}

    Na podstawie powyższego kontekstu i odpowiedzi użytkownika, sformułuj jedno głębokie, wnikliwe pytanie terapeutyczne, które:
    1. Odnosi się do tematów, emocji lub wzorców widocznych w powyższych odpowiedziach
    2. Zachęca do głębszej refleksji nad sobą
    3. Jest empatyczne i pełne zrozumienia
    4. Jest sformułowane w sposób otwarty (nie może być odpowiedzią tak/nie)
    5. Nie zawiera osądów ani założeń
    
    Wygeneruj wyłącznie jedno pytanie, bez żadnego dodatkowego tekstu czy wyjaśnień.
    """
    
    try:
        # Mechanizm ponownych prób z wykładniczym opóźnieniem
        max_retries = 3
        retry_delay = 1  # początkowe opóźnienie w sekundach
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Próba generowania pytania z Claude {attempt+1}/{max_retries}")
                
                # Call the Claude API
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024.
                    max_tokens=200,
                    temperature=0.7,
                    system="Jesteś empatycznym polskim psychoterapeutą specjalizującym się w terapii poznawczo-behawioralnej i refleksyjnym podejściu do problemów życiowych. Twoje pytania są głębokie, wnikliwe i zachęcają do autorefleksji.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                question = message.content[0].text.strip()
                
                # Cache the response
                question_cache[cache_key] = question
                
                return question
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Błąd podczas generowania pytania z Claude: {error_msg}")
                
                # Check if it's a rate limit error
                if "rate_limit" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.warning("Osiągnięto limit zapytań API Claude.")
                    
                    # Fallback to our standard question generation mechanism
                    from therapy import generate_question
                    return generate_question(context)
                    
                # If this is the last retry, use the standard question generation
                if attempt == max_retries - 1:
                    logger.error("Wyczerpano limit prób Claude. Używam standardowego mechanizmu.")
                    from therapy import generate_question
                    return generate_question(context)
                    
                # Add random jitter to delay
                jitter = random.uniform(0, 0.5)
                wait_time = retry_delay * (2 ** attempt) + jitter
                logger.info(f"Ponawiam próbę za {wait_time:.2f} sekund...")
                time.sleep(wait_time)
                
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd w module Claude: {str(e)}")
        # Fallback to standard question generation
        from therapy import generate_question
        return generate_question(context)