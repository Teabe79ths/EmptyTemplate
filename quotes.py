
import random
import logging
from anthropic import Anthropic
import os

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja klienta Anthropic
anthropic_client = None
try:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        anthropic_client = Anthropic(api_key=api_key)
except Exception as e:
    logger.warning(f"Nie można zainicjalizować Anthropic API: {str(e)}")

# Domyślna pula cytatów
DEFAULT_QUOTES = [
    "Każda podróż zaczyna się od pierwszego kroku.",
    "W ciszy odnajdujemy własne odpowiedzi.",
    "Zmiana zaczyna się od akceptacji tego, co jest.",
    "Twoje myśli kształtują Twoją rzeczywistość.",
    "Uważność to klucz do zrozumienia siebie.",
    "Nie możesz zmienić fal, ale możesz nauczyć się surfować.",
    "Największą przeszkodą w życiu jesteśmy my sami.",
    "Twoja wartość nie zależy od opinii innych.",
    "Każdy dzień to nowa szansa na zmianę.",
    "Siła nie pochodzi z tego, co potrafisz zrobić. Pochodzi z pokonywania tego, czego zrobić nie mogłeś.",
    "Szczęście to nie cel, to sposób podróżowania.",
    "Twoje granice są początkiem Ciebie, nie końcem.",
    "Akceptacja nie oznacza rezygnacji, oznacza zrozumienie.",
    "Jesteś silniejszy niż Ci się wydaje.",
    "Każda emocja jest ważna i ma swój cel.",
    "Zmiana perspektywy zmienia rzeczywistość.",
    "Trudności to okazje do rozwoju.",
    "Spokój umysłu zaczyna się od akceptacji teraźniejszości.",
    "Twoja historia jest ważna, ale nie definiuje Twojej przyszłości.",
    "Małe kroki też prowadzą do wielkich zmian.",
    "Czasem trzeba się zgubić, żeby się odnaleźć.",
    "Twoje uczucia są ważne i zasługują na wysłuchanie.",
    "Każdy moment to szansa na nowy początek.",
    "W głębi siebie znasz odpowiedź.",
    "Rozwój osobisty to podróż, nie cel.",
    "Twoje doświadczenia Cię kształtują, ale nie ograniczają.",
    "Najważniejsza relacja to ta z samym sobą.",
    "Życie nie musi być perfekcyjne, żeby być piękne.",
    "Każdy dzień to nowa szansa na lepsze wybory.",
    "Twoja wrażliwość to Twoja siła."
]

quote_cache = {}

def generate_therapeutic_quote(context=None):
    """
    Generuje lub wybiera terapeutyczny cytat.
    
    Args:
        context (str, optional): Kontekst dla generowania cytatu
        
    Returns:
        str: Terapeutyczny cytat
    """
    if not anthropic_client:
        return random.choice(DEFAULT_QUOTES)
        
    try:
        prompt = """
        Wygeneruj jeden krótki, mądry cytat terapeutyczny w języku polskim.
        Cytat powinien być inspirujący, głęboki i związany z samorozwojem, 
        ale nie dłuższy niż jedno zdanie.
        
        Odpowiedz tylko samym cytatem, bez cudzysłowów czy dodatkowego tekstu.
        """
        
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        quote = message.content[0].text.strip()
        return quote
        
    except Exception as e:
        logger.error(f"Błąd podczas generowania cytatu: {str(e)}")
        return random.choice(DEFAULT_QUOTES)
