import io
import base64
import re
import json
from collections import Counter
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Polskie stop words (i angielskie jako fallback)
from nltk.corpus import stopwords
try:
    STOPWORDS = set(stopwords.words('polish'))
except:
    # Fallback na angielskie stopwords, jeśli polskie nie są dostępne
    STOPWORDS = set(stopwords.words('english'))

# Dodatkowe stop words dla języka polskiego
ADDITIONAL_STOPWORDS = {
    'aby', 'ale', 'być', 'czy', 'dla', 'gdy', 'jak', 'jest', 'mieć',
    'nie', 'się', 'że', 'bardzo', 'dzisiaj', 'dziś', 'kiedy', 'mój',
    'też', 'tak', 'tego', 'tym', 'tego', 'tej', 'ten', 'ta', 'to', 'mi', 'mnie',
    'oraz', 'może', 'której', 'którym', 'których', 'który', 'która', 'które',
    'z', 'w', 'na', 'o', 'za', 'pod', 'nad', 'przy', 'po', 'od', 'do', 'przez'
}

# Łączymy wszystkie stop words
ALL_STOPWORDS = STOPWORDS.union(ADDITIONAL_STOPWORDS)

def preprocess_text(text):
    """
    Oczyszcza tekst, usuwając znaki specjalne, cyfry i przekształcając na małe litery.
    
    Args:
        text (str): Tekst do przetworzenia
    
    Returns:
        str: Przetworzony tekst
    """
    # Zamień na małe litery
    text = text.lower()
    
    # Usuń znaki specjalne i cyfry
    text = re.sub(r'[^a-ząćęłńóśźż\s]', '', text)
    
    # Usuń nadmiarowe białe znaki
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(responses, max_words=100):
    """
    Analizuje odpowiedzi użytkownika, aby zidentyfikować najczęściej używane słowa.
    
    Args:
        responses (list): Lista odpowiedzi użytkownika
        max_words (int): Maksymalna liczba słów do zwrócenia
    
    Returns:
        dict: Słownik {słowo: liczba_wystąpień}
    """
    # Połącz wszystkie odpowiedzi
    all_text = ' '.join([r['response'] for r in responses if r['response']])
    
    # Przetwórz tekst
    processed_text = preprocess_text(all_text)
    
    # Podziel na słowa
    words = processed_text.split()
    
    # Odfiltruj stop words
    filtered_words = [word for word in words if word not in ALL_STOPWORDS and len(word) > 2]
    
    # Zlicz wystąpienia słów
    word_counts = Counter(filtered_words)
    
    # Zwróć najczęstsze słowa
    return dict(word_counts.most_common(max_words))

def generate_wordcloud(word_frequencies, width=800, height=400):
    """
    Generuje chmurę tagów na podstawie częstotliwości słów.
    
    Args:
        word_frequencies (dict): Słownik {słowo: liczba_wystąpień}
        width (int): Szerokość obrazu
        height (int): Wysokość obrazu
    
    Returns:
        str: Zakodowany w base64 obraz chmury tagów
    """
    if not word_frequencies:
        return None
    
    # Tworzymy wykres chmury tagów
    wordcloud = WordCloud(
        width=width, 
        height=height,
        background_color='#0d1117',  # Ciemne tło pasujące do motywu
        colormap='viridis',  # Kolorowa paleta
        max_words=100,
        prefer_horizontal=0.9,  # 90% słów poziomo
        scale=3,  # Wysoka rozdzielczość
        min_font_size=10,
        max_font_size=200,
        random_state=42  # Dla powtarzalności
    )
    
    # Generujemy chmurę tagów
    wordcloud.generate_from_frequencies(word_frequencies)
    
    # Tworzymy wykres matplotlib
    plt.figure(figsize=(width/100, height/100), facecolor='#0d1117')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Zapisujemy do bufora
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0d1117', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    
    # Kodujemy do base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64
    
def analyze_user_responses_keywords(user_id, db):
    """
    Analizuje odpowiedzi użytkownika, aby zidentyfikować najczęściej używane słowa
    i generuje chmurę tagów.
    
    Args:
        user_id (int): ID użytkownika
        db: Obiekt bazy danych SQLAlchemy
    
    Returns:
        dict: Wyniki analizy zawierające:
            - top_keywords (dict): Najczęściej używane słowa i ich częstotliwość
            - wordcloud_image (str): Zakodowany w base64 obraz chmury tagów
    """
    from models import Conversation
    
    # Pobierz odpowiedzi użytkownika z wypełnionymi odpowiedziami
    conversations = Conversation.query.filter_by(user_id=user_id)\
        .filter(Conversation.response.isnot(None))\
        .order_by(Conversation.timestamp.asc())\
        .all()
    
    if not conversations:
        return {
            "top_keywords": {},
            "wordcloud_image": None,
            "message": "Potrzebujemy więcej Twoich odpowiedzi, aby przeprowadzić analizę słów kluczowych."
        }
    
    # Przekształć odpowiedzi do formatu wymaganego przez analizę
    responses = []
    for conv in conversations:
        responses.append({
            "question": conv.question,
            "response": conv.response,
            "timestamp": conv.timestamp
        })
    
    # Wyodrębnij najczęściej używane słowa
    word_frequencies = extract_keywords(responses)
    
    # Wygeneruj chmurę tagów
    wordcloud_image = generate_wordcloud(word_frequencies)
    
    # Przygotuj wynik
    top_keywords = dict(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:30])
    
    return {
        "top_keywords": top_keywords,
        "wordcloud_image": wordcloud_image,
        "message": None
    }