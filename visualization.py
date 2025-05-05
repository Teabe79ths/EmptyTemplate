import matplotlib.pyplot as plt
import matplotlib
import base64
import io
import json
from datetime import datetime, timedelta
import numpy as np

# Ustawienie nieinteraktywnego backendu
matplotlib.use('Agg')

def generate_emotion_chart(psychological_analyses, days=30):
    """
    Generuje zaawansowane wykresy pokazujące:
    - Trendy emocjonalne w czasie
    - Korelacje między różnymi stanami emocjonalnymi
    - Wzorce dobowe/tygodniowe
    - Punkty przełomowe w rozwoju emocjonalnym
    - Porównanie z poprzednimi okresami
    """
    """
    Generuje wykres zmian emocjonalnych na podstawie analizy psychologicznej.
    
    Args:
        psychological_analyses: Lista obiektów PsychologicalAnalysis
        days: Liczba dni do uwzględnienia w wykresie
    
    Returns:
        str: Zakodowany w base64 obraz wykresu
    """
    # Sprawdź czy mamy wystarczająco danych
    if not psychological_analyses or len(psychological_analyses) < 2:
        return None
    
    # Filtruj analizy z ostatnich X dni
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_analyses = [a for a in psychological_analyses if a.timestamp >= cutoff_date]
    
    if not recent_analyses:
        recent_analyses = psychological_analyses[-5:]  # Ostatnie 5 analiz, jeśli nie ma z ostatnich X dni
    
    # Sortuj analizy chronologicznie
    recent_analyses.sort(key=lambda x: x.timestamp)
    
    # Przygotuj dane
    dates = [a.timestamp for a in recent_analyses]
    ei_scores = [a.emotional_intelligence_score for a in recent_analyses]
    
    # Zbierz dane o emocjach
    emotions = {'Radość': [], 'Smutek': [], 'Lęk': [], 'Gniew': [], 'Zaskoczenie': []}
    
    for analysis in recent_analyses:
        data = analysis.get_analysis()
        if not data or 'emotional_patterns' not in data:
            continue
            
        # Szukamy wzorców emocjonalnych w danych
        for pattern in data['emotional_patterns']:
            for emotion in emotions.keys():
                if emotion.lower() in pattern.lower():
                    # Dodajemy szacunkową intensywność emocji (1-5)
                    intensity = 3  # Domyślna wartość
                    
                    # Próbujemy wyodrębnić liczbę z tekstu (np. "Wysoki poziom lęku (4/5)")
                    import re
                    match = re.search(r'(\d+)[/](\d+)', pattern)
                    if match:
                        try:
                            numerator = int(match.group(1))
                            denominator = int(match.group(2))
                            intensity = numerator
                        except ValueError:
                            pass
                            
                    emotions[emotion].append((analysis.timestamp, intensity))
    
    # Inicjalizuj wykres
    plt.figure(figsize=(10, 6))
    
    # Wykres inteligencji emocjonalnej
    plt.plot(dates, ei_scores, 'o-', label='Inteligencja Emocjonalna', color='purple', linewidth=2)
    
    # Dodaj wykresy dla emocji, dla których mamy dane
    colors = {'Radość': 'green', 'Smutek': 'blue', 'Lęk': 'orange', 'Gniew': 'red', 'Zaskoczenie': 'cyan'}
    
    for emotion, data_points in emotions.items():
        if data_points:
            emotion_dates = [d[0] for d in data_points]
            emotion_values = [d[1] for d in data_points]
            if len(emotion_dates) > 1:  # Tylko jeśli mamy więcej niż jeden punkt danych
                plt.plot(emotion_dates, emotion_values, 'o--', label=emotion, color=colors[emotion], alpha=0.7)
    
    # Formatowanie wykresu
    plt.title('Rozwój Emocjonalny w Czasie', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Intensywność / Wynik', fontsize=12)
    plt.ylim(0, 5.5)  # Skala od 0 do 5, z małym marginesem na górze
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Formatowanie osi x
    plt.gcf().autofmt_xdate()
    
    # Zapisz wykres do bufora
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Koduj obraz do base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64

def generate_emotional_intelligence_progress(psychological_analyses):
    """
    Generuje wykres postępu inteligencji emocjonalnej.
    
    Args:
        psychological_analyses: Lista obiektów PsychologicalAnalysis
    
    Returns:
        str: Zakodowany w base64 obraz wykresu
    """
    if not psychological_analyses or len(psychological_analyses) < 2:
        return None
    
    # Sortuj analizy chronologicznie
    analyses = sorted(psychological_analyses, key=lambda x: x.timestamp)
    
    # Przygotuj dane
    dates = [a.timestamp for a in analyses]
    ei_scores = [a.emotional_intelligence_score for a in analyses]
    
    # Inicjalizuj wykres
    plt.figure(figsize=(10, 6))
    
    # Wykres inteligencji emocjonalnej
    plt.plot(dates, ei_scores, 'o-', color='purple', linewidth=2)
    
    # Dodaj linię trendu
    if len(dates) > 2:
        x = np.array([(d - dates[0]).total_seconds() for d in dates])
        y = np.array(ei_scores)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        trend_x = np.array([min(x), max(x)])
        trend_dates = [dates[0] + timedelta(seconds=float(val)) for val in trend_x]
        plt.plot(trend_dates, p(trend_x), "r--", alpha=0.8, label='Trend')
    
    # Formatowanie wykresu
    plt.title('Postęp Inteligencji Emocjonalnej', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Wynik Inteligencji Emocjonalnej', fontsize=12)
    plt.ylim(0, 100)  # Skala od 0 do 100
    plt.grid(True, linestyle='--', alpha=0.7)
    if len(dates) > 2:
        plt.legend()
    
    # Formatowanie osi x
    plt.gcf().autofmt_xdate()
    
    # Zapisz wykres do bufora
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Koduj obraz do base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64