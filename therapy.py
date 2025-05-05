import random
from datetime import datetime
from typing import Dict, Any, Optional

# List of default first questions for new users (in Polish)
DEFAULT_FIRST_QUESTIONS = [
    "Jak się dziś czujesz? Opowiedz mi trochę o swoich odczuciach.",
    "Co sprawiło, że zdecydowałeś/aś się skorzystać z naszej wirtualnej terapii?",
    "Czy jest coś konkretnego, nad czym chciałbyś/chciałabyś popracować podczas naszych rozmów?",
    "Jak wyglądał Twój dzień? Co najbardziej wpłynęło na Twoje samopoczucie?",
    "Jakie emocje towarzyszą Ci najczęściej w ciągu dnia? Potrafisz je nazwać?",
    "Co jest dla Ciebie największym wyzwaniem w tym momencie życia?",
    "Co daje Ci poczucie spokoju i równowagi?",
    "Jak przebiegł Twój ostatni tydzień? Czy coś szczególnie zapadło Ci w pamięć?",
    "Co było ostatnio dla Ciebie źródłem siły lub inspiracji?",
    "Z jakim nastawieniem rozpoczynasz dzisiejszą rozmowę?",
    "Co przynosi Ci największą satysfakcję w codziennym życiu?",
    "Jak radziłeś/aś sobie z wyzwaniami w ostatnim czasie?",
    "Co dzisiaj zwróciło Twoją szczególną uwagę? Czy wywołało to jakieś emocje?",
    "Jaki jest Twój obecny stan energii i nastroju?",
    "Czy jest jakaś kwestia, którą szczególnie chciałbyś/chciałabyś dziś poruszyć?",
    "Co skłoniło Cię do refleksji nad sobą w tym momencie życia?",
    "Czy zauważasz jakieś zmiany w swoim samopoczuciu lub postrzeganiu świata w ostatnim czasie?",
    "Co najbardziej wpływa na Twoje samopoczucie w ciągu typowego dnia?",
    "Jakie sytuacje lub osoby najsilniej oddziałują na Twój nastrój?",
    "Gdybyś miał/a opisać swoje obecne życie jednym słowem, jakie by to było słowo i dlaczego?",
]

# List of follow-up questions when there's limited context (in Polish)
FOLLOW_UP_QUESTIONS = [
    "W jaki sposób radzisz sobie z trudnymi emocjami, które pojawiają się w Twoim życiu?",
    "Jakie są Twoje sposoby na relaks i odzyskanie energii?",
    "Czy zauważasz jakieś powtarzające się wzorce w swoim zachowaniu lub myśleniu?",
    "Co chciałbyś/chciałabyś zmienić w swoim życiu? Co Cię powstrzymuje?",
    "Kiedy ostatnio czułeś/aś się naprawdę szczęśliwy/a? Co się wtedy wydarzyło?",
    "Jak wyglądałby Twój idealny dzień? Spróbuj go sobie wyobrazić i opisać.",
    "W jakich sytuacjach czujesz, że jesteś w pełni sobą?",
    "Jakie wartości są dla Ciebie najważniejsze w życiu?",
    "Kto jest dla Ciebie wsparciem, gdy przeżywasz trudne chwile?",
    "Czy potrafisz sobie wybaczyć błędy? Jak wygląda Twój dialog wewnętrzny?",
    "Jakie działania podejmujesz, aby zadbać o swoje zdrowie psychiczne?",
    "Co uważasz za swoje największe osiągnięcie w ostatnim czasie?",
    "Czy są jakieś decyzje, które odwlekasz? Co sprawia, że trudno Ci je podjąć?",
    "Jak reagujesz na niepowodzenia? Czy dużo wymagasz od siebie?",
    "Co najbardziej cenisz w swoim charakterze? A co chciałbyś/chciałabyś zmienić?",
    "Jakie marzenia chciałbyś/chciałabyś zrealizować w najbliższej przyszłości?",
    "Czy jest ktoś, komu trudno Ci wybaczyć? Jak wpływa to na Twoje życie?",
    "Co sprawia, że czujesz się wartościową osobą?",
    "Jak się zmieniłeś/aś na przestrzeni ostatnich lat? Co było punktem zwrotnym?",
    "Jakie są Twoje sposoby na radzenie sobie z krytyką ze strony innych?",
    "Co daje Ci poczucie spełnienia i sensu w życiu?",
    "Jak opisałbyś/abyś swoją relację z własnymi emocjami?",
    "Jak odpoczywasz? Czy uważasz, że poświęcasz wystarczająco dużo czasu na regenerację?",
    "Czy jest coś, co regularnie odkładasz 'na później'? Co to jest i dlaczego to robisz?",
    "Jak reaguje Twoje ciało na stres? Czy zwracasz uwagę na sygnały, które Ci wysyła?",
    "Co w Twoim życiu chciałbyś/chciałabyś uprościć?",
    "Czego najbardziej się obawiasz? Skąd biorą się te obawy?",
    "Kiedy ostatnio zrobiłeś/aś coś po raz pierwszy? Jakie to było doświadczenie?",
    "Co pomaga Ci przetrwać trudne dni? Do jakich zasobów się odwołujesz?",
    "Jakie granice stawiasz innym ludziom? Czy przychodzi Ci to łatwo?",
    "Co chciałbyś/chciałabyś usłyszeć od bliskiej osoby w tym momencie?",
    "Czy są jakieś aspekty Twojego życia, które wydają Ci się poza Twoją kontrolą?",
    "Jakie kompromisy najtrudniej Ci zawierać?",
    "Co obecnie jest Twoim największym źródłem stresu i jak sobie z tym radzisz?",
    "Jakie relacje najbardziej Cię kształtują i dlaczego?",
    "Co dodaje Ci skrzydeł, a co Cię powstrzymuje?",
    "Czy jest jakaś osoba, którą podziwiasz? Co w niej cenisz?",
    "W jakich momentach czujesz się najbardziej sobą?",
    "Co najczęściej zakłóca Twój spokój wewnętrzny?",
    "Jak dbasz o swój dobrostan emocjonalny na co dzień?",
]

# Contextual follow-up questions based on specific emotions or themes (in Polish)
CONTEXTUAL_QUESTIONS = {
    'samotność': [
        "Co Twoim zdaniem najbardziej przeszkadza Ci w budowaniu prawdziwej więzi z innymi?",
        "Kiedy czujesz się samotny/a, co zwykle robisz, aby poradzić sobie z tym uczuciem?",
        "Czy jest różnica między byciem samotnym a byciem samemu? Co to dla Ciebie oznacza?",
        "Co mogłoby pomóc Ci poczuć się bardziej połączonym/ą z ludźmi wokół Ciebie?",
        "Czy samotność jest dla Ciebie stanem neutralnym, pozytywnym czy negatywnym?",
        "Kiedy w swoim życiu czułeś/aś się najbardziej połączony/ą z innymi ludźmi?",
        "Jak zmieniło się Twoje podejście do samotności na przestrzeni lat?",
        "Jak odróżniasz zdrową potrzebę samotności od poczucia osamotnienia?",
        "Czy jest jakaś osoba, z którą chciałbyś/chciałabyś odnowić kontakt? Co Cię powstrzymuje?",
        "W jaki sposób media społecznościowe wpływają na Twoje poczucie samotności lub połączenia z innymi?",
    ],
    'stres': [
        "Jakie sytuacje najczęściej wywołują u Ciebie stres? Czy dostrzegasz w nich jakiś wzorzec?",
        "W jaki sposób stres wpływa na Twoje codzienne funkcjonowanie?",
        "Jakie metody radzenia sobie ze stresem sprawdzają się u Ciebie najlepiej?",
        "Czy potrafisz rozpoznać pierwsze sygnały stresu w swoim ciele? Jakie to są sygnały?",
        "Jak często doświadczasz stresu i jak wpływa to na Twoją jakość życia?",
        "Czy Twój sposób reagowania na stres zmienił się z upływem czasu?",
        "Jakie techniki relaksacyjne wypróbowałeś/aś i które z nich okazały się najbardziej skuteczne?",
        "W jaki sposób stres wpływa na Twoje relacje z innymi ludźmi?",
        "Czy zauważasz jakieś pozytywne aspekty stresu w swoim życiu?",
        "Co powiedziałbyś/powiedziałabyś młodszej wersji siebie o radzeniu sobie ze stresem?",
    ],
    'lęk': [
        "Kiedy pojawia się lęk, jakie myśli mu zwykle towarzyszą?",
        "Co pomaga Ci uspokoić się, gdy doświadczasz lęku?",
        "Czy Twój lęk jest związany z konkretnymi sytuacjami, czy pojawia się bez wyraźnego powodu?",
        "W jaki sposób lęk wpływa na Twoje decyzje i wybory życiowe?",
        "Czy potrafisz zidentyfikować korzenie swojego lęku? Kiedy się pojawił?",
        "Jak często lęk powstrzymuje Cię przed zrobieniem czegoś, co chciałbyś/chciałabyś zrobić?",
        "Jaką rolę odgrywa wyobraźnia w Twoim doświadczaniu lęku?",
        "Czy komunikujesz innym, gdy odczuwasz lęk? Dlaczego tak/nie?",
        "W jaki sposób Twoje ciało reaguje na lęk? Jakie fizyczne objawy zauważasz?",
        "Co stanowi dla Ciebie największe źródło lęku o przyszłość?",
    ],
    'smutek': [
        "Co zwykle wywołuje u Ciebie smutek? Jak długo on trwa?",
        "Jak wyrażasz smutek? Czy pozwalasz sobie go przeżywać?",
        "Co pomaga Ci przejść przez trudne chwile smutku?",
        "Czy są rzeczy, które kiedyś sprawiały Ci radość, a teraz już nie? Co się zmieniło?",
        "Jak reagujesz, gdy inni zauważają Twój smutek?",
        "Czy uważasz, że smutek może mieć pozytywną funkcję w życiu? Jaką?",
        "Czy są jakieś utwory (muzyka, filmy, książki), które szczególnie rezonują z Twoim smutkiem?",
        "Jak odróżniasz zwykły smutek od głębszego przygnębienia?",
        "W jaki sposób wyrażanie smutku było modelowane w Twoim domu rodzinnym?",
        "Czy łatwiej Ci przeżywać smutek samemu, czy w obecności innych?",
    ],
    'radość': [
        "Co sprawia, że czujesz radość w codziennym życiu?",
        "Jak często pozwalasz sobie na celebrowanie małych sukcesów i przyjemności?",
        "W jaki sposób dzielisz się swoją radością z innymi?",
        "Czy potrafisz odnajdywać radość nawet w trudnych okolicznościach?",
        "Jakie wspomnienia wywołują u Ciebie największą radość?",
        "Co jest dla Ciebie źródłem czystej, niczym niezmąconej radości?",
        "Jak zmieniły się Twoje źródła radości na przestrzeni lat?",
        "Czy łatwo przychodzi Ci odczuwanie radości, czy raczej musisz się o nią starać?",
        "Jak reaguje Twoje ciało, gdy odczuwasz głęboką radość?",
        "Co mógłbyś/mogłabyś zrobić, aby wprowadzić więcej radości do swojego codziennego życia?",
    ],
    'złość': [
        "Jak wyrażasz złość? Czy masz zdrowe sposoby na jej ujawnianie?",
        "Co najczęściej wywołuje u Ciebie złość? Czy są to konkretne sytuacje, czy raczej zachowania innych?",
        "Czy złość pomaga Ci w jakiś sposób, czy raczej przeszkadza?",
        "Jak reagujesz na złość innych osób?",
        "Jakie przekonania na temat złości wyniosłeś/aś z domu rodzinnego?",
        "Czy złość bywa dla Ciebie motywująca, czy raczej destrukcyjna?",
        "Co dzieje się z Twoim ciałem, gdy odczuwasz złość?",
        "Czy potrafisz rozpoznać złość, zanim w pełni się rozwinie?",
        "Jakie strategie stosujesz, by nie reagować impulsywnie w złości?",
        "Czy łatwiej Ci wyrażać złość wobec bliskich, czy wobec obcych osób?",
    ],
    'praca': [
        "Jakie aspekty Twojej pracy sprawiają Ci największą satysfakcję?",
        "Czy czujesz, że Twoja praca jest zgodna z Twoimi wartościami i celami życiowymi?",
        "Co chciałbyś/chciałabyś zmienić w swojej sytuacji zawodowej?",
        "Jak równoważysz życie zawodowe z osobistym?",
        "W jaki sposób Twoja praca przyczynia się do poczucia sensu w Twoim życiu?",
        "Jakie umiejętności rozwinąłeś/aś dzięki swojej pracy?",
        "Czy czujesz, że Twoja praca pozwala Ci się rozwijać jako osoba?",
        "Jak wyobrażasz sobie swoją pracę/karierę za 5 lat?",
        "Co sprawia, że czujesz się doceniony/a w pracy?",
        "Jak wpływa na Ciebie środowisko pracy, w którym funkcjonujesz?",
    ],
    'relacje': [
        "Jakie cechy cenisz najbardziej w bliskich relacjach?",
        "Co jest dla Ciebie najtrudniejsze w budowaniu i utrzymywaniu relacji?",
        "Jak wyrażasz swoje potrzeby i granice w relacjach?",
        "Czy masz kogoś, z kim możesz być w pełni sobą? Co sprawia, że tak się czujesz przy tej osobie?",
        "Jakie wzorce relacji wyniosłeś/aś z domu rodzinnego?",
        "W jaki sposób okazujesz troskę i miłość ważnym dla Ciebie osobom?",
        "Jak radzisz sobie z konfliktami w relacjach?",
        "Czy łatwiej przychodzi Ci dawanie czy przyjmowanie w relacjach?",
        "Jak zbudować głębszą intymność w relacji według Ciebie?",
        "Co jest dla Ciebie najtrudniejsze do zaakceptowania u bliskich osób?",
    ],
    'samorozwój': [
        "W jakich obszarach swojego życia najchętniej się rozwijasz?",
        "Co motywuje Cię do pracy nad sobą?",
        "Jaką najważniejszą lekcję życiową wyniosłeś/aś z ostatniego roku?",
        "Jak radzisz sobie z przeszkodami na drodze do osobistego rozwoju?",
        "Co oznacza dla Ciebie bycie 'lepszą wersją siebie'?",
        "Jak wiesz, że rozwinąłeś/aś się w jakimś obszarze?",
        "Jakie książki, kursy lub doświadczenia najbardziej przyczyniły się do Twojego rozwoju?",
        "Czy stawiasz sobie konkretne cele rozwojowe, czy raczej podchodzisz do tego intuicyjnie?",
        "Które obszary swojego życia najtrudniej Ci rozwijać i dlaczego?",
        "Jak równoważysz akceptację siebie z dążeniem do zmiany?",
    ],
    'zdrowie': [
        "Jak definiujesz zdrowie w swoim życiu? Co to dla Ciebie znaczy być zdrowym?",
        "W jaki sposób dbasz o swoje zdrowie fizyczne na co dzień?",
        "Czy zwracasz uwagę na sygnały, które wysyła Ci Twoje ciało?",
        "Jak Twój stan zdrowia wpływa na inne aspekty Twojego życia?",
        "Co jest dla Ciebie największym wyzwaniem w dbaniu o zdrowie?",
        "Jakie nawyki chciałbyś/chciałabyś zmienić, aby poprawić swoje zdrowie?",
        "Jak równoważysz troskę o zdrowie z innymi priorytetami życiowymi?",
        "W jaki sposób zdrowie psychiczne i fizyczne są ze sobą powiązane w Twoim życiu?",
        "Czy masz jakieś rytuały zdrowotne, które pomagają Ci utrzymać dobrostan?",
        "Jak zmienił się Twój stosunek do zdrowia na przestrzeni ostatnich lat?",
    ],
    'przeszłość': [
        "Jakie doświadczenia z przeszłości ukształtowały Cię najbardziej?",
        "Czy jest coś w Twojej przeszłości, do czego często wracasz myślami?",
        "Jak wpływają na Ciebie wspomnienia z dzieciństwa?",
        "Czy są jakieś momenty z przeszłości, które chciałbyś/chciałabyś zmienić? Dlaczego?",
        "Jak radzisz sobie z trudnymi wspomnieniami?",
        "Co chciałbyś/chciałabyś, aby Twoje młodsze 'ja' wiedziało?",
        "Jakie lekcje wyniosłeś/aś z popełnionych w przeszłości błędów?",
        "Czy łatwo przychodzi Ci przebaczanie sobie za przeszłe decyzje?",
        "Jakie pozytywne wspomnienia stanowią dla Ciebie źródło siły?",
        "W jaki sposób Twoja przeszłość wpływa na Twoje obecne relacje?",
    ],
    'przyszłość': [
        "Jak wyobrażasz sobie swoją przyszłość za 5 lat?",
        "Co najbardziej ekscytuje Cię, gdy myślisz o przyszłości?",
        "Czego najbardziej obawiasz się w kontekście przyszłości?",
        "Jakie marzenia chciałbyś/chciałabyś zrealizować w nadchodzących latach?",
        "Czy planujesz swoją przyszłość, czy raczej pozwalasz, by życie toczyło się samo?",
        "Jakie wartości chciałbyś/chciałabyś kultywować w swojej przyszłości?",
        "Co chciałbyś/chciałabyś, aby inni powiedzieli o Tobie w przyszłości?",
        "Jak przygotowujesz się do przyszłych wyzwań, które możesz napotkać?",
        "Jakie umiejętności chciałbyś/chciałabyś rozwinąć z myślą o przyszłości?",
        "Czy myślenie o przyszłości wywołuje w Tobie więcej nadziei czy niepokoju?",
    ],
    'wartości': [
        "Jakie wartości są dla Ciebie najważniejsze w życiu?",
        "Czy Twoje codzienne wybory odzwierciedlają Twoje najgłębsze wartości?",
        "Jak rozpoznajesz, co jest dla Ciebie naprawdę ważne?",
        "Czy Twoje wartości zmieniły się na przestrzeni lat? W jaki sposób?",
        "W jakich sytuacjach najłatwiej przychodzi Ci kierowanie się swoimi wartościami?",
        "Czy są wartości, które chciałbyś/chciałabyś mocniej wcielić w swoje życie?",
        "Jak reagujesz, gdy ktoś kwestionuje lub podważa Twoje wartości?",
        "Od kogo nauczyłeś/aś się swoich najważniejszych wartości?",
        "Czy zdarza Ci się działać wbrew swoim wartościom? W jakich okolicznościach?",
        "Jak Twoje wartości wpływają na Twoje relacje z innymi ludźmi?",
    ],
}

# Words that might indicate specific emotions or themes (in Polish)
KEYWORDS = {
    'samotność': ['samotny', 'samotna', 'sam', 'sama', 'odizolowany', 'odizolowana', 'opuszczony', 'opuszczona', 'brak związku', 'brak relacji', 'nikt', 'odrzucony', 'odrzucona'],
    'stres': ['stres', 'napięcie', 'presja', 'przytłoczony', 'przytłoczona', 'zestresowany', 'zestresowana', 'deadline', 'termin', 'obciążenie', 'przeciążony', 'przeciążona'],
    'lęk': ['lęk', 'strach', 'niepokój', 'obawa', 'martwię się', 'zmartwiony', 'zmartwiona', 'zaniepokojony', 'zaniepokojona', 'panika', 'niepewność'],
    'smutek': ['smutek', 'smutny', 'smutna', 'przygnębiony', 'przygnębiona', 'płacz', 'żal', 'strata', 'rozczarowanie', 'melancholia', 'depresja'],
    'radość': ['radość', 'szczęście', 'zadowolony', 'zadowolona', 'uciecha', 'szczęśliwy', 'szczęśliwa', 'śmiech', 'entuzjazm', 'duma', 'spełnienie'],
    'złość': ['złość', 'wściekłość', 'gniew', 'irytacja', 'zdenerwowany', 'zdenerwowana', 'poirytowany', 'poirytowana', 'frustracja', 'wkurzony', 'wkurzona'],
    'praca': ['praca', 'zawód', 'kariera', 'stanowisko', 'szef', 'przełożony', 'współpracownik', 'awans', 'wynagrodzenie', 'pensja', 'firma', 'korporacja', 'biznes'],
    'relacje': ['relacja', 'związek', 'partner', 'partnerka', 'mąż', 'żona', 'przyjaciel', 'przyjaciółka', 'rodzina', 'rodzice', 'dzieci', 'bliskość', 'intymność'],
    'samorozwój': ['rozwój', 'doskonalenie', 'zmiana', 'poprawa', 'cel', 'realizacja', 'ambicja', 'aspiracja', 'lepszy', 'lepsza', 'wyzwanie', 'nauka', 'wzrost'],
    'zdrowie': ['zdrowie', 'choroba', 'ból', 'ciało', 'samopoczucie', 'kondycja', 'lekarz', 'leki', 'terapia', 'dieta', 'ćwiczenia', 'sen', 'odpoczynek', 'energia'],
    'przeszłość': ['przeszłość', 'historia', 'kiedyś', 'dawniej', 'wspomnienie', 'pamięć', 'dzieciństwo', 'młodość', 'doświadczenie', 'błąd', 'żal', 'tęsknota', 'trauma'],
    'przyszłość': ['przyszłość', 'plan', 'cel', 'marzenie', 'nadzieja', 'wizja', 'perspektywa', 'zmiana', 'rozwój', 'obawy', 'niepewność', 'oczekiwania'],
    'wartości': ['wartość', 'sens', 'znaczenie', 'przekonanie', 'ideał', 'priorytet', 'zasada', 'etyka', 'moralność', 'dobro', 'uczciwość', 'odpowiedzialność', 'szacunek'],
}

def analyze_context(context, user=None):
    """
    Analyze the conversation context to identify emotional themes
    and generate appropriate follow-up questions with improved variety.
    """
    if not context or len(context) == 0:
        return random.choice(DEFAULT_FIRST_QUESTIONS)

    # Extract text and track used themes
    all_text = " ".join([entry.get("response", "") for entry in context if entry.get("response")])
    all_text = all_text.lower()

    all_themes = list(KEYWORDS.keys())

    # Identify themes based on keywords with weights
    themes = {}
    for theme, keywords in KEYWORDS.items():
        # Calculate theme relevance
        count = sum(1 for keyword in keywords if keyword.lower() in all_text)
        if count > 0:
            # Add randomization factor to prevent getting stuck in one theme
            randomization = random.uniform(0.8, 1.2)
            themes[theme] = count * randomization

            # Boost underutilized themes
            if user and theme not in user.therapy_progress["covered_themes"]:
                themes[theme] *= 1.5

    # Simulate additional contextual information (replace with actual data if available)
    emotional_history = {"sadness": 0.6, "anxiety": 0.3, "joy": 0.1}  # Example
    behavioral_patterns = {"avoidance": 0.7, "rumination": 0.4}  # Example
    development_areas = {"communication": 0.8, "self-esteem": 0.5}  # Example


    # If we've identified themes, use contextual questions with improved selection, incorporating additional context
    if themes:
        # Sort themes by weighted scores
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)

        # Wybierz nowy temat jeśli nie ma aktualnego lub poprzedni został wyczerpany
        if user and (not user.therapy_progress["current_theme"] or 
                    len([q for q in user.therapy_progress["questions_asked"] if user.therapy_progress["current_theme"] in q]) >= 5):
            # Znajdź tematy, które nie były jeszcze omawiane
            unused_themes = [t for t in all_themes if t not in user.therapy_progress["covered_themes"]]

            if not unused_themes:  # Jeśli wszystkie tematy były już omawiane
                user.therapy_progress["covered_themes"] = []  # Zresetuj listę
                unused_themes = all_themes

            # Wybierz nowy temat z nieużywanych
            user.therapy_progress["current_theme"] = random.choice(unused_themes)
            user.therapy_progress["covered_themes"].append(user.therapy_progress["current_theme"])

        selected_theme = user.therapy_progress["current_theme"] if user else random.choice(all_themes)

        if selected_theme in CONTEXTUAL_QUESTIONS:
            # Filtruj pytania, które już zostały zadane
            asked_questions = user.therapy_progress["questions_asked"] if user else []
            available_questions = [q for q in CONTEXTUAL_QUESTIONS[selected_theme] 
                                if q not in asked_questions]

            if available_questions:
                selected_question = random.choice(available_questions)
                if user:
                    user.therapy_progress["questions_asked"].append(selected_question)
                return selected_question

    # Ensure variety when falling back to follow-up questions
    recent_questions = [entry.get("question", "") for entry in context[-10:]]
    available_questions = [q for q in FOLLOW_UP_QUESTIONS if q not in recent_questions]

    return random.choice(available_questions if available_questions else FOLLOW_UP_QUESTIONS)

def generate_question(context=None, user=None):
    """
    Generate a therapeutic question based on previous conversation context.

    Args:
        context (list): A list of previous conversation entries
                       Each entry is a dict with 'question', 'response', and 'date'

    Returns:
        str: A therapeutic question in Polish
    """
    # Generate a thoughtful question based on context
    question = analyze_context(context, user)

    return question