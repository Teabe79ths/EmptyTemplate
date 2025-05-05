import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from database import db

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
if not app.secret_key:
    raise ValueError("SESSION_SECRET environment variable is not set")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure PostgreSQL database
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://neondb_owner:npg_v7wLqX2AFzHa@ep-mute-cake-a2rfn0ki-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# initialize the app with the extension
db.init_app(app)

# Import models after db initialization
from models import User, Conversation, PsychologicalAnalysis, ReminderLog

with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        app.logger.error(f"Błąd podczas tworzenia tabel: {str(e)}")
        pass

# Import therapy functionality
from therapy import generate_question
from claude_api import generate_claude_question
from psychology import generate_psychological_insight, get_emotional_intelligence_score
from visualization import generate_emotion_chart, generate_emotional_intelligence_progress
from wordcloud_analyzer import analyze_user_responses_keywords
from quotes import generate_therapeutic_quote

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        user_id = session['user_id']
        from models import User, Conversation
        user = User.query.get(user_id)

        # Get the last conversation entry
        last_conversation = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.timestamp.desc()).first()

        # Get conversation history for context
        conversation_history = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.timestamp.desc()).limit(5).all()
        context = [{"question": c.question, "response": c.response, "date": c.timestamp} for c in conversation_history]

        # Check if we need a new question (if last entry has a response or no entries exist)
        new_question = None
        if not last_conversation or last_conversation.response:
            # Verify user exists before creating conversation
            user = User.query.get(user_id)
            if not user:
                flash('User not found.', 'danger')
                session.pop('user_id', None)
                return redirect(url_for('login'))

            # Użyj Claude do generowania pytania, jeśli dostępny, lub standardowego mechanizmu
            try:
                new_question = generate_claude_question(context if context else None)
                if not new_question or len(new_question.strip()) == 0:
                    from therapy import DEFAULT_FIRST_QUESTIONS
                    import random
                    new_question = random.choice(DEFAULT_FIRST_QUESTIONS)
                    logging.warning("Otrzymano puste pytanie - użyto pytania domyślnego")
            except Exception as e:
                # W przypadku jakichkolwiek błędów użyj domyślnego pytania
                logging.error(f"Błąd podczas generowania pytania: {str(e)}")
                from therapy import DEFAULT_FIRST_QUESTIONS
                import random
                new_question = random.choice(DEFAULT_FIRST_QUESTIONS)

            # Create a new conversation entry with the question
            new_conversation = Conversation(
                user_id=user_id,
                question=new_question,
                response=None,
                timestamp=datetime.now()
            )
            try:
                db.session.add(new_conversation)
                db.session.commit()
                last_conversation = new_conversation

                # Generate therapeutic quote
                from quotes import generate_therapeutic_quote
                quote = generate_therapeutic_quote()
                return render_template('index.html', user=user, conversation=last_conversation, quote=quote)
            except Exception as e:
                db.session.rollback()
                flash('Error creating conversation.', 'danger')
                return redirect(url_for('index'))

        return render_template('index.html', user=user, conversation=last_conversation)
    return render_template('index.html')

@app.route('/submit_response', methods=['POST'])
def submit_response():
    if 'user_id' not in session:
        flash('Musisz być zalogowany, aby kontynuować.', 'danger')
        return redirect(url_for('login'))

    response_text = request.form.get('response')
    conversation_id = request.form.get('conversation_id')

    if not response_text or not conversation_id:
        flash('Nieprawidłowe dane odpowiedzi.', 'danger')
        return redirect(url_for('index'))

    from models import Conversation
    conversation = Conversation.query.get(conversation_id)

    if not conversation or conversation.user_id != session['user_id']:
        flash('Nie znaleziono rozmowy lub nie masz do niej dostępu.', 'danger')
        return redirect(url_for('index'))

    conversation.response = response_text
    db.session.commit()

    flash('Twoja odpowiedź została zapisana. Dziękuję za refleksję!', 'success')
    return redirect(url_for('index'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Musisz być zalogowany, aby zobaczyć historię.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    from models import User, Conversation
    user = User.query.get(user_id)

    # Get all past conversations with responses
    conversations = Conversation.query.filter_by(user_id=user_id).filter(Conversation.response.isnot(None)).order_by(Conversation.timestamp.desc()).all()

    return render_template('history.html', user=user, conversations=conversations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        from models import User
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            flash('Zalogowano pomyślnie!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Nieprawidłowa nazwa użytkownika lub hasło.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        from models import User
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_user:
            flash('Nazwa użytkownika jest już zajęta.', 'danger')
        elif existing_email:
            flash('Email jest już zarejestrowany.', 'danger')
        else:
            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(new_user)
            db.session.commit()

            flash('Rejestracja zakończona sukcesem! Możesz się teraz zalogować.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/analysis')
def analysis():
    if 'user_id' not in session:
        flash('Musisz być zalogowany, aby zobaczyć analizę psychologiczną.', 'danger')
        return redirect(url_for('index'))

    user_id = session['user_id']
    from models import User, PsychologicalAnalysis, Conversation
    user = User.query.get(user_id)

    # Sprawdź, czy chcemy wygenerować nową analizę
    regenerate = request.args.get('regenerate') == 'True'

    # Pobierz najnowszą analizę lub wygeneruj nową
    latest_analysis = None
    if not regenerate:
        latest_analysis = PsychologicalAnalysis.query.filter_by(user_id=user_id).order_by(PsychologicalAnalysis.timestamp.desc()).first()

        # Sprawdź czy analiza jest aktualna (nie starsza niż ostatnia odpowiedź)
        if latest_analysis:
            last_response = Conversation.query.filter_by(user_id=user_id, response=None).order_by(Conversation.timestamp.desc()).first()
            if last_response and last_response.timestamp > latest_analysis.timestamp:
                latest_analysis = None  # Wymusi nową analizę

    # Jeśli nie mamy analizy lub chcemy ją zregenerować
    if not latest_analysis or regenerate:
        try:
            # Generuj nową analizę
            analysis_data = generate_psychological_insight(user_id, db)

            # Sprawdź, czy otrzymano komunikat o błędzie API w insights
            has_api_error = any("API" in insight for insight in analysis_data.get("insights", []))

            # Oblicz wynik inteligencji emocjonalnej
            ei_score = get_emotional_intelligence_score(analysis_data)

            # Zapisz analizę w bazie danych
            new_analysis = PsychologicalAnalysis(
                user_id=user_id,
                emotional_intelligence_score=ei_score
            )
            new_analysis.set_analysis(analysis_data)

            db.session.add(new_analysis)
            db.session.commit()

            latest_analysis = new_analysis

            if regenerate:
                if has_api_error:
                    flash('Twoja analiza została częściowo zaktualizowana. Wystąpiły problemy z usługami AI - używamy mechanizmu awaryjnego.', 'warning')
                else:
                    flash('Twoja analiza psychologiczna została zaktualizowana.', 'success')
        except Exception as e:
            logging.error(f"Błąd podczas generowania analizy: {str(e)}")
            flash('Wystąpił błąd podczas generowania analizy. Spróbuj ponownie później.', 'danger')
            if latest_analysis is None:
                # Jeśli nie mamy żadnej analizy, utwórz minimalną z komunikatem o błędzie
                from psychology import DEFAULT_ANALYSIS
                analysis_data = DEFAULT_ANALYSIS.copy()
                analysis_data["insights"] = ["Wystąpił błąd podczas generowania analizy. Spróbuj ponownie później."]

                new_analysis = PsychologicalAnalysis(
                    user_id=user_id,
                    emotional_intelligence_score=50
                )
                new_analysis.set_analysis(analysis_data)

                db.session.add(new_analysis)
                db.session.commit()

                latest_analysis = new_analysis

    # Przygotuj dane dla szablonu
    try:
        analysis_data = {
            'data': latest_analysis.get_analysis() if latest_analysis else None,
            'timestamp': latest_analysis.timestamp if latest_analysis else None,
            'emotional_intelligence_score': latest_analysis.emotional_intelligence_score if latest_analysis else 0
        }

        # Sprawdź, czy analiza zawiera wszystkie wymagane klucze
        if analysis_data['data'] is None or not all(key in analysis_data['data'] for key in ["personality_traits", "emotional_patterns", "cognitive_patterns", "insights", "growth_areas"]):
            # Jeśli nie, zwróć domyślną analizę
            from psychology import DEFAULT_ANALYSIS
            analysis_data['data'] = DEFAULT_ANALYSIS.copy()
            logging.warning("Zastosowano domyślną analizę psychologiczną - brakujące dane")
    except Exception as e:
        # W przypadku jakichkolwiek błędów użyj domyślnej analizy
        from psychology import DEFAULT_ANALYSIS
        analysis_data = {
            'data': DEFAULT_ANALYSIS.copy(),
            'timestamp': datetime.now(),
            'emotional_intelligence_score': 50
        }
        logging.error(f"Błąd podczas przygotowywania danych analizy: {str(e)}")

    # Pobierz wszystkie analizy użytkownika
    all_analyses = PsychologicalAnalysis.query.filter_by(user_id=user_id).order_by(PsychologicalAnalysis.timestamp.asc()).all()

    # Generuj wykresy tylko jeśli mamy więcej niż jedną analizę
    emotion_chart = None
    ei_progress_chart = None

    if len(all_analyses) > 1:
        emotion_chart = generate_emotion_chart(all_analyses)
        ei_progress_chart = generate_emotional_intelligence_progress(all_analyses)

    # Analizuj słowa kluczowe z odpowiedzi użytkownika
    keywords_analysis = analyze_user_responses_keywords(user_id, db)

    return render_template('analysis.html', 
                          user=user, 
                          analysis=analysis_data,
                          emotion_chart=emotion_chart,
                          ei_progress_chart=ei_progress_chart,
                          keywords_analysis=keywords_analysis)

@app.route('/reminder_settings', methods=['GET', 'POST'])
def reminder_settings():
    """Strona ustawień przypomnień dla użytkownika."""
    if 'user_id' not in session:
        flash('Musisz być zalogowany, aby zarządzać powiadomieniami.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    from models import User, ReminderLog
    from reminders import HAS_TWILIO, twilio_client
    import pytz

    user = User.query.get(user_id)

    # Lista dostępnych stref czasowych
    timezones = pytz.common_timezones

    # Pobierz logi przypomnień
    reminder_logs = ReminderLog.query.filter_by(user_id=user_id).order_by(ReminderLog.timestamp.desc()).limit(10).all()

    # Obsługa formularza przy metodzie POST
    if request.method == 'POST':
        # Zbierz dane z formularza
        reminder_enabled = 'reminder_enabled' in request.form
        reminder_method = request.form.get('reminder_method', 'email')
        phone_number = request.form.get('phone_number', '')
        reminder_time_str = request.form.get('reminder_time', '20:00')
        reminder_timezone = request.form.get('reminder_timezone', 'Europe/Warsaw')

        # Przetwórz czas przypomnienia
        try:
            hours, minutes = map(int, reminder_time_str.split(':'))
            from datetime import time
            reminder_time = time(hours, minutes)
        except:
            reminder_time = user.reminder_time or time(20, 0)

        # Walidacja numeru telefonu dla SMS
        if reminder_method == 'sms' and not phone_number.startswith('+'):
            flash('Numer telefonu musi zaczynać się od znaku "+" i zawierać kod kraju.', 'danger')
            return render_template('reminder_settings.html', 
                                  user=user, 
                                  has_twilio=HAS_TWILIO and twilio_client is not None,
                                  timezones=timezones,
                                  reminder_logs=reminder_logs,
                                  error_message='Nieprawidłowy format numeru telefonu.')

        # Zaktualizuj dane użytkownika
        user.reminder_enabled = reminder_enabled
        user.reminder_method = reminder_method
        user.phone_number = phone_number
        user.reminder_time = reminder_time
        user.reminder_timezone = reminder_timezone

        # Zapisz zmiany
        db.session.commit()

        flash('Ustawienia przypomnień zostały zaktualizowne.', 'success')

    return render_template('reminder_settings.html', 
                          user=user, 
                          has_twilio=HAS_TWILIO and twilio_client is not None,
                          timezones=timezones,
                          reminder_logs=reminder_logs)

@app.route('/test_reminder', methods=['POST'])
def test_reminder():
    """Wysyła testowe przypomnienie do użytkownika."""
    if 'user_id' not in session:
        flash('Musisz być zalogowany, aby przetestować powiadomienia.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    from models import User
    from reminders import send_reminder

    user = User.query.get(user_id)

    # Wyślij testowe przypomnienie
    success, error = send_reminder(user)

    if success:
        flash('Testowe powiadomienie zostało wysłane pomyślnie.', 'success')
    else:
        flash(f'Błąd podczas wysyłania powiadomienia: {error}', 'danger')

    return redirect(url_for('reminder_settings'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Wylogowano pomyślnie.', 'success')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)