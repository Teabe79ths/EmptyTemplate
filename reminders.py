"""
Moduł obsługujący system przypomnień dla aplikacji terapeutycznej.

Zapewnia funkcje do wysyłania przypomnień przez SMS lub e-mail
oraz zarządzania harmonogramem przypomnień.
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import pytz

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sprawdź Twilio API
HAS_TWILIO = False
twilio_client = None

try:
    from twilio.rest import Client
    HAS_TWILIO = True
    
    # Inicjalizacja klienta Twilio
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER")
    
    if account_sid and auth_token and twilio_phone:
        twilio_client = Client(account_sid, auth_token)
        logger.info("Zainicjalizowano klienta Twilio dla powiadomień SMS")
    else:
        missing = []
        if not account_sid: missing.append("TWILIO_ACCOUNT_SID")
        if not auth_token: missing.append("TWILIO_AUTH_TOKEN")
        if not twilio_phone: missing.append("TWILIO_PHONE_NUMBER")
        logger.warning(f"Brak kluczy Twilio API: {', '.join(missing)}")
except ImportError:
    logger.warning("Nie można zaimportować Twilio. Powiadomienia SMS są niedostępne.")
except Exception as e:
    logger.warning(f"Problem z inicjalizacją Twilio: {str(e)}")

# Konfiguracja SMTP dla powiadomień email
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", SMTP_USERNAME)

def send_email_reminder(user):
    """
    Wysyła przypomnienie e-mail do użytkownika.
    
    Args:
        user: Obiekt User, do którego wysyłamy przypomnienie
        
    Returns:
        tuple: (sukces: bool, komunikat błędu: str lub None)
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return False, "Brak konfiguracji SMTP"
        
    try:
        # Utwórz wiadomość
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = user.email
        msg['Subject'] = "Przypomnienie o codziennej refleksji"
        
        # Treść wiadomości
        body = f"""
        Cześć {user.username}!
        
        To Twoje codzienne przypomnienie o refleksji terapeutycznej.
        Poświęć kilka minut, aby odpowiedzieć na pytanie dnia i zbliżyć się do lepszego zrozumienia siebie.
        
        Przejdź do aplikacji: https://terapia.replit.app
        
        Pozdrawiamy,
        Zespół RefleksjaApp
        
        ---
        To jest wiadomość automatyczna. Możesz wyłączyć lub zmodyfikować powiadomienia w ustawieniach swojego konta.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Połącz z serwerem i wyślij
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, user.email, text)
        server.quit()
        
        logger.info(f"Wysłano przypomnienie e-mail do użytkownika {user.username}")
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Błąd podczas wysyłania e-mail do {user.username}: {error_msg}")
        return False, error_msg

def send_sms_reminder(user):
    """
    Wysyła przypomnienie SMS do użytkownika.
    
    Args:
        user: Obiekt User, do którego wysyłamy przypomnienie
        
    Returns:
        tuple: (sukces: bool, komunikat błędu: str lub None)
    """
    if not HAS_TWILIO or not twilio_client:
        return False, "Brak konfiguracji Twilio"
        
    if not user.phone_number:
        return False, "Brak numeru telefonu użytkownika"
    
    try:
        # Treść wiadomości SMS
        message_body = f"Refleksja: Przypominamy o codziennej refleksji terapeutycznej. Poświęć chwilę na odpowiedź w aplikacji."
        
        # Wyślij wiadomość
        message = twilio_client.messages.create(
            body=message_body,
            from_=os.environ.get("TWILIO_PHONE_NUMBER"),
            to=user.phone_number
        )
        
        logger.info(f"Wysłano przypomnienie SMS do użytkownika {user.username} (SID: {message.sid})")
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Błąd podczas wysyłania SMS do {user.username}: {error_msg}")
        return False, error_msg

def send_reminder(user):
    """
    Wysyła przypomnienie do użytkownika wybraną przez niego metodą.
    
    Args:
        user: Obiekt User z modelu danych
        
    Returns:
        tuple: (sukces: bool, komunikat błędu: str lub None)
    """
    from models import ReminderLog
    from app import db
    
    # Wybierz metodę wysyłki
    if user.reminder_method == 'sms':
        success, error = send_sms_reminder(user)
    else:  # domyślnie email
        success, error = send_email_reminder(user)
    
    # Zapisz log
    log = ReminderLog(
        user_id=user.id,
        method=user.reminder_method,
        status='success' if success else 'failed',
        error_message=error
    )
    
    # Aktualizuj czas ostatniego przypomnienia
    if success:
        user.last_reminder_sent = datetime.now(pytz.utc)
    
    # Zapisz zmiany w bazie
    db.session.add(log)
    db.session.commit()
    
    return success, error

def check_and_send_due_reminders():
    """
    Sprawdza i wysyła zaplanowane przypomnienia dla wszystkich użytkowników.
    
    Returns:
        int: Liczba wysłanych przypomnień
    """
    from models import User
    from app import db
    
    logger.info("Sprawdzanie przypomnień do wysłania...")
    users = User.query.filter_by(reminder_enabled=True).all()
    sent_count = 0
    
    for user in users:
        if user.is_reminder_due():
            logger.info(f"Wysyłanie przypomnienia do {user.username}")
            success, error = send_reminder(user)
            
            if success:
                sent_count += 1
    
    logger.info(f"Wysłano {sent_count} przypomnień spośród {len(users)} aktywnych użytkowników")
    return sent_count

# Jeśli ten plik jest uruchamiany bezpośrednio, sprawdź i wyślij przypomnienia
if __name__ == "__main__":
    from app import app
    with app.app_context():
        check_and_send_due_reminders()