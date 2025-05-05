from app import db
from datetime import datetime, time
from flask_login import UserMixin
import json
import pytz

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    joined_date = db.Column(db.DateTime, default=datetime.now)
    
    # Pola związane z przypomnieniami
    phone_number = db.Column(db.String(20), nullable=True)  # do powiadomień SMS
    reminder_enabled = db.Column(db.Boolean, default=False)  # czy powiadomienia są włączone
    reminder_time = db.Column(db.Time, default=time(20, 0))  # domyślnie 20:00
    reminder_timezone = db.Column(db.String(50), default='Europe/Warsaw')  # strefa czasowa użytkownika
    reminder_method = db.Column(db.String(10), default='email')  # 'email' lub 'sms'
    last_reminder_sent = db.Column(db.DateTime, nullable=True)  # kiedy ostatnio wysłano przypomnienie
    
    # Relacje
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    psychological_analyses = db.relationship('PsychologicalAnalysis', backref='user', lazy=True)
    reminder_logs = db.relationship('ReminderLog', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'
    
    def is_reminder_due(self):
        """Sprawdza, czy należy wysłać przypomnienie użytkownikowi."""
        if not self.reminder_enabled:
            return False
            
        # Jeśli nie ma żadnych powiadomień w historii
        if not self.last_reminder_sent:
            return True
            
        # Pobierz aktualny czas w strefie czasowej użytkownika
        timezone = pytz.timezone(self.reminder_timezone)
        now = datetime.now(timezone)
        last_reminder = self.last_reminder_sent.astimezone(timezone)
        
        # Sprawdź, czy dziś już wysłano przypomnienie
        if (now.date() == last_reminder.date()):
            return False
            
        # Sprawdź, czy aktualna godzina jest późniejsza niż ustawiona godzina przypomnienia
        reminder_datetime = datetime.combine(now.date(), self.reminder_time)
        reminder_datetime = timezone.localize(reminder_datetime)
        
        return now >= reminder_datetime

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f'<Conversation {self.id}>'

class PsychologicalAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    analysis_data = db.Column(db.Text, nullable=False)  # JSON z analizą psychologiczną
    emotional_intelligence_score = db.Column(db.Integer, default=0)
    
    def get_analysis(self):
        """Konwertuje dane JSON na słownik Pythona"""
        return json.loads(self.analysis_data)
    
    def set_analysis(self, analysis_dict):
        """Konwertuje słownik Pythona na JSON do przechowywania"""
        self.analysis_data = json.dumps(analysis_dict)
    
    def __repr__(self):
        return f'<PsychologicalAnalysis {self.id} User {self.user_id}>'


class ReminderLog(db.Model):
    """Log wszystkich wysłanych przypomnień."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    method = db.Column(db.String(10), nullable=False)  # 'email' lub 'sms'
    status = db.Column(db.String(20), nullable=False)  # 'success', 'failed'
    error_message = db.Column(db.Text, nullable=True)  # w przypadku błędu
    
    def __repr__(self):
        return f'<ReminderLog {self.id} User {self.user_id} Method {self.method} Status {self.status}>'
