
from app import app
from database import db

def init_db():
    with app.app_context():
        from models import User, Conversation, PsychologicalAnalysis, ReminderLog
        print("Creating database tables...")
        db.create_all()
        print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()
