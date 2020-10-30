from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# 회원모델 생성
class Fcuser(db.Model):
    __tablename__ = 'fsuser'
    id = db.Column(db.Integer, primary_key=True)
    password = db.Column(db.String(64))
    userid = db.Column(db.String(32))
    username = db.Column(db.String(8))