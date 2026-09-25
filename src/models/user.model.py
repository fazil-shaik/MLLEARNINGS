# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy

# app = Flask(__name__)

# app.config["SQLALCHEMY_DATABASE_URI"] = (
#     "postgresql://datagrip_user:datagrip123@localhost:5432/flaskdb"
# )

# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# db = SQLAlchemy(app)


# class User(db.Model):
#     __tablename__ = "profile"

#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.String(255), nullable=False)


# with app.app_context():
#     db.create_all()


# @app.route("/")
# def home():
#     return "Flask + PostgreSQL connected"


# if __name__ == "__main__":
#     app.run(debug=True)

