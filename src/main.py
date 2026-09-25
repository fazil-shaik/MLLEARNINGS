from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# MODELS

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )


class Task(db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    title = db.Column(
        db.String(200),
        nullable=False
    )

    description = db.Column(
        db.Text
    )

    completed = db.Column(
        db.Boolean,
        default=False
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )


# CREATE TABLES

with app.app_context():
    db.create_all()


# POST USER

@app.route("/users", methods=["POST"])
def create_user():

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    password_hash = data.get("password_hash")

    user = User(
        name=name,
        email=email,
        password_hash=password_hash
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "User created successfully",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }), 201


# GET USERS

@app.route("/users", methods=["GET"])
def get_users():

    users = User.query.all()

    result = []

    for user in users:

        result.append({
            "id": user.id,
            "name": user.name,
            "email": user.email
        })

    return jsonify(result)


# POST TASK

@app.route("/tasks", methods=["POST"])
def create_task():

    data = request.get_json()

    title = data.get("title")
    description = data.get("description")
    user_id = data.get("user_id")

    task = Task(
        title=title,
        description=description,
        user_id=user_id
    )

    db.session.add(task)
    db.session.commit()

    return jsonify({
        "message": "Task created successfully",
        "task": {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "completed": task.completed,
            "user_id": task.user_id
        }
    }), 201




@app.route("/tasks", methods=["GET"])
def get_tasks():

    tasks = Task.query.all()

    result = []

    for task in tasks:

        result.append({
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "completed": task.completed,
            "user_id": task.user_id
        })

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)








# from flask import Flask, url_for,request,json,redirect
# import requests
# import os
# import datetime
# from flask import jsonify


# app = Flask(__name__)

# # @app.route('/')
# # def index():
# #     return '<h1>Welcome to main page<h1>'



# # @app.route('/demo')
# # def demoRoute():
# #     return "<h2>Welcome to demo route</h2>"


# # # @app.route('/user',methods=['POST'])
# # # def user():
# # #     data = request.get_json()
# # #     name = data.get('name')
# # #     age = data.get('age')

# # #     return f"Name: {name}, Age: {age}"





# # # # @app.route('/login')
# # # # def login():
# # # #     return 'login'

# # # @app.route('/user/<username>')
# # # def profile(username):
# # #     return f'{username}\'s profile'

# # # with app.test_request_context():
# # #     print(url_for('index'))
# # #     print(url_for('login'))
# # #     # print(url_for('login', next='/'))
# # #     print(url_for('profile', username='John Doe'))


# # # from flask import request

# # @app.route('/login', methods=['GET', 'POST'])
# # def login():
# #     if request=='POST':
# #         return do_the_login()
# #     else:
# #         return show_the_login_form()

# # def do_the_login():
# #     return "welcome to login page"


# # def show_the_login_form():
# #     return "welcome to login form"




# # @app.route('/write')

# # def writeFile():

# #     with open('data.txt','+w') as file:
# #         file.write('Hello from file')

# #     return "File written successfully"



# # @app.route('/read')

# # def readFile():

# #     with open('data.txt','r') as file:
# #         content = file.read()


# #     return content



# # # @app.route('/user', methods=['POST'])
# # # def user():

# # #     data = request.get_json()

# # #     name = data.get('name')
# # #     age = data.get('age')

# # #     with open("users.txt", 'a+') as file:
# # #         file.write(f"{name},{age}\n")

# # #     return "User saved"


# # @app.route('/upload',methods=['POST'])

# # def upload():

# #     file = request.files['file']

# #     file.save('uploads/'+file.filename)

# #     return "File uploaded"


# # UPLOAD_FOLDER = "uploads"


# # @app.route('/files')
# # def show_files():

# #     if not os.path.exists(UPLOAD_FOLDER):
# #         return "Uploads folder doesn't exist"

# #     files = []

# #     for filename in os.listdir(UPLOAD_FOLDER):
# #         path = os.path.join(UPLOAD_FOLDER, filename)

# #         if os.path.isfile(path):
# #             files.append(filename)

# #     return "<br>".join(files)



# # @app.route('/jsoninfy')
# # def jsonifydata():
# #     return jsonify({
# #     "name": "Fazil",
# #     "age": 25
# # })

# # @app.route('/api/string')
# # def get_json_string():
# #     data = {"message": "Hello World"}
# #     return json.dumps(data), 200, {'Content-Type': 'application/json'}

# # @app.route('/receive', methods=['POST'])
# # def receive_data():
# #     # force=True allows parsing even if Content-Type is not application/json
# #     data = request.get_json(force=True)
# #     return jsonify({"received": data})


# # @app.route('/')
# # def index():
# #     return 'Home Page'

# # @app.route('/user/<username>')
# # def profile(username):
# #     return f'Profile of {username}'

# # @app.route('/admin')
# # def admin():
# #     # Redirect to the profile of "john" using url_for
# #     return redirect(url_for('profile', username='john'))





# @app.before_request
# def before_request():
#     print("Request received")
#     print(request.method)
#     print(request.path)


# @app.route("/users")
# def users():
#     return {"message": "Users"}

# @app.after_request
# def after_request(response):
#     response.headers["X-App"] = "My Flask API"
#     return response


# if __name__ == '__main__':
#     app.run(debug=True)



# import time
# from flask import Flask, request

# app = Flask(__name__)


# @app.route('/')
# def mainpage():
#     return "Welcome to main page"


# @app.before_request
# def start_timer():
#     request.start_time = time.time()


# @app.after_request
# def calculate_time(response):
#     duration = time.time() - request.start_time

#     response.headers["X-Response-Time"] = f"{duration:.4f}s"

#     return response