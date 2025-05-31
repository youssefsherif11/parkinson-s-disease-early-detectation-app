from flask import Flask, request, jsonify, render_template, redirect, url_for, session, g
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = "mysecretkey"
app.config["MONGO_URI"] = "mongodb://localhost:27017/mydatabase"
mongo = PyMongo(app)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = session.get("token")
        if not token:
            return redirect(url_for("login"))
        try:
            decoded_token = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            g.user = decoded_token["username"]
        except jwt.ExpiredSignatureError:
            return redirect(url_for("login"))
        except jwt.InvalidTokenError:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

@app.route("/")
def home():
    return render_template("homep.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not full_name or not username or not password:
            return render_template("signuppage.html", error="يجب ملء جميع الحقول")

        existing_user = mongo.db.users.find_one({"username": username})
        if existing_user:
            return render_template("signuppage.html", error="اسم المستخدم مستخدم بالفعل")

        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({
            "full_name": full_name,
            "username": username,
            "password": hashed_password,
            "recordings": []
        })

        token = jwt.encode(
            {"username": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
            app.config["SECRET_KEY"], algorithm="HS256"
        )
        session["token"] = token

        return redirect(url_for("dashboard"))

    return render_template("signuppage.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = mongo.db.users.find_one({"username": username})
        if not user or not check_password_hash(user["password"], password):
            return render_template("loginpage.html", error="اسم المستخدم أو كلمة المرور غير صحيحة")

        session.pop("token", None)

        token = jwt.encode(
            {"username": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
            app.config["SECRET_KEY"], algorithm="HS256"
        )
        session["token"] = token

        return redirect(url_for("dashboard"))

    return render_template("loginpage.html")

@app.route("/dashboard")
@token_required
def dashboard():
    user_data = mongo.db.users.find_one({"username": g.user})
    if not user_data:
        return redirect(url_for("login"))

    full_name = user_data.get("full_name", "User")

    # ✅ هنا نستخدم username كنص مش ObjectId
    total_records = mongo.db.voice_records.count_documents({"user_id": g.user})
    under_examination_count = mongo.db.voice_records.count_documents({"user_id": g.user, "status": "Under examination"})
    checked_count = mongo.db.voice_records.count_documents({"user_id": g.user, "status": "Checked"})

    return render_template("Patient_Dashboard.html", user=full_name,
                           total_records=total_records,
                           under_examination_count=under_examination_count,
                           checked_count=checked_count)

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("token", None)
    return redirect(url_for("login"))

# تحميل النموذج
model = load_model("parkinson_model.h5", compile=False)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("optimal_threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

@app.route('/predict_mdvp', methods=['POST'])
@token_required
def predict_mdvp():
    try:
        data = request.json
        features = list(data.values())
        features = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features)
        prob = model.predict(scaled)[0][0]
        prediction = "Parkinson" if prob >= 0.5 else "Healthy"

        # ✅ هنا نحفظ التسجيل بالـ username مباشرة
        mongo.db.voice_records.insert_one({
            "user_id": g.user,
            "status": "Checked",
            "probability": float(prob),
            "prediction": prediction,
            "timestamp": datetime.datetime.utcnow()
        })

        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": round(float(prob), 4)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/voice_test")
@token_required
def voice_test_page():
    return render_template("voice_test_page.html")

if __name__ == '__main__':
    app.run(debug=True)
