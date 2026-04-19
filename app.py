from functools import wraps
import os
import sqlite3
import re
from datetime import datetime
import requests
import tempfile
import whisper
import traceback

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    flash,
    url_for,
    jsonify,
)

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, cast, String
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.base.exceptions import TwilioRestException
import assemblyai as aai

from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# Helper: normalize phone to E.164
# ------------------------------
def normalize_phone(phone: str) -> str:
    if not phone:
        return ""

    digits = re.sub(r"\D", "", phone)

    if not digits:
        return ""

    # إذا الرقم أصلاً بصيغة دولية (+966...)
    if phone.startswith("+") and len(digits) >= 10:
        return f"+{digits}"

    if digits.startswith("00966"):
        digits = digits[2:]   # تصير 966...
    elif digits.startswith("966"):
        digits = digits       # خليه زي ما هو
    elif digits.startswith("05"):
        digits = "966" + digits[1:]
    elif digits.startswith("5"):
        digits = "966" + digits
    else:
        return ""  # رقم غير معروف

    return f"+{digits}"

# ------------------------------
# App Configuration
# ------------------------------
app = Flask(__name__)
app.secret_key = "secret123"

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'database.db')
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "ablgah.official@gmail.com"
app.config["MAIL_PASSWORD"] = "ollgvdgnfkqodscc"
app.config["MAIL_DEFAULT_SENDER"] = "ablgah.official@gmail.com"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+17622132864")
SUPPORT_AGENT_NUMBER = os.getenv("SUPPORT_AGENT_NUMBER")
if SUPPORT_AGENT_NUMBER:
    SUPPORT_AGENT_NUMBER = normalize_phone(SUPPORT_AGENT_NUMBER)

# Public base URL (for Twilio callbacks). Set this in .env if needed.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

AAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if AAI_API_KEY:
    aai.settings.api_key = AAI_API_KEY

db = SQLAlchemy(app)
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("✅ Twilio client initialized")
else:
    twilio_client = None
    print("⚠️ Twilio credentials missing - voice features disabled")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ADMIN_EMAIL = "reemasaad756@gmail.com"

# Arabic voice for Twilio (Amazon Polly - female Arabic)
ARABIC_VOICE = "Polly.Zeina"
ARABIC_LANG = "arb"

# Load Whisper model once
whisper_model = None
try:
    whisper_model = whisper.load_model("small")
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load Whisper model: {e}")

def transcribe_audio_with_whisper(audio_path):
    if whisper_model is None:
        raise Exception("Whisper model not loaded")
    result = whisper_model.transcribe(audio_path, language="ar", fp16=False)
    return result["text"].strip()

def get_public_url(path):
    """Return a public URL for Twilio callbacks. Prefers PUBLIC_BASE_URL env var."""
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{path}"
    # Fallback to Flask's url_for with _external
    return url_for_external_fallback(path)

def url_for_external_fallback(path):
    """If called within request context, build external URL."""
    try:
        from flask import request as _req
        return f"{_req.url_root.rstrip('/')}{path}"
    except Exception:
        return path

# ------------------------------
# Database Models
# ------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    language = db.Column(db.String(20), default="العربية")
    theme = db.Column(db.String(20), default="light")
    avatar = db.Column(db.String(255), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="جديد")
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class SupportMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    issue_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="جديدة")
    reply = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class CallReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)
    problem_category = db.Column(db.String(50), nullable=True)
    transcript = db.Column(db.Text, nullable=True)
    location_lat = db.Column(db.Float, nullable=True)
    location_lng = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default="pending")
    call_sid = db.Column(db.String(100), nullable=True)
    recording_url = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class EmergencyCall(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    problem_category = db.Column(db.String(50), nullable=False)
    transcript = db.Column(db.Text, nullable=False)
    location = db.Column(db.String(200), nullable=True)
    call_sid = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(20), default="initiated")
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# ------------------------------
# Helper functions
# ------------------------------
def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("يجب تسجيل الدخول أولًا", "error")
            return redirect(url_for("login_page"))
        return view_func(*args, **kwargs)
    return wrapped_view

def admin_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        user = get_current_user()
        if not user or user.email.lower() != ADMIN_EMAIL.lower():
            flash("ليس لديك صلاحية للوصول لهذه الصفحة", "error")
            return redirect(url_for("home"))
        return view_func(*args, **kwargs)
    return wrapped_view

def get_current_user():
    if "user_id" in session:
        return db.session.get(User, session["user_id"])
    return None

def classify_problem(transcript):
    t = transcript.lower()
    if any(word in t for word in ["حريق", "fire", "flame", "burning"]):
        return "حريق"
    if any(word in t for word in ["حادث", "accident", "crash", "collision"]):
        return "حادث"
    if any(word in t for word in ["نزيف", "bleeding", "blood"]):
        return "نزيف"
    if any(word in t for word in ["سرقة", "theft", "robbery", "steal"]):
        return "سرقة"
    if any(word in t for word in ["شجار", "fight", "quarrel"]):
        return "شجار"
    return "عام"

@app.context_processor
def inject_user_preferences():
    user = get_current_user()
    unread_support_count = 0
    unread_admin_support_count = 0
    if user:
        unread_support_count = SupportMessage.query.filter_by(
            user_id=user.id, status="تم الرد", is_read=False
        ).count()
        if user.email.lower() == ADMIN_EMAIL.lower():
            unread_admin_support_count = SupportMessage.query.filter_by(status="جديدة").count()
    return {
        "current_user": user,
        "current_theme": user.theme if user else "light",
        "current_language": "العربية",
        "user": user,
        "unread_support_count": unread_support_count,
        "unread_admin_support_count": unread_admin_support_count
    }

def table_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

def ensure_columns():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'database.db')
    if not os.path.exists(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        if table_exists(conn, "report"):
            cursor.execute("PRAGMA table_info(report)")
            cols = [c[1] for c in cursor.fetchall()]
            if 'created_at' not in cols:
                cursor.execute("ALTER TABLE report ADD COLUMN created_at TIMESTAMP")
                conn.commit()
        if table_exists(conn, "support_message"):
            cursor.execute("PRAGMA table_info(support_message)")
            cols = [c[1] for c in cursor.fetchall()]
            if 'created_at' not in cols:
                cursor.execute("ALTER TABLE support_message ADD COLUMN created_at TIMESTAMP")
                conn.commit()
        if table_exists(conn, "call_report"):
            cursor.execute("PRAGMA table_info(call_report)")
            cols = [c[1] for c in cursor.fetchall()]
            if 'report_type' not in cols:
                cursor.execute("ALTER TABLE call_report ADD COLUMN report_type VARCHAR(50) NOT NULL DEFAULT ''")
                conn.commit()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emergency_call (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                problem_category VARCHAR(50) NOT NULL,
                transcript TEXT NOT NULL,
                location VARCHAR(200),
                call_sid VARCHAR(100),
                status VARCHAR(20) DEFAULT 'initiated',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user(id)
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Column ensure warning: {e}")

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return render_template("home.html", user=get_current_user())

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    identifier = request.form.get("identifier", "").strip()
    password = request.form.get("password", "").strip()
    if not identifier or not password:
        flash("يرجى تعبئة جميع الحقول", "error")
        return redirect(url_for("login_page"))
    user = User.query.filter((User.email == identifier) | (User.phone == identifier)).first()
    if user is None or not check_password_hash(user.password, password):
        flash("يرجى التاكد من البيانات", "error")
        return redirect(url_for("login_page"))
    session["user_id"] = user.id
    session["user_name"] = user.name
    flash("تم تسجيل الدخول بنجاح", "success")
    return redirect(url_for("home"))

@app.route("/admin-login", methods=["POST"])
def admin_login():
    identifier = request.form.get("identifier", "").strip()
    password = request.form.get("password", "").strip()
    if not identifier or not password:
        flash("يرجى تعبئة جميع الحقول", "error")
        return redirect(url_for("login_page"))
    user = User.query.filter((User.email == identifier) | (User.phone == identifier)).first()
    if user is None or not check_password_hash(user.password, password):
        flash("يرجى التاكد من البيانات", "error")
        return redirect(url_for("login_page"))
    if user.email.lower() != ADMIN_EMAIL.lower():
        flash("ليس لديك صلاحية أدمن", "error")
        return redirect(url_for("login_page"))
    session["user_id"] = user.id
    session["user_name"] = user.name
    flash("مرحباً! تم تسجيل دخولك كأدمن", "success")
    return redirect(url_for("home"))

@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    password = request.form.get("password", "").strip()
    if not name or not email or not phone or not password:
        flash("يرجى تعبئة جميع الحقول", "error")
        return redirect(url_for("register_page"))
    phone = normalize_phone(phone)
    if User.query.filter_by(email=email).first():
        flash("هذا البريد مسجل من قبل", "error")
        return redirect(url_for("register_page"))
    if User.query.filter_by(phone=phone).first():
        flash("رقم الجوال مسجل من قبل", "error")
        return redirect(url_for("register_page"))
    hashed_password = generate_password_hash(password)
    is_admin = (email.lower() == ADMIN_EMAIL.lower())
    new_user = User(name=name, email=email, password=hashed_password, phone=phone,
                    language="العربية", theme="light", avatar="", is_admin=is_admin)
    db.session.add(new_user)
    db.session.commit()
    flash("تم إنشاء الحساب بنجاح، يمكنك تسجيل الدخول الآن", "success")
    return redirect(url_for("login_page"))

@app.route("/forgot-password", methods=["GET"])
def forgot_password_page():
    return render_template("forgot_password.html")

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    email = request.form.get("email", "").strip()
    user = User.query.filter_by(email=email).first()
    if user is None:
        flash("هذا البريد غير مسجل", "error")
        return redirect(url_for("forgot_password_page"))
    try:
        token = serializer.dumps(user.email, salt="reset-password-salt")
        reset_link = url_for("reset_password_page", token=token, _external=True)
        msg = Message(subject="إعادة تعيين كلمة المرور - منصة أبلغ", recipients=[user.email])
        msg.body = f"""مرحبًا {user.name}،

تلقينا طلبًا لإعادة تعيين كلمة المرور الخاصة بحسابك في منصة أبلغ.

يمكنك إعادة تعيين كلمة المرور من خلال الرابط التالي:
{reset_link}

ملاحظة:
هذا الرابط صالح لمدة ساعة واحدة فقط.

إذا لم تطلب إعادة تعيين كلمة المرور، يمكنك تجاهل هذه الرسالة.

مع التحية،
فريق منصة أبلغ"""
        mail.send(msg)
        flash("تم إرسال رابط إعادة تعيين كلمة المرور إلى بريدك الإلكتروني", "success")
        return redirect(url_for("login_page"))
    except Exception:
        flash("حدث خطأ أثناء إرسال البريد الإلكتروني.", "error")
        return redirect(url_for("forgot_password_page"))

@app.route("/reset-password/<token>", methods=["GET"])
def reset_password_page(token):
    try:
        email = serializer.loads(token, salt="reset-password-salt", max_age=3600)
        return render_template("reset_password.html", token=token, email=email)
    except SignatureExpired:
        flash("انتهت صلاحية رابط إعادة التعيين", "error")
        return redirect(url_for("forgot_password_page"))
    except BadTimeSignature:
        flash("رابط إعادة التعيين غير صالح", "error")
        return redirect(url_for("forgot_password_page"))

@app.route("/reset-password/<token>", methods=["POST"])
def reset_password(token):
    try:
        email = serializer.loads(token, salt="reset-password-salt", max_age=3600)
    except (SignatureExpired, BadTimeSignature):
        flash("الرابط غير صالح أو منتهي الصلاحية", "error")
        return redirect(url_for("forgot_password_page"))
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()
    if not password or not confirm_password:
        flash("يرجى تعبئة جميع الحقول", "error")
        return render_template("reset_password.html", token=token, email=email)
    if password != confirm_password:
        flash("كلمتا المرور غير متطابقتين", "error")
        return render_template("reset_password.html", token=token, email=email)
    user = User.query.filter_by(email=email).first()
    if user is None:
        flash("المستخدم غير موجود", "error")
        return redirect(url_for("forgot_password_page"))
    user.password = generate_password_hash(password)
    db.session.commit()
    flash("تم تغيير كلمة المرور بنجاح، يمكنك تسجيل الدخول الآن", "success")
    return redirect(url_for("login_page"))

@app.route("/report", methods=["GET"])
@login_required
def report_page():
    return render_template("report.html")

@app.route("/report/<string:report_type>", methods=["GET"])
@login_required
def report_form(report_type):
    return render_template("report_form.html", type=report_type)

@app.route("/submit", methods=["POST"])
@login_required
def submit_report():
    report_type = request.form.get("type", "").strip()
    description = request.form.get("description", "").strip()
    if not description:
        flash("يرجى تعبئة وصف البلاغ", "error")
        return redirect(url_for("report_page"))
    final_type = report_type if report_type else "عام"
    new_report = Report(type=final_type, description=description, status="جديد", user_id=session["user_id"])
    db.session.add(new_report)
    db.session.commit()
    flash("تم إرسال البلاغ بنجاح", "success")
    return redirect(url_for("success_page"))

@app.route("/success")
@login_required
def success_page():
    reports = Report.query.filter_by(user_id=session["user_id"]).all()
    return render_template("success.html", reports=reports)

@app.route("/dashboard")
@admin_required
def dashboard():
    reports = Report.query.all()
    users = User.query.all()
    return render_template("dashboard.html", reports=reports, users=users)

@app.route("/my-reports")
@login_required
def my_reports():
    regular_reports = Report.query.filter_by(user_id=session["user_id"]).all()
    call_reports = CallReport.query.filter_by(user_id=session["user_id"]).all()
    combined = []
    for r in regular_reports:
        combined.append({'id': r.id, 'type': r.type, 'description': r.description, 'status': r.status,
                         'created_at': r.created_at, 'is_call': False, 'call_id': None, 'transcript': None})
    for cr in call_reports:
        combined.append({'id': cr.id, 'type': cr.report_type,
                         'description': f"[بلاغ هاتفي] {cr.transcript[:150] if cr.transcript else 'جاري المعالجة...'}",
                         'status': 'جديد', 'created_at': cr.created_at, 'is_call': True,
                         'call_id': cr.id, 'transcript': cr.transcript})
    combined.sort(key=lambda x: x['created_at'] or datetime.min, reverse=True)
    return render_template("my_reports.html", reports=combined)

@app.route("/call-details/<int:call_id>")
@login_required
def call_report_details(call_id):
    call_report = CallReport.query.get_or_404(call_id)
    if call_report.user_id != session["user_id"] and not get_current_user().is_admin:
        flash("ليس لديك صلاحية لعرض هذا البلاغ", "error")
        return redirect(url_for("my_reports"))
    if not call_report.transcript and call_report.call_sid and twilio_client:
        try:
            recordings = twilio_client.recordings.list(call_sid=call_report.call_sid)
            if recordings:
                recording = recordings[0]
                recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording.sid}.mp3"
                auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                resp = requests.get(recording_url, auth=auth, stream=True)
                if resp.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        for chunk in resp.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name
                    try:
                        transcript = transcribe_audio_with_whisper(tmp_path)
                        category = classify_problem(transcript)
                        call_report.transcript = transcript
                        call_report.problem_category = category
                        call_report.status = "transcribed"
                        db.session.commit()
                        existing = Report.query.filter_by(user_id=call_report.user_id,
                                                          description=f"[مكالمة هاتفية] {transcript[:200]}").first()
                        if not existing:
                            normal_report = Report(type=call_report.report_type,
                                                   description=f"[مكالمة هاتفية] {transcript[:200]}",
                                                   status="جديد", user_id=call_report.user_id)
                            db.session.add(normal_report)
                            db.session.commit()
                    except Exception as e:
                        print(f"Whisper error: {e}")
                    finally:
                        os.unlink(tmp_path)
        except Exception as e:
            print(f"Error in call-details: {e}")
    return render_template("call_details.html", call=call_report)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/about-us')
def about_us():
    return render_template('about_us.html')

@app.route("/details/<int:report_id>")
@login_required
def details(report_id):
    report = Report.query.get_or_404(report_id)
    return render_template("details.html", report=report)

@app.route("/update/<int:report_id>", methods=["POST"])
@admin_required
def update_report(report_id):
    report = Report.query.get_or_404(report_id)
    new_status = request.form.get("new_status", "")
    if new_status == "processing":
        report.status = "قيد المعالجة"
    elif new_status == "closed":
        report.status = "مغلق"
    elif new_status == "new":
        report.status = "جديد"
    else:
        flash("حالة غير صالحة", "error")
        return redirect(url_for("dashboard"))
    db.session.commit()
    flash("تم تحديث حالة البلاغ بنجاح", "success")
    return redirect(url_for("dashboard"))

@app.route("/delete/<int:report_id>", methods=["POST"])
@admin_required
def delete_report(report_id):
    report = Report.query.get_or_404(report_id)
    db.session.delete(report)
    db.session.commit()
    flash("تم حذف البلاغ بنجاح", "success")
    return redirect(url_for("dashboard"))

@app.route("/profile", methods=["GET"])
@login_required
def profile_page():
    user = User.query.get_or_404(session["user_id"])
    return render_template("profile.html", user=user)

@app.route("/profile", methods=["POST"])
@login_required
def update_profile():
    user = User.query.get_or_404(session["user_id"])
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    if not name or not email:
        flash("يرجى تعبئة الاسم والبريد الإلكتروني", "error")
        return redirect(url_for("profile_page"))
    existing_user = User.query.filter(User.email == email, User.id != user.id).first()
    if existing_user:
        flash("هذا البريد الإلكتروني مستخدم من قبل", "error")
        return redirect(url_for("profile_page"))
    user.name = name
    user.email = email
    if phone:
        user.phone = normalize_phone(phone)
    avatar_file = request.files.get("avatar")
    if avatar_file and avatar_file.filename:
        filename = secure_filename(avatar_file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        avatar_file.save(filepath)
        user.avatar = f"uploads/{filename}"
    db.session.commit()
    session["user_name"] = user.name
    flash("تم تحديث الملف الشخصي بنجاح", "success")
    return redirect(url_for("profile_page"))

@app.route("/settings", methods=["GET"])
@login_required
def settings_page():
    user = User.query.get_or_404(session["user_id"])
    return render_template("settings.html", user=user)

@app.route("/settings", methods=["POST"])
@login_required
def update_settings():
    user = User.query.get_or_404(session["user_id"])
    theme = request.form.get("theme", "").strip()
    current_password = request.form.get("current_password", "").strip()
    new_password = request.form.get("new_password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()
    user.language = "العربية"
    user.theme = theme if theme else "light"
    if current_password or new_password or confirm_password:
        if not current_password or not new_password or not confirm_password:
            flash("لتغيير كلمة المرور يجب تعبئة جميع حقول كلمة المرور", "error")
            return redirect(url_for("settings_page"))
        if not check_password_hash(user.password, current_password):
            flash("كلمة المرور الحالية غير صحيحة", "error")
            return redirect(url_for("settings_page"))
        if new_password != confirm_password:
            flash("كلمتا المرور الجديدتان غير متطابقتين", "error")
            return redirect(url_for("settings_page"))
        user.password = generate_password_hash(new_password)
    db.session.commit()
    flash("تم تحديث الإعدادات بنجاح", "success")
    return redirect(url_for("settings_page"))

@app.route("/support", methods=["GET", "POST"])
def support_page():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        issue_type = request.form.get("issue_type", "").strip()
        message = request.form.get("message", "").strip()
        if not name or not email or not issue_type or not message:
            flash("يرجى تعبئة جميع الحقول", "error")
            return redirect(url_for("support_page"))
        new_message = SupportMessage(name=name, email=email, issue_type=issue_type, message=message,
                                     user_id=session.get("user_id"))
        db.session.add(new_message)
        db.session.commit()
        flash("تم إرسال رسالتك بنجاح، وسيتم الرد عليك من داخل الموقع", "success")
        return redirect(url_for("support_page"))
    user_messages = []
    if "user_id" in session:
        user_messages = SupportMessage.query.filter_by(user_id=session["user_id"]).order_by(SupportMessage.id.desc()).all()
        unread_messages = SupportMessage.query.filter_by(user_id=session["user_id"], status="تم الرد", is_read=False).all()
        for msg in unread_messages:
            msg.is_read = True
        if unread_messages:
            db.session.commit()
    return render_template("support.html", user_messages=user_messages)

@app.route("/admin-support")
@admin_required
def admin_support():
    messages = SupportMessage.query.order_by(SupportMessage.id.desc()).all()
    return render_template("admin_support.html", messages=messages)

@app.route("/support/reply/<int:id>", methods=["POST"])
@admin_required
def reply_support(id):
    msg = SupportMessage.query.get_or_404(id)
    reply_text = request.form.get("reply", "").strip()
    if not reply_text:
        flash("يرجى كتابة الرد أولًا", "error")
        return redirect(url_for("admin_support"))
    msg.reply = reply_text
    msg.status = "تم الرد"
    msg.is_read = False
    db.session.commit()
    flash("تم إرسال الرد داخل الموقع بنجاح", "success")
    return redirect(url_for("admin_support"))

@app.route("/support/update/<int:id>/<string:new_status>")
@admin_required
def update_support_status(id, new_status):
    msg = SupportMessage.query.get_or_404(id)
    if new_status == "replied":
        msg.status = "تم الرد"
    elif new_status == "closed":
        msg.status = "مغلقة"
    elif new_status == "new":
        msg.status = "جديدة"
    else:
        flash("حالة غير صالحة", "error")
        return redirect(url_for("admin_support"))
    db.session.commit()
    return redirect(url_for("admin_support"))

@app.route("/support/delete/<int:id>")
@admin_required
def delete_support(id):
    msg = SupportMessage.query.get_or_404(id)
    db.session.delete(msg)
    db.session.commit()
    flash("تم حذف الرسالة بنجاح", "success")
    return redirect(url_for("admin_support"))

@app.route("/notifications")
@login_required
def notifications_page():
    notifications = SupportMessage.query.filter_by(user_id=session["user_id"], status="تم الرد").order_by(SupportMessage.id.desc()).all()
    unread_messages = SupportMessage.query.filter_by(user_id=session["user_id"], status="تم الرد", is_read=False).all()
    for msg in unread_messages:
        msg.is_read = True
    if unread_messages:
        db.session.commit()
    return render_template("notifications.html", notifications=notifications)

@app.route("/admin-notifications")
@admin_required
def admin_notifications_page():
    notifications = SupportMessage.query.order_by(SupportMessage.id.desc()).all()
    new_messages = SupportMessage.query.filter_by(status="جديدة").all()
    for msg in new_messages:
        msg.status = "مقروءة"
    if new_messages:
        db.session.commit()
    return render_template("admin_notifications.html", notifications=notifications, messages=notifications)

@app.route("/search-suggestions")
def search_suggestions():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])
    user = get_current_user()
    results = []
    static_pages = [
        {"title": "الرئيسية", "subtitle": "الانتقال إلى الصفحة الرئيسية", "status": "صفحة", "url": url_for("home"), "keywords": ["الرئيسية", "home", "الصفحة الرئيسية"]},
        {"title": "من نحن", "subtitle": "التعريف بالمنصة وأهدافها", "status": "صفحة", "url": url_for("about"), "keywords": ["من نحن", "عن الفريق", "نبذة", "تعريف"]},
        {"title": "عن المنصة", "subtitle": "معلومات عن منصة أبلغ وآلية عملها", "status": "صفحة", "url": url_for("about_us"), "keywords": ["عن المنصة", "المنصة", "أبلغ", "الية العمل", "آلية العمل"]},
    ]
    if user:
        static_pages.extend([
            {"title": "ملفي الشخصي", "subtitle": "عرض وتعديل بيانات الحساب", "status": "حساب", "url": url_for("profile_page"), "keywords": ["ملفي الشخصي", "الملف الشخصي", "الملف", "البروفايل", "profile"]},
            {"title": "الإعدادات", "subtitle": "تعديل الثيم وكلمة المرور", "status": "إعدادات", "url": url_for("settings_page"), "keywords": ["الإعدادات", "اعدادات", "الثيم", "كلمة المرور", "settings"]},
        ])
        if user.email.lower() == ADMIN_EMAIL.lower():
            static_pages.extend([
                {"title": "لوحة التحكم", "subtitle": "إدارة البلاغات والمستخدمين", "status": "أدمن", "url": url_for("dashboard"), "keywords": ["لوحة التحكم", "لوحة الادمن", "لوحة الأدمن", "البلاغات", "المستخدمين", "dashboard"]},
                {"title": "رسائل الدعم", "subtitle": "عرض رسائل الدعم المرسلة من المستخدمين", "status": "أدمن", "url": url_for("admin_support"), "keywords": ["رسائل الدعم", "الدعم", "الدعم الفني", "admin support"]},
                {"title": "إشعارات الأدمن", "subtitle": "عرض إشعارات ورسائل الدعم الجديدة", "status": "أدمن", "url": url_for("admin_notifications_page"), "keywords": ["إشعارات", "اشعارات", "إشعارات الأدمن", "جرس", "notifications"]},
            ])
        else:
            static_pages.extend([
                {"title": "تقديم بلاغ", "subtitle": "اختيار نوع البلاغ وإرساله", "status": "بلاغ", "url": url_for("report_page"), "keywords": ["تقديم بلاغ", "بلاغ", "ارسال بلاغ", "إرسال بلاغ", "report"]},
                {"title": "بلاغاتي", "subtitle": "عرض جميع البلاغات الخاصة بك", "status": "بلاغ", "url": url_for("my_reports"), "keywords": ["بلاغاتي", "بلاغاتي الخاصة", "تقاريري", "my reports"]},
                {"title": "الدعم الفني", "subtitle": "إرسال رسالة للدعم ومتابعة الردود", "status": "دعم", "url": url_for("support_page"), "keywords": ["الدعم الفني", "الدعم", "رسالة", "مساعدة", "support"]},
                {"title": "الإشعارات", "subtitle": "عرض الردود والإشعارات الجديدة", "status": "إشعارات", "url": url_for("notifications_page"), "keywords": ["الإشعارات", "اشعارات", "جرس", "notifications"]},
            ])
    else:
        static_pages.extend([
            {"title": "تسجيل الدخول", "subtitle": "الدخول إلى الحساب", "status": "حساب", "url": url_for("login_page"), "keywords": ["تسجيل الدخول", "دخول", "login"]},
            {"title": "إنشاء حساب", "subtitle": "تسجيل مستخدم جديد", "status": "حساب", "url": url_for("register_page"), "keywords": ["إنشاء حساب", "تسجيل", "حساب جديد", "register"]},
        ])
    for page in static_pages:
        searchable_text = " ".join([page["title"], page["subtitle"]] + page["keywords"]).lower()
        if query in searchable_text:
            results.append({"title": page["title"], "subtitle": page["subtitle"], "status": page["status"], "url": page["url"]})
    reports = db.session.query(Report, User).outerjoin(User, Report.user_id == User.id).filter(
        or_(cast(Report.id, String).ilike(f"%{query}%"), Report.type.ilike(f"%{query}%"),
            Report.description.ilike(f"%{query}%"), Report.status.ilike(f"%{query}%"),
            User.name.ilike(f"%{query}%"), User.email.ilike(f"%{query}%"))
    ).limit(6).all()
    for report, report_user in reports:
        owner_name = report_user.name if report_user else "مستخدم"
        results.append({"title": f"بلاغ رقم {report.id} - {report.type}", "subtitle": f"{report.description[:60]} | مقدم البلاغ: {owner_name}",
                        "status": report.status, "url": url_for("details", report_id=report.id) if user and user.email.lower() == ADMIN_EMAIL.lower() else url_for("my_reports")})
    unique_results = []
    seen = set()
    for item in results:
        key = (item["title"], item["url"])
        if key not in seen:
            seen.add(key)
            unique_results.append(item)
    return jsonify(unique_results[:8])

@app.route("/logout")
def logout():
    session.clear()
    flash("تم تسجيل الخروج", "success")
    return redirect(url_for("login_page"))

# ------------------------------
# CALL REPORT ROUTES - REWORKED
# ------------------------------
@app.route("/save-location", methods=["POST"])
@login_required
def save_location():
    data = request.get_json()
    lat = data.get("lat")
    lng = data.get("lng")
    if lat is not None and lng is not None:
        session["temp_lat"] = lat
        session["temp_lng"] = lng
        return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route("/initiate-call-report", methods=["POST"])
@login_required
def initiate_call_report():
    """Initiate a Twilio call to the user for voice reporting.
    Since Twilio phone number's webhook is configured to /voice-incoming,
    we pass the report_id via URL params so /voice-incoming knows the context.
    """
    try:
        data = request.get_json()
        report_type = data.get("type")
        if not report_type:
            return jsonify({"error": "نوع البلاغ مطلوب"}), 400

        user = get_current_user()
        if not user:
            return jsonify({"error": "يجب تسجيل الدخول"}), 401

        if not user.phone:
            return jsonify({"error": "رقم الهاتف غير موجود. يرجى إضافته في الملف الشخصي."}), 400

        user_phone = normalize_phone(user.phone)

        print("USER PHONE FROM PROFILE:", user.phone)
        print("NORMALIZED USER PHONE:", user_phone)

print("=== INITIATE CALL REPORT ===")
print("USER PHONE FROM PROFILE:", user.phone)
print("NORMALIZED USER PHONE:", user_phone)

        if not user_phone:
            return jsonify({"error": "رقم الهاتف غير صالح. يرجى تحديثه بصيغة دولية."}), 400

        lat = session.get("temp_lat")
        lng = session.get("temp_lng")
        if not lat or not lng:
            return jsonify({"error": "يرجى تحديد موقعك باستخدام زر 'إرسال الموقع الحالي' أولاً"}), 400

        call_report = CallReport(
            user_id=user.id,
            report_type=report_type,
            location_lat=lat,
            location_lng=lng,
            status="pending"
        )
        db.session.add(call_report)
        db.session.commit()

        if not twilio_client:
            return jsonify({"error": "خدمة المكالمات غير متاحة حالياً (Twilio غير مهيأ)"}), 500

        base_url = PUBLIC_BASE_URL if PUBLIC_BASE_URL else request.url_root.rstrip("/")
        webhook_url = f"{base_url}/voice-incoming?report_id={call_report.id}"
        
print("TWILIO FROM NUMBER:", TWILIO_PHONE_NUMBER)
print("WEBHOOK URL:", webhook_url)


        call = twilio_client.calls.create(
            url=webhook_url,
            to=user_phone,
            from_=TWILIO_PHONE_NUMBER,
            timeout=30
        )
        call_report.call_sid = call.sid
        db.session.commit()
        return jsonify({"success": True, "message": "جاري الاتصال بك...", "call_id": call_report.id})

    except TwilioRestException as e:
        error_msg = f"خطأ في Twilio: {e.msg}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        print(f"Error in initiate_call_report: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"حدث خطأ: {str(e)}"}), 500

@app.route("/voice-incoming", methods=["GET", "POST"])
def voice_incoming():
    """
    Unified entry point for Twilio calls.
    - If called with ?report_id=XX (from initiate_call_report), it plays the
      report-gathering flow: greets in Arabic, asks user to press any key, then records.
    - Otherwise (inbound call without context), it plays a generic Arabic greeting.
    """
    report_id = request.args.get("report_id") or request.form.get("report_id")
    response = VoiceResponse()

    if report_id:
        # Outbound call flow for a specific report
        base_url = PUBLIC_BASE_URL if PUBLIC_BASE_URL else request.url_root.rstrip("/")

        # Greet in Arabic and ask user to press any key to begin recording
        gather = Gather(
            num_digits=1,
            timeout=10,
            action=f"{base_url}/voice-start-recording?report_id={report_id}",
            method="POST",
            language=ARABIC_LANG
        )
        gather.say(
            "مرحباً بك في منصة أبلغ. "
            "لتسجيل بلاغك الصوتي، يرجى الضغط على أي رقم من لوحة المفاتيح للبدء بالتسجيل.",
            voice=ARABIC_VOICE,
            language=ARABIC_LANG
        )
        response.append(gather)

        # If no key pressed, retry once
        response.say(
            "لم نستقبل أي مدخل. شكراً لاتصالك بمنصة أبلغ. مع السلامة.",
            voice=ARABIC_VOICE,
            language=ARABIC_LANG
        )
        response.hangup()
    else:
        # Generic inbound call
        response.say(
            "مرحباً بك في منصة أبلغ. الرجاء تسجيل الدخول إلى الموقع لاستخدام خدمة المكالمات.",
            voice=ARABIC_VOICE,
            language=ARABIC_LANG
        )
        response.hangup()

    return str(response), 200, {'Content-Type': 'text/xml'}


@app.route("/voice-start-recording", methods=["POST"])
def voice_start_recording():
    """After user presses a key, prompt them and start recording."""
    report_id = request.args.get("report_id") or request.form.get("report_id")
    response = VoiceResponse()

    base_url = PUBLIC_BASE_URL if PUBLIC_BASE_URL else request.url_root.rstrip("/")

    response.say(
        "بعد سماع صوت التنبيه، يرجى وصف البلاغ بوضوح، ثم اضغط على مفتاح المربع للإنهاء أو انتظر حتى انتهاء المدة.",
        voice=ARABIC_VOICE,
        language=ARABIC_LANG
    )
    response.record(
        action=f"{base_url}/process-recording/{report_id}",
        method="POST",
        max_length=60,
        timeout=5,
        play_beep=True,
        finish_on_key="#",
        trim="trim-silence"
    )
    response.say(
        "لم يتم تسجيل أي رسالة. شكراً لك، مع السلامة.",
        voice=ARABIC_VOICE,
        language=ARABIC_LANG
    )
    response.hangup()
    return str(response), 200, {'Content-Type': 'text/xml'}


@app.route("/process-recording/<int:report_id>", methods=["POST"])
def process_recording(report_id):
    recording_url = request.form.get("RecordingUrl")
    call_report = CallReport.query.get(report_id)
    if not call_report:
        return "Report not found", 404

    call_report.recording_url = recording_url
    call_report.status = "downloading"
    db.session.commit()

    transcript = ""
    category = "عام"

    if recording_url and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            audio_url = recording_url + ".mp3"
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            resp = requests.get(audio_url, auth=auth, stream=True)
            if resp.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    for chunk in resp.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name
                try:
                    transcript = transcribe_audio_with_whisper(tmp_path)
                    category = classify_problem(transcript)
                    call_report.status = "transcribed"
                except Exception as e:
                    print(f"Whisper error: {e}")
                    call_report.status = "failed"
                finally:
                    os.unlink(tmp_path)
            else:
                call_report.status = "failed"
        except Exception as e:
            call_report.status = "error"
            print(f"Transcription error: {e}")
    else:
        call_report.status = "no_api_key"

    call_report.transcript = transcript
    call_report.problem_category = category
    db.session.commit()

    normal_report = Report(
        type=call_report.report_type,
        description=f"[مكالمة هاتفية] {transcript[:200]}" if transcript else "[لم يتم التعرف على الصوت]",
        status="جديد",
        user_id=call_report.user_id
    )
    db.session.add(normal_report)
    db.session.commit()

    response = VoiceResponse()
    response.say(
        "تم استلام بلاغك بنجاح. سيتم مراجعته والرد عليك في أقرب وقت. شكراً لاستخدامك منصة أبلغ. مع السلامة.",
        voice=ARABIC_VOICE,
        language=ARABIC_LANG
    )
    response.hangup()
    return str(response), 200, {'Content-Type': 'text/xml'}


# ------------------------------
# Emergency Voice Report (chatbot)
# ------------------------------
@app.route("/emergency-voice-report", methods=["POST"])
@login_required
def emergency_voice_report():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty audio file"}), 400

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    audio_file.save(temp_audio.name)
    temp_audio.close()

    transcript = ""
    category = "عام"
    try:
        transcript = transcribe_audio_with_whisper(temp_audio.name)
        category = classify_problem(transcript)
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        traceback.print_exc()
        os.unlink(temp_audio.name)
        return jsonify({"error": f"فشل التعرف على الصوت: {str(e)}", "transcript": ""}), 500

    os.unlink(temp_audio.name)

    response_data = {
        "transcript": transcript,
        "category": category,
        "emergency": category != "عام"
    }

    user = get_current_user()
    lat = session.get("temp_lat")
    lng = session.get("temp_lng")
    location_str = f"lat:{lat},lng:{lng}" if lat and lng else None

    emergency = EmergencyCall(
        user_id=user.id if user else None,
        problem_category=category,
        transcript=transcript,
        location=location_str,
        status="initiated"
    )
    db.session.add(emergency)
    db.session.commit()
    response_data["emergency_id"] = emergency.id
    print(f"✅ EmergencyCall record {emergency.id} saved to database.")

    # If emergency, initiate call to support agent - in ARABIC via Polly.Zeina
    if category != "عام" and SUPPORT_AGENT_NUMBER and twilio_client:
        user_name = user.name if user else "مستخدم مجهول"
        location_text = ""
        if lat and lng:
            location_text = f" الموقع الجغرافي: خط العرض {lat}، خط الطول {lng}."

        # Arabic emergency message
        message_body = (
            f"تنبيه طارئ من منصة أبلغ. "
            f"المستخدم {user_name} أبلغ عن حالة طارئة من نوع {category}. "
            f"يرجى التعامل معها بشكل عاجل.{location_text} "
            f"تفاصيل البلاغ: {transcript[:200]}. "
            f"نكرر: حالة طارئة من نوع {category}، يرجى التدخل السريع."
        )

        # Build proper TwiML with Arabic voice (Polly.Zeina)
        twiml_response = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response>'
            f'<Say voice="{ARABIC_VOICE}" language="{ARABIC_LANG}">{message_body}</Say>'
            f'<Pause length="1"/>'
            f'<Say voice="{ARABIC_VOICE}" language="{ARABIC_LANG}">'
            f'سيتم إعادة الرسالة مرة أخرى.'
            f'</Say>'
            f'<Say voice="{ARABIC_VOICE}" language="{ARABIC_LANG}">{message_body}</Say>'
            f'</Response>'
        )

        try:
            agent_number = normalize_phone(SUPPORT_AGENT_NUMBER)
print("SUPPORT_AGENT_NUMBER RAW:", SUPPORT_AGENT_NUMBER)
print("SUPPORT_AGENT_NUMBER NORMALIZED:", agent_number)
print("TWILIO FROM NUMBER:", TWILIO_PHONE_NUMBER)

            print("=== EMERGENCY VOICE REPORT ===")
            call = twilio_client.calls.create(
                to=agent_number,
                from_=TWILIO_PHONE_NUMBER,
                twiml=twiml_response
            )
            emergency.call_sid = call.sid
            emergency.status = "completed"
            db.session.commit()
            response_data["call_initiated"] = True
            response_data["call_sid"] = call.sid
            print(f"✅ Twilio call initiated: {call.sid}")
        except TwilioRestException as e:
            print(f"Twilio call error: {e}")
            emergency.status = "failed"
            db.session.commit()
            response_data["call_initiated"] = False
            response_data["call_error"] = f"خطأ Twilio: {e.msg}"
        except Exception as e:
            print(f"General error: {e}")
            emergency.status = "failed"
            db.session.commit()
            response_data["call_initiated"] = False
            response_data["call_error"] = str(e)
    else:
        response_data["call_initiated"] = False
        if not SUPPORT_AGENT_NUMBER:
            response_data["call_error"] = "رقم الدعم غير مهيأ"
        elif not twilio_client:
            response_data["call_error"] = "خدمة المكالمات غير متاحة"

    return jsonify(response_data)

# ------------------------------
# Database initialization
# ------------------------------
with app.app_context():
    instance_dir = os.path.dirname(db_path)
    os.makedirs(instance_dir, exist_ok=True)
    db.create_all()
    print("✅ Database tables created (or already exist).")
    ensure_columns()
    admin = User.query.filter_by(email=ADMIN_EMAIL).first()
    if admin and not admin.is_admin:
        admin.is_admin = True
        db.session.commit()
        print(f"✅ Admin flag set for {ADMIN_EMAIL}")
    elif not admin:
        print(f"⚠️ Admin user {ADMIN_EMAIL} not found. Please register first.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
