# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, abort
import re
from functools import wraps
from pymongo import MongoClient
from config import Config
import spacy
from transformers import pipeline
from twilio.rest import Client
from datetime import datetime
import os
from bson import ObjectId
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv

# Importer le module d'inférence du chatbot ML/TF-IDF
from chatbot_inference import ChatbotInference

# Charger les variables d'environnement (PUBLIC_URL, ADVISOR_PHONE, etc.)
load_dotenv()
PUBLIC_URL = os.getenv("PUBLIC_URL")

# Initialiser l'application Flask
app = Flask(__name__)
app.config.from_object(Config)

# --------------------------
# Contexte global pour les templates
# --------------------------
@app.context_processor
def inject_user():
    """
    Injecte current_user dans les templates (username) + une liste des endpoints disponibles
    et une fonction safe_url pour construire des urls sans provoquer BuildError.
    """
    # available_endpoints : ensemble des endpoints valides dans l'application
    available_endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}

    def safe_url(endpoint, **kwargs):
        """
        Essaye de construire une URL pour `endpoint`.
        - Si l'endpoint existe, retourne url_for(...)
        - Sinon retourne None (le template peut afficher un fallback ou masquer le lien)
        """
        try:
            if endpoint in available_endpoints:
                return url_for(endpoint, **kwargs)
            return None
        except Exception:
            return None

    return {
        'current_user': session.get('username'),
        'available_endpoints': available_endpoints,
        'safe_url': safe_url
    }

# Garantir une clé secrète pour sessions (utilise Config si présente sinon aléatoire)
app.secret_key = app.config.get('SECRET_KEY') or os.urandom(24)

# --------------------------
# Initialisation de Flask-Mail
# --------------------------
mail = Mail(app)

# --------------------------
# Connexion à MongoDB
# --------------------------
client = MongoClient(app.config['MONGO_URI'])
db = client['chatbot']
users_collection = db['users']
appointments_collection = db['Rappel']
messages_collection = db['messages']
conversations_collection = db['conversations']

# --------------------------
# Initialisation du serializer pour tokens d'email
# --------------------------
ts = URLSafeTimedSerializer(app.config["SECRET_KEY"])

# --------------------------
# Charger le modèle SpaCy pour le français
# --------------------------
nlp = spacy.load('fr_core_news_sm')

# --------------------------
# Charger le modèle de langage pré-entraîné (transformers)
# --------------------------
nlp_model = None
try:
    nlp_model = pipeline("text2text-generation", model="t5-small")
    app.logger.info("nlp_model chargé")
except Exception as e:
    app.logger.warning(f"Impossible de charger nlp_model (transformers). Le bot utilisera les règles. Détail: {e}")

# --------------------------
# Initialisation du modèle ML / TF-IDF
# --------------------------
bot_model = ChatbotInference(
    tfidf_path='tfidf.pkl',            
    tokenizer_path='tokenizer.pkl',    
    label_encoder_path='label_encoder.pkl',
    model_path='chatbot_model.h5',
    data_csv='data/data_clean.csv',    
    spacy_model='fr_core_news_sm'
)

# --------------------------
# Routes publiques et auth
# --------------------------

@app.route('/')
def home():
    return render_template('index.html', current_user=session.get('username'))

@app.route('/register', methods=['GET'])
def show_register_form():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_user():
    data = request.form
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not username or not email or not password or not confirm_password:
        flash('Tous les champs sont requis', 'error')
        return redirect(url_for('show_register_form'))

    if password != confirm_password:
        flash('Les mots de passe ne correspondent pas', 'error')
        return redirect(url_for('show_register_form'))

    existing_user = users_collection.find_one({'email': email})

    if existing_user:
        flash('Cet email est déjà enregistré', 'error')
        return redirect(url_for('show_register_form'))

    # Générer un token de confirmation
    token = ts.dumps(email, salt='email-confirm-key')

    # Enregistrer l'utilisateur avec le token
    new_user = {
        "username": username,
        "email": email,
        "password": password,
        "confirmed": False,
        "confirmation_token": token,
        "registration_date": datetime.now()
    }
    users_collection.insert_one(new_user)

    # Envoyer l'email de confirmation
    send_confirmation_email(username, email, token)

    flash('Un email de confirmation a été envoyé à votre adresse. Veuillez le vérifier.', 'success')
    return redirect(url_for('registration_success', username=username))

@app.route('/registration_success/<username>', methods=['GET'])
def registration_success(username):
    return render_template('registration_success.html', user={'username': username})

@app.route('/confirm/<token>', methods=['GET'])
def confirm_email(token):
    try:
        email = ts.loads(token, salt='email-confirm-key', max_age=86400)
    except:
        flash('Le lien de confirmation est invalide ou a expiré', 'error')
        return redirect(url_for('show_register_form'))

    user = users_collection.find_one({'email': email})

    if user and not user['confirmed']:
        users_collection.update_one({'_id': user['_id']}, {'$set': {'confirmed': True}})
        flash('Votre compte a été activé avec succès. Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('show_login_form'))
    else:
        flash('Votre compte est déjà activé. Vous pouvez vous connecter.', 'success')
        return redirect(url_for('show_login_form'))

@app.route('/login', methods=['GET'])
def show_login_form():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.form
    username = data.get('username')
    password = data.get('password')

    user = users_collection.find_one({'username': username, 'password': password, 'confirmed': True})

    if user:
        session['user_id'] = str(user['_id'])
        session['username'] = user.get('username')
        flash('Connexion réussie', 'success')
        return redirect(url_for('home'))
    else:
        flash('Nom d\'utilisateur ou mot de passe incorrect', 'error')
        return redirect(url_for('show_login_form'))

# --------------------------
# Fonctions email / confirmation
# --------------------------
def make_confirmation_link(token):
    if PUBLIC_URL:
        return f"{PUBLIC_URL.rstrip('/')}{url_for('confirm_email', token=token)}"
    else:
        return url_for('confirm_email', token=token, _external=True)

def send_confirmation_email(username, email, token):
    link = make_confirmation_link(token)
    msg = Message('Activation de votre compte', sender=app.config['MAIL_DEFAULT_SENDER'], recipients=[email])
    msg.body = f"""Bonjour {username},

Veuillez cliquer sur le lien ci-dessous pour activer votre compte :

{link}

Cordialement,
L'équipe de votre site."""
    try:
        mail.send(msg)
        app.logger.info(f"Email de confirmation envoyé à {email} — lien: {link}")
    except Exception as e:
        app.logger.error(f"Échec envoi email à {email} — {e}", exc_info=True)
        print(" Lien de confirmation (échec envoi email) :")
        print(link)

# --------------------------
# Décorateurs / déconnexion
# --------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter pour accéder à cette page.', 'error')
            return redirect(url_for('show_login_form'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    flash('Vous avez été déconnecté.', 'success')
    return redirect(url_for('home'))

# --------------------------
# Chat & NLP (intégration modèle ML + règles)
# --------------------------

@app.route('/ask', methods=['GET'])
def show_ask_form():
    return render_template('chatbot.html')

def extract_user_data(user_message):
    """Extraire les informations utilisateur (nom, âge, grossesse) à partir du message."""
    doc = nlp(user_message)
    user_data = {}
    
    for ent in doc.ents:
        if ent.label_ == "PER":
            user_data['name'] = ent.text
        elif ent.label_ == "AGE":
            user_data['age'] = ent.text
        elif ent.label_ == "DATE":
            if "semaine" in ent.text.lower():
                try:
                    nums = [t for t in ent if t.like_num]
                    if nums:
                        user_data['weeks_pregnant'] = int(nums[0].text)
                except Exception:
                    user_data['weeks_pregnant'] = 0
    return user_data

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    user_data = extract_user_data(user_message)

    # --------------------------
    # Intégration modèle ML / TF-IDF
    # --------------------------
    ml_response = bot_model.predict(user_message)
    
    if ml_response['score'] >= 0.45 or ml_response['method'] in ['tfidf', 'tfidf_low']:
        response_message = ml_response['answer']
    else:
        # fallback sur règles existantes
        response_message = handle_user_message(user_data, user_message)
    
    # Enregistrer le message utilisateur dans MongoDB
    user_message_doc = {
        "user": user_data.get('name', session.get('username', 'Inconnu')),
        "text": user_message,
        "timestamp": datetime.now()
    }
    messages_collection.insert_one(user_message_doc)

    # Enregistrer la réponse du bot
    bot_message_doc = {
        "user": "Bot",
        "text": response_message,
        "timestamp": datetime.now()
    }
    messages_collection.insert_one(bot_message_doc)

    # Générer un titre pour le chat
    messages = list(messages_collection.find().sort("timestamp", -1).limit(10))
    chat_title = generate_chat_title(messages, user_message)

    # Enregistrer la conversation
    conversations_collection.insert_one({
        "title": chat_title,
        "date": datetime.now(),
        "messages": messages,
        "user_id": session.get('user_id')
    })

    return jsonify({"message": response_message, "method": ml_response['method'], "score": ml_response['score']})

def handle_user_message(user_data, message):
    """Gérer les différentes requêtes de l'utilisateur en fonction du contexte (règles existantes)."""
    pregnancy_keywords = [
        "symptômes", "alimentation", "exercices", 
        "signes de danger", "soins prénatals", "soins postnataux",  
        "visites médicales" , "nutriments", "yoga prénatal",
        "visites prénatales", "tests de dépistage", "préparations pour l'accouchement",
        "soins du nouveau-né", "allaitement", "alimentation du bébé", "reprise après l'accouchement",
        "nutrition des enfants", "aliments solides", "alimentation équilibrée"
    ]
    personalized_suggestions_keywords = [
        "trimestre", "âge de l'enfant", "âge", "nouveau-né", "bébé", "enfant"
    ]
    if any(keyword in message for keyword in pregnancy_keywords):
        return pregnancy_info(user_data, message)  
    if any(keyword in message for keyword in personalized_suggestions_keywords):
        return personalized_suggestions(user_data, message)
    return "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler?"

# --------------------------
# Reste des fonctions rules-based (pregnancy_info, personalized_suggestions)
# --------------------------
# Ici tu peux garder exactement ton code existant pour pregnancy_info() et personalized_suggestions()
# ... (pas de changement nécessaire)

# --------------------------
# Rappels / Rendez-vous / Contact
# --------------------------

# Route pour afficher la page de contact ou de prise de rendez-vous
# L'accès à cette page est réservé aux utilisateurs authentifiés.
# Si un utilisateur non connecté tente d'y accéder, il sera redirigé vers la page de login
# grâce au décorateur login_required déjà défini dans le fichier.
@app.route('/contact', methods=['GET'])
@login_required
def show_contact_or_appointment():
    """
    Affiche le template qui permet soit d'envoyer un message au conseiller
    soit de demander un rendez-vous (contact_or_appointment.html).
    L'utilisateur doit être connecté pour accéder à cette page.
    """
    return render_template('contact_or_appointment.html')

# Route pour afficher la page de Rappel (publique)
@app.route('/add_reminder', methods=['GET'])
def show_add_reminder_form():
    return render_template('add_reminder.html')

def save_appointment(user_data, appointment_date, reminder_date, reminder_time):
    """Enregistrer un rappel de rendez-vous ou vaccination dans MongoDB."""
    
    # Enregistrer dans la collection des rendez-vous (appointments)
    appointments_collection.insert_one({
        "name": user_data.get("name"),  # Nom de l'utilisateur
        "phone_number": user_data.get("phone_number"),  # Numéro de téléphone
        "appointment_date": appointment_date,  # Date du rendez-vous
        "reminder_date": reminder_date,  # Date du rappel (ex: un jour avant)
        "reminder_time": reminder_time,  # Heure de rappel
        "type": "rappel"  # Type de rappel (visite prénatale, postnatale, vaccination)
    })

    print(f"Rappel pour {user_data.get('name', 'Inconnu')} enregistré avec succès pour le {appointment_date}.")

@app.route('/set_reminder', methods=['POST'])
def set_reminder():
    data = request.json or {}
    name = data.get('name') 
    reminder_type = data.get('type')
    reminder_date = data.get('date')
    reminder_time = data.get('time')  
    phone_number = data.get('phone')

    if reminder_type and reminder_date and phone_number:
        # Enregistrer le rappel dans MongoDB (ou toute autre action)
        save_appointment({
            'name': name,  # Remplacer par le nom de l'utilisateur si disponible
            'phone_number': phone_number,
        }, reminder_date, reminder_date, reminder_time,)  # Enregistrer le rappel et la date de rappel

        # Préparer le message SMS
        message = f"Bonjour {name}, votre rappel de {reminder_type} est fixé pour le {reminder_date} à {reminder_time}."
        
        # Envoyer le SMS
        try:
            send_sms(phone_number, message)
        except Exception as e:
            app.logger.warning(f"Envoi SMS rappel échoué: {e}")

        return jsonify({'message': 'Rappel enregistré avec succès!'})
    return jsonify({'message': 'Erreur: Veuillez remplir tous les champs.'}), 400

# Route d'ancienne compatibilité : /appointment redirige vers la nouvelle page sécurisée
@app.route('/appointment', methods=['GET'])
def legacy_appointment():
    return redirect(url_for('show_contact_or_appointment'))

# Alias de compatibilité pour éviter BuildError si des templates appellent show_appointment_form
@app.route('/show_appointment_form', methods=['GET'])
def show_appointment_form():
    """
    Endpoint de compatibilité : redirige vers la page unique de contact/rdv.
    Garde le nom ancien pour éviter les BuildError dans les templates existants.
    """
    return redirect(url_for('show_contact_or_appointment'))

@app.route('/schedule_appointment', methods=['POST'])
@login_required
def schedule_appointment():
    username = session.get('username')
    data = request.json or {}
    phone_number = data.get('phone_number')
    appointment_date = data.get('appointment_date')
    description = data.get('description')
    
    if not phone_number or not appointment_date or not description:
        return jsonify({"error": "Tous les champs sont requis"}), 400

    # Exemple : on crée un dictionnaire user_data minimal
    user_data = {
        "name": username or "Inconnu",
        "phone_number": phone_number
    }

    # On enregistre le rendez-vous
    save_appointment(user_data, appointment_date, appointment_date, "09:00")  # 09:00 = heure de rappel par défaut

    return jsonify({"message": "Rendez-vous enregistré avec succès !"})

def send_sms(to, message):
    account_sid = Config.TWILIO_ACCOUNT_SID  # Variable d'environnement
    auth_token = Config.TWILIO_AUTH_TOKEN    # Variable d'environnement
    twilio_number = Config.TWILIO_PHONE_NUMBER     # Variable d'environnement

    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_number,
        to=to
    )
    print(f"SMS envoyé à {to}: {message}")

# --------------------------
# Contact conseiller (anciennement contact_advisor)
# --------------------------
PHONE_RE = re.compile(r'^\+?\d{8,15}$')  # adapte à tes besoins

# GET legacy : redirige vers la page unifiée (protégée)
@app.route('/contact_advisor', methods=['GET'])
def legacy_contact_advisor_get():
    return redirect(url_for('show_contact_or_appointment'))

# POST : envoi réel (protégé, utilisateur connecté requis)
@app.route('/contact_advisor', methods=['POST'])
@login_required
def contact_advisor():
    data = request.get_json() or {}
    phone_number = (data.get('phone_number') or '').strip()
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip()
    message = (data.get('message') or '').strip()

    # validation simple côté serveur
    if not phone_number or not name or not email or not message:
        return jsonify({"error": "Numéro de téléphone, nom, email et message requis"}), 400

    if not PHONE_RE.match(phone_number):
        return jsonify({"error": "Format du numéro invalide. Utilisez le format international, ex: +2265412XXXX."}), 400

    # Optionnel : limiter la fréquence (rate limiting simple)
    last = session.get('last_contact_ts')
    import time
    now = time.time()
    if last and now - last < 15:  # 15 secondes mini entre envois
        return jsonify({"error": "Veuillez patienter avant d'envoyer un nouveau message."}), 429
    session['last_contact_ts'] = now

    # Envoie du SMS à l'advisor (numéro interne)
    advisor_phone = os.getenv("ADVISOR_PHONE", "+22673556708")  # Remplace par le numéro réel du conseiller

    try:
        send_sms(advisor_phone, f"Message de {name} ({phone_number}, {email}): {message}")
        app.logger.info(f"Contact advisor: from {name} ({phone_number}) — forwarded to {advisor_phone}")
        return jsonify({"message": "Message envoyé au conseiller"}), 200
    except Exception as e:
        app.logger.error("Erreur lors de l'envoi du message au conseiller", exc_info=True)
        if app.debug:
            return jsonify({"error": f"Erreur serveur: {str(e)}"}), 500
        return jsonify({"error": "Impossible d'envoyer le message pour le moment. Veuillez réessayer plus tard."}), 500

# --------------------------
# Historique (limité à l'utilisateur)
# --------------------------
@app.route('/get_history', methods=['GET'])
def get_history():
    """
    Retourne uniquement l'historique des conversations appartenant à l'utilisateur connecté.
    - Si utilisateur non connecté : retourne une liste vide (ou on peut renvoyer 401 selon le besoin).
    """
    user_id = session.get('user_id')
    if not user_id:
        # L'utilisateur n'est pas connecté : on ne retourne aucun historique pour préserver la confidentialité
        return jsonify({"history": []})

    # Tri par ordre chronologique (le plus récent en premier)
    history = conversations_collection.find({"user_id": user_id}).sort("date", -1)
    history_list = []
    for chat in history:
        # Convertir les ObjectId en chaînes de caractères
        chat_dict = {
            "id": str(chat["_id"]),
            "title": chat.get("title", "Sans titre"),
            "date": chat.get("date", datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            "messages": [
                {
                    "user": message.get("user", "Inconnu"),
                    "text": message.get("text", ""),
                    "timestamp": message.get("timestamp", datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                }
                for message in chat.get("messages", [])
            ]
        }
        history_list.append(chat_dict)
    return jsonify({"history": history_list})

# Génération du titre d'une conversation
def generate_chat_title(messages, user_message):
    """Génère un titre pour le chat en fonction du premier message."""
    if messages:
        first_message = messages[0].get('text', '') if isinstance(messages[0], dict) else str(messages[0])
    else:
        first_message = user_message

    # Utilisez une expression régulière pour extraire les mots clés
    keywords = re.findall(r'\b\w+\b', first_message)
    title = ' '.join(keywords[:5])  # Utilisez les 5 premiers mots comme titre
    return title

# Récupérer une conversation en vérifiant la propriété
@app.route('/get_chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """
    Récupère une conversation par id, mais vérifie que l'utilisateur connecté est propriétaire.
    """
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Non autorisé"}), 401

    chat = conversations_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    if chat:
        chat_dict = {
            "id": str(chat["_id"]),
            "title": chat.get("title", "Sans titre"),
            "date": chat.get("date", datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            "messages": [
                {
                    "user": message.get("user", "Inconnu"),
                    "text": message.get("text", ""),
                    "timestamp": message.get("timestamp", datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                }
                for message in chat.get("messages", [])
            ]
        }
        return jsonify(chat_dict)
    else:
        return jsonify({"error": "Chat not found or inaccessible"}), 404

# --------------------------
# Lancement de l'application
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
