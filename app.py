import os
import datetime
import hashlib
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
from database import list_users, verify, delete_user_from_db, add_user
from database import read_note_from_db, write_note_into_db, delete_note_from_db, match_user_id_with_note_id
from database import image_upload_record, list_images_for_user, match_user_id_with_image_uid, delete_image_from_db
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import matplotlib.pyplot as plt
from flask import send_from_directory
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
from flask_migrate import Migrate
app = Flask(__name__)
app.config.from_object('config')


from flask import Flask, render_template, send_from_directory
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from model import CNN  # Adjust this import statement as needed
from PIL import Image
import imageio
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.secret_key = 'your_secret_key'

db = SQLAlchemy(app)
# Initialize the Flask-Migrate extension
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


UPLOAD_FOLDER = 'static/videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and set it to evaluation mode
model_path = 'model/model.pth'  # Update to the path where your model is saved
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust as needed for your model
])
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_blocked = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        """Set the password for the user by generating a hash"""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Check if the provided password matches the user's hashed password"""
        return check_password_hash(self.password, password)

    def __repr__(self):
        """Return a string representation of the User object"""
        return f"User('{self.username}', '{self.is_admin}')"

@login_manager.user_loader
def load_user(user_id):
    """Load the user object from the database based on the user_id"""
    return User.query.get(int(user_id))


# Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle the login functionality"""
    if current_user.is_authenticated:
        return redirect('/admin' if current_user.is_admin else '/user')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect('/admin' if user.is_admin else '/user')
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle the registration functionality"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return redirect('/login')

    return render_template('register.html')


@app.route('/reset_password', methods=['GET', 'POST'])
def reset():
    """Render the reset password page"""
    return render_template('reset_password.html')

@app.route('/admin')
@login_required
def admin_dashboard():
    """Render the admin dashboard"""
    if current_user.is_admin:
        users = User.query.all()
        return render_template('admin_dashboard.html', users=users)

    return redirect('/user')

@app.route('/admin/users')
@login_required
def admin_users():
    """Render the admin users page"""
    if current_user.is_admin:
        users = User.query.all()
        return render_template('public_page.html', users=users)
    else:
        return redirect('/admin')

@app.route('/user')
@login_required
def user_dashboard():
    """Render the user dashboard"""
    return render_template('user_dashboard.html')

# @app.route('/register')
# @login_required
# def register_dashboard():
#     """Render the user dashboard"""
#     return render_template('user_dashboard.html')

@app.route('/logout')
@login_required
def logout():
    """Handle the logout functionality"""
    logout_user()
    return redirect('/login')

@app.route('/admin/block_user/<int:user_id>', methods=['POST'])
@login_required
def block_user(user_id):
    """Block a user by setting is_blocked to True and modifying the password"""
    if current_user.is_admin:
        user = User.query.get(user_id)
        if user:
            user.is_blocked = True
            user.password = f'{user.password}_blocked'
            db.session.commit()
    return redirect('/admin/users')

@app.route('/admin/unblock_user/<int:user_id>', methods=['POST'])
@login_required
def unblock_user(user_id):
    """Unblock a user by setting is_blocked to False and removing the "_blocked" suffix from the password"""
    if current_user.is_admin:
        user = User.query.get(user_id)
        if user:
            user.is_blocked = False
            user.password = user.password.replace('_blocked', '')
            db.session.commit()
    return redirect('/admin/users')

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete a user by removing them from the database"""
    if current_user.is_admin:
        user = User.query.get(user_id)
        if user:
            db.session.delete(user)
            db.session.commit()
    return redirect('/admin/users')

def create_tables():
    """Create database tables"""
    with app.app_context():
        db.create_all()









@app.route('/test/')
def upload_file():
    return render_template('test_page.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            #filename = secure_filename(file.filename)
            filename='abc.mp4'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                success_message = "Operation successful!"
                # Process the file for emotion detection
                display_video_with_percentage(file_path)
                
                
               # video_path = 'output/Test_result.mp4'  # Adjust path as needed
               #gif_path = 'output/Test_result.gif'  # Adjust path as needed
                #return render_template('test_page.html', video_path=video_path, gif_path=gif_path)
                             
                return render_template('output.html', message=success_message)
                #return 'File uploaded and processed successfully. Check the output files.'
            except Exception as e:
                return str(e)

    return 'Unexpected error'



    
@app.route('/videos/<filename>')
def videos(filename):
    return send_from_directory(os.getcwd(), filename)



@app.route("/output/")
def output():
    return render_template("output.html")
    
    
@app.route("/test/")
def test():
    return render_template("test_page.html")


@app.route("/private/")
def FUN_private():
    if "current_user" in session.keys():
        notes_list = read_note_from_db(session['current_user'])
        notes_table = zip([x[0] for x in notes_list],\
                          [x[1] for x in notes_list],\
                          [x[2] for x in notes_list],\
                          ["/delete_note/" + x[0] for x in notes_list])

        images_list = list_images_for_user(session['current_user'])
        images_table = zip([x[0] for x in images_list],\
                          [x[1] for x in images_list],\
                          [x[2] for x in images_list],\
                          ["/delete_image/" + x[0] for x in images_list])

        return render_template("private_page.html", notes = notes_table, images = images_table)
    else:
        return abort(401)

@app.route("/admin/")
def FUN_admin():
    if session.get("current_user", None) == "ADMIN":
        user_list = list_users()
        user_table = zip(range(1, len(user_list)+1),\
                        user_list,\
                        [x + y for x,y in zip(["/delete_user/"] * len(user_list), user_list)])
        return render_template("admin.html", users = user_table)
    else:
        return abort(401)






@app.route("/write_note", methods = ["POST"])
def FUN_write_note():
    text_to_write = request.form.get("text_note_to_take")
    write_note_into_db(session['current_user'], text_to_write)

    return(redirect(url_for("FUN_private")))

@app.route("/delete_note/<note_id>", methods = ["GET"])
def FUN_delete_note(note_id):
    if session.get("current_user", None) == match_user_id_with_note_id(note_id): # Ensure the current user is NOT operating on other users' note.
        delete_note_from_db(note_id)
    else:
        return abort(401)
    return(redirect(url_for("FUN_private")))


# Reference: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods = ['POST'])
def FUN_upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', category='danger')
            return(redirect(url_for("FUN_private")))
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            flash('No selected file', category='danger')
            return(redirect(url_for("FUN_private")))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_time = str(datetime.datetime.now())
            image_uid = hashlib.sha1((upload_time + filename).encode()).hexdigest()
            # Save the image into UPLOAD_FOLDER
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_uid + "-" + filename))
            # Record this uploading in database
            image_upload_record(image_uid, session['current_user'], filename, upload_time)
            return(redirect(url_for("FUN_private")))

    return(redirect(url_for("FUN_private")))

@app.route("/delete_image/<image_uid>", methods = ["GET"])
def FUN_delete_image(image_uid):
    if session.get("current_user", None) == match_user_id_with_image_uid(image_uid): # Ensure the current user is NOT operating on other users' note.
        # delete the corresponding record in database
        delete_image_from_db(image_uid)
        # delete the corresponding image file from image pool
        image_to_delete_from_pool = [y for y in [x for x in os.listdir(app.config['UPLOAD_FOLDER'])] if y.split("-", 1)[0] == image_uid][0]
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_to_delete_from_pool))
    else:
        return abort(401)
    return(redirect(url_for("FUN_private")))



@app.route("/logout/")
def FUN_logout():
    session.pop("current_user", None)
    return(redirect(url_for("FUN_root")))

@app.route("/delete_user/<id>/", methods = ['GET'])
def FUN_delete_user(id):
    if session.get("current_user", None) == "ADMIN":
        if id == "ADMIN": # ADMIN account can't be deleted.
            return abort(403)

        # [1] Delete this user's images in image pool
        images_to_remove = [x[0] for x in list_images_for_user(id)]
        for f in images_to_remove:
            image_to_delete_from_pool = [y for y in [x for x in os.listdir(app.config['UPLOAD_FOLDER'])] if y.split("-", 1)[0] == f][0]
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_to_delete_from_pool))
        # [2] Delele the records in database files
        delete_user_from_db(id)
        return(redirect(url_for("FUN_admin")))
    else:
        return abort(401)

@app.route("/add_user", methods = ["POST"])
def FUN_add_user():
    if session.get("current_user", None) == "ADMIN": # only Admin should be able to add user.
        # before we add the user, we need to ensure this is doesn't exsit in database. We also need to ensure the id is valid.
        if request.form.get('id').upper() in list_users():
            user_list = list_users()
            user_table = zip(range(1, len(user_list)+1),\
                            user_list,\
                            [x + y for x,y in zip(["/delete_user/"] * len(user_list), user_list)])
            return(render_template("admin.html", id_to_add_is_duplicated = True, users = user_table))
        if " " in request.form.get('id') or "'" in request.form.get('id'):
            user_list = list_users()
            user_table = zip(range(1, len(user_list)+1),\
                            user_list,\
                            [x + y for x,y in zip(["/delete_user/"] * len(user_list), user_list)])
            return(render_template("admin.html", id_to_add_is_invalid = True, users = user_table))
        else:
            add_user(request.form.get('id'), request.form.get('pw'))
            return(redirect(url_for("FUN_admin")))
    else:
        return abort(401)

        
def detect_emotion(frame):
    emotion = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    video = imageio.mimread(frame, memtest=False)
    mov = []
    emotion_counts = {key: 0 for key in emotion.values()}
    suspicious_emotions = ["Fearful", "Angry"]  # Merge Fear and Angry for suspicion detection
    suspicious_frame_count = 0
    total_frame_count = 0

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for frame_index, frame in enumerate(video):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            resized_img = cv2.resize(roi_gray_frame, (48, 48)).astype(np.uint8)  # Ensure correct type for PIL conversion

            # Apply the transformation, keeping the image as 1-channel grayscale
            transformed_img = transform(resized_img).unsqueeze(0).to(device)  # Add batch dimension and move to device

            with torch.no_grad():
                emotion_prediction = model(transformed_img)
                maxindex = int(torch.argmax(emotion_prediction, 1))
                emotion_label = emotion[maxindex]
                emotion_counts[emotion_label] += 1

                # Check if the detected emotion is suspicious
                if emotion_label in suspicious_emotions:
                    suspicious_frame_count += 1

                # Update the total frame count
                total_frame_count += 1

                # Add psychologist signals based on detected emotions
                psychologist_signal = ""
                if emotion_label == "Fearful":
                    psychologist_signal = "Indicates high levels of anxiety or stress."
                elif emotion_label == "Angry":
                    psychologist_signal = "Suggests frustration or hostility."

            cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, psychologist_signal, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        img = ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        ax[1].clear()

        # Data for plotting
        labels = list(emotion_counts.keys())
        sizes = list(emotion_counts.values())

        # Check if there are any emotions detected before plotting
        if sum(sizes) > 0:
            ax[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax[1].set_title('Emotion Frequency Analysis\nSuspiciousness: {:.2f}%'.format((suspicious_frame_count / total_frame_count) * 100))

            # Add psychologist signals under the pie chart
            ax[1].text(0.5, -0.1, psychologist_signal, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes, fontsize=10, color='red')

        else:
            ax[1].clear()  # Clear the plot if no data to show
            ax[1].text(0.5, 0.5, 'No emotions detected', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
            plt.suptitle('Comprehensive Emotion Analysis Outcomes')  # Title for the entire figure

        mov.append([img])

    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anime.save('Test_result.mp4', writer=writer)
    anime.save('Test_result.gif', writer='pillow', fps=20)
    plt.close()
    
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


   

       
        


def display_video_with_percentage(video_path):
    emotion = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    video = imageio.mimread(video_path, memtest=False)
    mov = []
    emotion_counts = {key: 0 for key in emotion.values()}
    suspicious_emotions = ["Fearful", "Angry",]
    suspicious_frame_count = 0
    total_frame_count = 0

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_rate = 24  # Assuming a frame rate of 24 FPS
    emotions_with_signals = []  # List to store emotion signals with timestamps

    for frame_index, frame in enumerate(video):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            resized_img = cv2.resize(roi_gray_frame, (48, 48)).astype(np.uint8)

            transformed_img = transform(resized_img).unsqueeze(0).to(device)

            with torch.no_grad():
                emotion_prediction = model(transformed_img)
                maxindex = int(torch.argmax(emotion_prediction, 1))
                emotion_label = emotion[maxindex]
                emotion_counts[emotion_label] += 1

                if emotion_label in suspicious_emotions:
                    suspicious_frame_count += 1

                total_frame_count += 1

                timestamp = frame_index / frame_rate

                psychologist_signal = ""
                if emotion_label == "Fearful":
                    psychologist_signal = "May indicate high levels of anxiety or stress, potential for unpredictable behavior."
                elif emotion_label == "Angry":
                    psychologist_signal = "Suggests frustration or hostility, possible aggressive tendencies."
                elif emotion_label == "Disgusted":
                    psychologist_signal = "May reflect moral or physical repulsion, potential for confrontational behavior."
                elif emotion_label == "Sad":
                    psychologist_signal = "Indicates sadness or despair, which could be related to hopelessness or depression."
                elif emotion_label == "Surprised":
                    psychologist_signal = "Reflects unexpectedness or shock, which might be due to unforeseen events."
                elif emotion_label == "Neutral":
                    psychologist_signal = "Neutral expression might indicate suppression of emotions or disengagement."
                elif emotion_label == "Happy":
                    psychologist_signal = "Happiness could either be genuine or a mask to hide other emotions."

                emotions_with_signals.append((timestamp, psychologist_signal))

            cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, psychologist_signal, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        img = ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        ax[1].clear()

        labels = list(emotion_counts.keys())
        sizes = list(emotion_counts.values())

        if sum(sizes) > 0:
            ax[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax[1].set_title('Emotion Frequency Analysis\nSuspiciousness: {:.2f}%'.format((suspicious_frame_count / total_frame_count) * 100))

            ax[1].text(0.5, -0.1, psychologist_signal, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes, fontsize=10, color='red')

        else:
            ax[1].clear()
            ax[1].text(0.5, 0.5, 'No emotions detected', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
            plt.suptitle('Comprehensive Emotion Analysis Outcomes')

        mov.append([img])

    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anime.save('Test_result.mp4', writer=writer)
    anime.save('Test_result.gif', writer='pillow', fps=20)
    plt.close()

    # Save the emotions with signals report
    with open('static/emotion_report.txt', 'w') as report_file:
        for timestamp, psychologist_signal in emotions_with_signals:
            report_file.write(f"Time: {timestamp:.2f}s - Signal: {psychologist_signal}\n")


    
  
  
"""
# Assuming a path to your video
video_path = video_path # Update this path to your actual video file location
display_video_with_percentage(video_path)

print("Animation saved")
"""





if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
