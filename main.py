import cv2
import os
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask import Flask,request,render_template

app = Flask(__name__)


date_format_2 = date.today().strftime("%d-%B-%Y")

date_format = date.today().strftime("%m_%d_%y")

detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')



if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

if f'Attendance--{date_format}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{date_format}.csv','w') as f:
        f.write('Name,Roll,Time')

def over_all_registrations():
    return len(os.listdir('static/faces'))

def extraction_of_faces(obj):
    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    face_points = detector.detectMultiScale(gray, 1.3, 5)
    return face_points
def detect_face(array):
    load_model = joblib.load('static/recognition_model.pkl')
    return load_model.predict(array)

def train_recognition_model():
    faces = []
    labels = []
    students_list = os.listdir('static/faces')
    for student in students_list:
        for objName in os.listdir(f'static/faces/{student}'):
            image = cv2.imread(f'static/faces/{student}/{objName}')
            resized_face = cv2.resize(image, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(student)
    faces = np.array(faces)
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(faces,labels)
    joblib.dump(KNN,'static/recognition_model.pkl')

def get_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{date_format}.csv')
    Names = df['Name']
    Rolls = df['Roll']
    Times = df['Time']
    l = len(df)
    return Names, Rolls, Times, l

def add_new_student(name):
    user_name = name.split('_')[0]
    user_id = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{date_format}.csv')
    if int(user_id) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{date_format}.csv', 'a') as f:
            f.write(f'\n{user_name},{user_id},{current_time}')




@app.route('/')
def homePage():
    Names,Rolls,Times,l = get_attendance()
    return render_template('homePage.html',names= Names,rolls = Rolls,times = Times,l =l,totalStudents = over_all_registrations(),date = date_format_2)


@app.route('/api/take/attendance',methods = ['GET'])
def take_attendance():
    if 'recognition_model.pkl' not in os.listdir(('static')):
        return render_template('homePage.html',totalStudents = over_all_registrations(),date = date_format_2,message = 'You are here for the first time!! Please do register.')
    capture = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = capture.read()
        if extraction_of_faces(frame)!=():
            (x, y, w, h) = extraction_of_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = detect_face(face.reshape(1, -1))[0]
            add_new_student(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
    Names, Rolls, Times, l = get_attendance()
    return render_template('homePage.html', names=Names, rolls=Rolls, times=Times, l=l, totalStudents = over_all_registrations(),
                           date=date_format_2)

@app.route('/register/new/user',methods= ['GET','POST'])
def new_student():
    new_student_name = request.form['new_student_name']
    new_student_id = request.form['new_student_id']
    student_image_folder = 'static/faces/' + new_student_name + '_' + str(new_student_id)
    if not os.path.isdir(student_image_folder):
        os.makedirs(student_image_folder)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
    i,j = 0,0
    while 1:
        ret,frame = capture.read()
        faces = extraction_of_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,20),2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10 ==0:
                Name = new_student_name+'_'+str(i)+'.jpg'
                cv2.imwrite(student_image_folder+'/'+Name,frame[y:y+h,x:x+w])
                i+=1
            j +=1
        if j == 500:
            break
        cv2.imshow('Adding you!!!Please wait and click "esc" button after completion',frame)
        if cv2.waitKey(1) == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
    print("loading Model for new student")
    train_recognition_model()
    Names,Rolls,Times,l = get_attendance()
    return render_template('homePage.html', names=Names, rolls=Rolls, times=Times, l=l, totalStudents = over_all_registrations(),
                           date=date_format_2)

if __name__ == '__main__':
    app.run(debug=True)




