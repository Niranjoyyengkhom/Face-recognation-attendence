from tkinter import *
import os
import cv2
import sys
from PIL import Image, ImageTk
import numpy
import sqlite3
import time as Time
import datetime
from xlsxwriter.workbook import Workbook
#fileName = os.environ['ALLUSERSPROFILE'] + "\WebcamCap.txt"
cancel = False
root = Tk()
root.geometry("655x333")
global l4,MessageLbl,count,countIsLogin,tempText,closeCamp
global roll_no, name, course,cap
closeCamp=0
countIsLogin=0
tempText=""
f1=Frame(root,bg="grey", borderwidth=6)
f2=Frame(root,bg="grey", borderwidth=6)
f1.pack(side=LEFT,fill="y")
f2.pack(side=LEFT,fill="y")

l1=Label(f1,text="Name :",bg="grey")
l2=Label(f1,text="Roll No :",bg="grey")
l3=Label(f1,text="Course :",bg="grey")
l4=Label(f2,compound=CENTER, anchor=CENTER, relief=RAISED,bg="grey")
global lNoCount
l4.pack()
l1.grid(row=1)
l2.grid(row=2)
l3.grid(row=3)

name_value=StringVar(),
roll_value=StringVar()
course_value=StringVar()

name_entry=Entry(f1)
roll_entry=Entry(f1)
course_entry=Entry(f1)
name_entry.grid(row=1,column=1)
roll_entry.grid(row=2,column=1)
course_entry.grid(row=3,column=1)

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def insertOrUpdate():
    roll_no = roll_entry.get()
    name = name_entry.get()
    course = course_entry.get()
    conn = sqlite3.connect("facerecognition.db")
    c = conn.cursor()
    cmd = "SELECT Name,RollNo,Course FROM StudentInfo WHERE RollNo=" + str(roll_no)
    c.execute(cmd)
    result = c.fetchall()
    now = datetime.datetime.now()
    if result:
        for row in result:
            cmd = "UPDATE StudentInfo SET Name='" + str(name) + "',Course=='" + str(course) + "' WHERE RollNo=" + str(roll_no)
            conn.execute(cmd)
            conn.commit()
            conn.close()
            MessageLbl = Label(f1, text="Updated Sucessfully :", fg="green")
            MessageLbl.grid(row=0)

    else:
        cmd = "INSERT INTO StudentInfo(Name,Rollno,Course,Date,IsActive) VALUES('" + str(name) + "','" + str(roll_no) + "','" + str(course) + "','"+now.strftime('%Y-%m-%d %H:%M:%S')+ "',1)"
        conn.execute(cmd)
        conn.commit()
        '''cmd="SELECT MAX(StudentID) AS StudentID FROM StudentInfo WHERE IsActive=1"
        c.execute(cmd)
        StID=c.fetchall()
        for id in StID:
            cmd= "INSERT INTO TotalPresentDays(StudentID,TotalPresentDays)  VALUES('"+str(id[0])+"',0)"
            conn.execute(cmd)
            conn.commit()'''
        conn.close()
        MessageLbl = Label(f1, text="Saved Sucessfully :", fg="green")
        MessageLbl.grid(row=0)



#root.count=0
font = cv2.FONT_HERSHEY_SIMPLEX

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count=0;

    #l4.configure(bg='grey',image=None)

def captureImage():
    if (closeCamp == 0):
        name = name_entry.get()
        roll_no = roll_entry.get()
        course = course_entry.get()
        ret, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecting different faces
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        if faces is not None:
            for (x, y, w, h) in faces:
                global count
                count += 1
                cv2.rectangle(cv2image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if (count <= 20):
                    cv2.imwrite("training_data/"+ str(name) + "."+ str(roll_no) + "." + str(count) + ".jpg", gray[y:y + h, x:x + w])
                    cv2.putText(cv2image, str(count), (x, y - 40), font, 0.6, (255, 255, 255), 2)
        if (count <= 20):
            prevImg = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=prevImg)
            l4.imgtk = imgtk
            l4.configure(image=imgtk,bg='green')
            l4.after(5, captureImage)

        if(count==21):
            count=0
            cap.release()
            l4.configure(image='',bg='grey')
            path='training_data'
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

            # empty face sample initialised
            faceSamples = []

            # IDS for each individual
            ids = []

            # Looping through all the file path
            for imagePath in imagePaths:

                # converting image to grayscale
                PIL_img = Image.open(imagePath).convert('L')

                # converting PIL image to numpy array using array() method of numpy
                img_numpy = numpy.array(PIL_img, 'uint8')

                # Getting the image id
                id = int(os.path.split(imagePath)[-1].split(".")[1])

                # Getting the face from the training images
                faces = face_detector.detectMultiScale(img_numpy)

                # Looping for each face and appending it to their respective IDs
                for (x, y, w, h) in faces:
                    # Add the image to face samples
                    faceSamples.append(img_numpy[y:y + h, x:x + w])

                    # Add the ID to IDs
                    ids.append(id)

            # Passing the face array and IDs array
            # Training the model using the faces and IDs
            recognizer.train(faceSamples, numpy.array(ids))
            # Saving the model into s_model.yml
            assure_path_exists('saved_model/')
            recognizer.write('saved_model/s_model.yml')

def OnCam():
    global closeCamp
    global cap
    cap = cv2.VideoCapture(0)
    closeCamp = 0
    captureImage()

def getProfile(Id):
    conn = sqlite3.connect("facerecognition.db")
    c = conn.cursor()
    cmd = "SELECT s.Name,s.RollNo,s.StudentID,at.LogStatus,at.LogInTime,at.LogOutTime FROM StudentInfo s left join AttendanceInfo at on at.StudentID=s.StudentID WHERE s.RollNo='" + str(Id)+"' ORDER BY at.AttendanceID DESC LIMIT 1"
    c.execute(cmd)
    result = c.fetchall()
    Record = None
    for row in result:
        Record = row
    conn.close()
    return Record

def identify_Image():
    global closeCamp
    global countIsLogin
    global cap
    cap = cv2.VideoCapture(0)
    closeCamp = 0
    countIsLogin = 0
    assure_path_exists("saved_model/")
    recognizer.read('saved_model/s_model.yml')
    identify()

def OffCam():
    global closeCamp
    global cap
    cap = cv2.VideoCapture(0)
    cap.release()
    closeCamp = 1
def identify():
    if(closeCamp==0):

        global tempText
        global countIsLogin

        ret, im = cap.read()
        cv2image = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Getting all faces from the video frame
        faces = face_detector.detectMultiScale(gray, 1.3, 5)  # default

        if faces is not None:
        # For each face in faces, we will start predicting using pre trained model
            for (x, y, w, h) in faces:

                # Create rectangle around the face
                cv2.rectangle(cv2image, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

                # Recognize the face belongs to which ID
                Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # Our trained model is working here

                # Set the name according to id
                if Id is not None:
                        profile = getProfile(Id)
                        if (profile != None):


                            cv2.rectangle(cv2image, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), 4)

                            if (100 * (1 - (confidence) / 300) >= 75):
                                    countIsLogin += 1
                                    cv2.putText(cv2image, str(profile[0]), (x, y - 40), font, 0.6,(255, 255, 255), 2)
                                    conn = sqlite3.connect("facerecognition.db")

                                    now = datetime.datetime.now()
                                    if(profile[3]==None):
                                        cv2.putText(cv2image, str(profile[0]) + " , Log Out!", (x, y - 40), font, 0.6,(0, 255, 0), 2)
                                        if(countIsLogin==1):
                                            cv2.putText(cv2image, str(profile[0]) + " , Log In!", (x, y - 40), font, 0.6, (0, 255, 0), 2)
                                            cmd5 = "INSERT INTO AttendanceInfo(StudentID,LogStatus,LogInTime,LogOutTime,IsActive) VALUES('"+str(profile[2])+"','IN','"+now.strftime('%Y-%m-%d %H:%M:%S')+"',NULL,1)"
                                            conn.execute(cmd5)
                                            conn.commit()
                                            conn.close()
                                            countIsLogin += 1
                                    if (profile[3] == 'OUT'):
                                        cv2.putText(cv2image, str(profile[0]) + " , Log Out!", (x, y - 40), font, 0.6,
                                                    (0, 255, 0), 2)
                                        if (countIsLogin == 1):
                                            cv2.putText(cv2image, str(profile[0]) + " , Log In!", (x, y - 40), font,
                                                        0.6, (0, 255, 0), 2)
                                            cmd = "INSERT INTO AttendanceInfo(StudentID,LogStatus,LogInTime,LogOutTime,IsActive) VALUES('" + str(
                                                profile[2]) + "','IN','" + now.strftime(
                                                '%Y-%m-%d %H:%M:%S') + "',NULL,1)"
                                            conn.execute(cmd)
                                            conn.commit()
                                            conn.close()
                                            countIsLogin += 1
                                    elif(profile[3] == 'IN'):
                                        cv2.putText(cv2image, str(profile[0]) + " , Log In!", (x, y - 40), font, 0.6, (0, 255, 0), 2)
                                        if (countIsLogin == 1):
                                            cv2.putText(cv2image, str(profile[0]) + " , Log Out!", (x, y - 40), font,0.6, (0, 255, 0), 2)
                                            cmd4 = "INSERT INTO AttendanceInfo(StudentID,LogStatus,LogInTime,LogOutTime,IsActive) VALUES('" + str(profile[2]) + "','OUT',NULL,'" + now.strftime('%Y-%m-%d %H:%M:%S') + "',1)"
                                            conn.execute(cmd4)
                                            conn.commit()
                                            cmd2 = ("SELECT COUNT(S.StudentID) FROM StudentInfo S LEFT JOIN AttendanceInfo A ON A.StudentID=S.StudentID "
                                                    " WHERE  A.LogInTime=(SELECT MIN(LogInTime) FROM AttendanceInfo  WHERE  StudentID=S.StudentID AND DATE(LogInTime)=DATE(A.LogInTime))"
                                                    "AND S.StudentID='" + str(profile[2]) + "'")
                                            result = conn.execute(cmd2)
                                            countTotalPresentDays = result.fetchall()
                                            for days in countTotalPresentDays:
                                                cmd9 = "SELECT MAX(AttendanceID) FROM AttendanceInfo WHERE StudentID='" + str(profile[2]) + "' AND LogStatus='OUT'"
                                                result2 = conn.execute(cmd9)
                                                AttendanceID = result2.fetchall()
                                                for id in AttendanceID:
                                                    cmd3 = "UPDATE AttendanceInfo SET TotalDaysPresent='" + str(days[0]) + "' WHERE AttendanceID='" + str(id[0]) + "'"
                                                    conn.execute(cmd3)
                                                    conn.commit()
                                            conn.close()

                                            countIsLogin += 1

                            else:
                                cv2.putText(cv2image, "Unknown", (x, y - 40), font, 0.6, (255, 255, 255), 2)
                        else:
                            cv2.putText(cv2image, "Unknown", (x, y - 40), font, 0.6, (0, 0, 255), 2)
                else:
                   cv2.putText(cv2image, "Unknown", (x, y - 40), font, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(cv2image, "Unknown", font, 0.6, (0, 0, 255), 2)

        prevImg = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=prevImg)
        l4.imgtk = imgtk
        l4.configure(image=imgtk,bg='green')
        l4.after(5, identify)
    else:
        l4.imgtk = None
        l4.configure(image=None)
        cap.release()
def exportData():
    now = datetime.datetime.now()
    workbook = Workbook('AttendanceReport.xlsx')
    worksheet = workbook.add_worksheet()
    conn = sqlite3.connect("facerecognition.db")
    mysel = conn.execute("SELECT S.Name,S.RollNo,S.Course,A.LogStatus,A.LogInTime,A2.LogStatus,"
                         "A2.LogOutTime,strftime('%d', date(A.LogInTime)) AS DATE,"
                         "strftime('%m', date(A.LogInTime)) AS MONTH,strftime('%Y', date(A.LogInTime))"
                         " AS YEAR ,A2.TotalDaysPresent FROM StudentInfo S LEFT JOIN AttendanceInfo A ON A.StudentID=S.StudentID"
                         " LEFT JOIN AttendanceInfo A2 ON A2.StudentID=A.StudentID LEFT JOIN TotalPresentDays PD ON PD.StudentID=S.StudentID "
                         "WHERE DATE(A.LogInTime)=DATE(A2.LogOutTime)"
                         " AND strftime('%m', date(A.LogInTime))=strftime('%m', date(A2.LogOutTime)) "
                         "AND strftime('%Y', date(A.LogInTime))=strftime('%Y', date(A2.LogOutTime)) "
                         "AND A.LogInTime=(SELECT MIN(LogInTime) FROM AttendanceInfo  "
                         "WHERE  StudentID=S.StudentID AND "
                         " DATE(LogInTime)=DATE(A.LogInTime)"
                         " AND strftime('%m', date(LogInTime))=strftime('%m', date(A.LogInTime)) "
                         "AND strftime('%Y', date(LogInTime))=strftime('%Y', date(A.LogInTime)))"
                         "AND A2.LogOutTime=(SELECT MAX(LogOutTime) "
                         "FROM AttendanceInfo  WHERE  StudentID=S.StudentID AND"
                         " DATE(LogOutTime)=DATE(A2.LogOutTime)"
                         " AND strftime('%m', date(LogOutTime))=strftime('%m', date(A2.LogOutTime)) "
                         "AND strftime('%Y', date(LogOutTime))=strftime('%Y', date(A2.LogOutTime)) "
                         " ) ORDER BY A2.AttendanceID")
    worksheet.write(0, 0, 'Name')
    worksheet.write(0, 1, 'RollNo')
    worksheet.write(0, 2, 'Course')
    worksheet.write(0, 3, 'LogInStatus')
    worksheet.write(0, 4, 'LogInTime')
    worksheet.write(0, 5, 'LogOutStatus')
    worksheet.write(0, 6, 'LogOutTime')
    worksheet.write(0, 7, 'Date')
    worksheet.write(0, 8, 'Month')
    worksheet.write(0, 9, 'Year')
    worksheet.write(0, 10, 'Total Days Present')
    for i, row in enumerate(mysel):
        for j, value in enumerate(row):
            worksheet.write(i+1, j, value)
    workbook.close()


def reload():
    name_entry.delete(0, END)
    roll_entry.delete(0, END)
    course_entry.delete(0, END)
    l4.configure(image='', bg='grey')
    OffCam();
    #root.destroy()
Button(f1,text="Submit Details",command=insertOrUpdate).grid()
Button(f1,text="Capture Image",command=OnCam).grid()
Button(f1,text="Identify Image",command=identify_Image).grid()
Button(f1,text="Reload",command=reload ).grid()
Button(f1,text="Export",command=exportData).grid()
#show_frame()
root.mainloop()
