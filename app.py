from bson import ObjectId
from flask import Flask, render_template, Response, request, redirect, url_for, session
# from flask import *
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import json
import pytesseract
import re
from PIL import Image
import ftfy
import io
import os.path
import difflib
import csv
import dateutil.parser as dparser
import winsound
import threading

import pymongo
from numpy.compat import unicode
from pymongo import MongoClient, ReadPreference

# define cluster

cluster = MongoClient("mongodb://localhost:27017/")
db = cluster["visitor"]
collection = db["users"]
visitorlogtable = db["VisitorLog"]
activevisitorstable = db["ActiveVisitors"]
reqvistable = db["RequestedVisitor"]
acceptedvistable = db["AcceptedVisitors"]
rejectedvistable = db["RejectedVisitors"]
securitylog = db["SecurityUsers"]
adminlog = db["AdminUsers"]

global dobee, countreq
dobee = 0
countreq = 0
for x in reqvistable.find():
    countreq = countreq + 1


def printit():
    global dobee, countreq
    threading.Timer(5.0, printit).start()
    ysum = 0
    for xy in reqvistable.find():
        ysum = ysum + 1
    if ysum != countreq:
        winsound.Beep(440, 500)
        countreq = ysum


printit()

global capture, rec_frame, grey, switch, neg, face, rec, out, data, approvedby, dataobject1
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
data = {}
approvedby = ""
dataobject1 = {}

# make shots directory to save pics
try:
    os.mkdir('./shots')
    os.mkdir('./videos')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')
app.secret_key = "auth"

camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame, data
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (neg):
                frame = cv2.bitwise_not(frame)
            if (capture):
                capture = 0
                data = {}
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
                data = extract_card_details(p)
                print("data->", data)

            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


def pan_read_data(text):
    name = None
    fname = None
    dob = None
    pan = None
    nameline = []
    dobline = []
    panline = []
    text0 = []
    text1 = []
    text2 = []
    govRE_str = '(GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT\
                |PARTMENT|ARTMENT|INDIA|NDIA)$'
    numRE_str = '(Number|umber|Account|ccount|count|Permanent|\
                ermanent|manent)$'

    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)

    pan_type = ''
    x = 0
    while x < len(text1):
        if text1[x].find("Father") != -1:
            pan_type = "newpan"
            break
        else:
            x = x + 1

    if pan_type == '':
        pan_type = "oldpan"

    if pan_type == 'newpan':
        lineno = 0

        for wordline in text1:
            xx = wordline.split()
            if ([w for w in xx if re.search(govRE_str, w)]):
                lineno = text1.index(wordline)
                break

        text0 = text1[lineno + 1:]
        print("\ntext0->", text0)

        x = 0
        while x < len(text0):
            if text0[x].find("Account") != -1 or text0[x].find("Number") != -1 or text0[x].find("Card") != -1:
                if text0[x + 1] != '':
                    pan = text0[x + 1]
                else:
                    pan = text0[x + 2]
                break
            else:
                x = x + 1

        x = 0

        while x < len(text0):
            if text0[x].find("Name") != -1:
                if text0[x + 1] != '':
                    name = text0[x + 1]
                else:
                    name = text0[x + 2]
                break
            else:
                x = x + 1

        x = 0
        while x < len(text0):
            if text0[x].find("Father") != -1:
                if text0[x + 1] != '':
                    fname = text0[x + 1]
                else:
                    fname = text0[x + 2]
                if name == fname:
                    if text0[x - 1] != '':
                        name = text0[x - 1]
                    else:
                        name = text0[x - 2]
                break
            else:
                x = x + 1

        x = 0

        while x < len(text0):
            if text0[x].find("Birth") != -1:
                if len(text0[x + 1]) == 0:
                    dob = text0[x + 2][0:10]
                else:
                    dob = text0[x + 1][0:10]
                break
            else:
                x = x + 1

        if dob == None:
            for no in text0:
                if re.match("^(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])[- /.](19|20)\d\d$", no):
                    dob = no

        with open('namedb.csv', 'rt') as f:
            reader = csv.reader(f)
            newlist = list(reader)
        newlist = sum(newlist, [])

        try:
            for x in text0:
                for y in x.split():
                    if (difflib.get_close_matches(y.upper(), newlist)):
                        nameline.append(x)
                        break
        except Exception as ex:
            pass

        dataobj = {}
        dataobj['Name'] = name
        dataobj['Father Name'] = fname
        dataobj['Date of Birth'] = dob
        dataobj['UID'] = pan
        dataobj['ID Type'] = "PAN"
        return dataobj

    else:
        lineno = 0

        for wordline in text1:
            xx = wordline.split()
            if ([w for w in xx if re.search(govRE_str, w)]):
                lineno = text1.index(wordline)
                break

        text0 = text1[lineno + 1:]
        text0 = list(filter(('').__ne__, text0))
        name = text0[0]
        fname = text0[1]
        dob = text0[2][0:10]
        pan = text0[4][0:10]

        dataobj = {}
        dataobj['Name'] = name
        dataobj['Father Name'] = fname
        dataobj['Date of Birth'] = dob
        dataobj['UID'] = pan
        dataobj['ID Type'] = "PAN"
        return dataobj


def adhaar_read_data(text):
    name = None
    gender = None
    ayear = None
    uid = None
    uuid = None
    yearline = []
    genline = []
    nameline = []
    text1 = []
    text2 = []
    genderStr = '(Female|Male|emale|male|ale|FEMALE|MALE|EMALE)$'
    lines = text
    for wordlist in lines.split('\n'):
        xx = wordlist.split()
        if [w for w in xx if re.search('(Year|Birth|irth|YoB|YOB:|DOB:|DOB|008)$', w)]:
            yearline = wordlist
            break
        else:
            text1.append(wordlist)

    if len(text1) >= 1:
        if len(text1) >= 2:
            if text1[-1] == '':
                name = text1[-2]
            else:
                name = text1[-1]
        else:
            name = text1[-1]

    try:
        text2 = text.split(yearline, 1)[1]
    except Exception:
        pass

    try:
        yearline = re.split('Year|Birth|irth|YoB|YOB:|DOB:|DOB|008|DOS', yearline)[1:]
        yearline = ''.join(str(e) for e in yearline)
        if yearline:
            ayear = dparser.parse(yearline, fuzzy=True).year

    except Exception:
        pass

    try:
        for wordlist in lines.split('\n'):
            xx = wordlist.split()
            if [w for w in xx if re.search(genderStr, w)]:
                genline = wordlist
                break

        if 'Female' in genline or 'FEMALE' in genline:
            gender = "Female"
        elif 'Male' in genline or 'MALE' in genline:
            gender = "Male"

        text2 = text.split(genline, 1)[1]
    except Exception:
        pass

    uid = set()
    try:
        newlist = []
        for xx in text2.split('\n'):
            newlist.append(xx)
        newlist = list(filter(lambda x: len(x) > 12, newlist))
        for no in newlist:
            if re.match("^[0-9 ]+$", no):
                uid.add(no)

    except Exception:
        pass

    if newlist == []:
        newlist = list(filter(lambda x: len(x) >= 12, text1))
        for no in newlist:
            if re.match("^[0-9 ]+$", no):
                uid.add(no)

    if len(list(uid)) >= 1:
        uuid = list(uid)[0]

    dataobj = {}
    dataobj['Name'] = name
    dataobj['Date of Birth'] = yearline[:11]
    dataobj['UID'] = uuid
    dataobj['Sex'] = gender
    dataobj['ID Type'] = "Aadhar"
    return dataobj


def extract_card_details(filename):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(img, cv2.CV_64F).var()
    if var < 100:
        gaussian_blur = cv2.GaussianBlur(img, (7, 7), 2)
        sharpened2 = cv2.addWeighted(img, 3.5, gaussian_blur, -2.5, 0)
        cv2.imwrite("sh2.jpg", sharpened2)
        filename = r"sh2.jpg"

    text = pytesseract.image_to_string(Image.open(filename), lang='eng')

    text_output = open('output.txt', 'w', encoding='utf-8')
    text_output.write(text)
    text_output.close()

    file = open('output.txt', 'r', encoding='utf-8')
    text = file.read()

    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)
    if "income" in text.lower() or "tax" in text.lower() or "department" in text.lower() or "permanent" in text.lower() or "account" in text.lower() or "number" in text.lower() or "card" in text.lower() or "father" in text.lower() or "signature" in text.lower():
        data = pan_read_data(text)
    elif "male" in text.lower() or "VID" in text:
        data = adhaar_read_data(text)

    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str
    with io.open('info.json', 'w', encoding='utf-8') as outfile:
        data = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(data))
    with open('info.json', encoding='utf-8') as data:
        data_loaded = json.load(data)
    print(data_loaded)
    return data_loaded


@app.route('/')
def index():
    return render_template('login.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, data, approvedby

    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1

        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                p = os.path.sep.join(['videos', "vid_{}.avi".format(str(now).replace(":", ''))])
                out = cv2.VideoWriter(p, fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()


    elif request.method == 'GET':
        return render_template('Security Dashboard.html', data={}, approvedby=approvedby)
    time.sleep(6)
    return render_template('Security Dashboard.html', data=data, approvedby=approvedby)


@app.route('/admindash')
def admindash():
    visitobj = list(visitorlogtable.find())
    activeobj = list(activevisitorstable.find())
    reqobj = list(reqvistable.find())
    rejectobj = list(rejectedvistable.find())
    adminobj = list(adminlog.find())
    secobj = list(securitylog.find())
    countsecurityusers = len(secobj)
    countadminusers = len(adminobj)
    countvis = len(visitobj)
    countact = len(activeobj)
    return render_template('Admin Dashboard.html', visitobj=visitobj, activeobj=activeobj, reqobj=reqobj,
                           rejectobj=rejectobj, secobj=secobj, adminobj=adminobj, countsecurityusers=countsecurityusers,
                           countadminusers=countadminusers, countvis=countvis, countact=countact)


@app.route('/securitydash')
def securitydash():
    visitobj = list(visitorlogtable.find())
    activeobj = list(activevisitorstable.find())
    return render_template('Security Dashboard.html', data={}, visitobj=visitobj, activeobj=activeobj,
                           approvedby=approvedby)


@app.route('/logoutadmin')
def logoutadmin():
    session.pop("emailadmin", None)
    return redirect(url_for('index'))


@app.route('/logoutsecurity')
def logoutsecurity():
    session.pop("emailsecurity", None)
    return redirect(url_for('index'))


@app.route('/auth', methods=['GET', 'POST'])
def auth():
    global approvedby
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        visitobj = list(visitorlogtable.find())
        activeobj = list(activevisitorstable.find())
        reqobj = list(reqvistable.find())
        rejectobj = list(rejectedvistable.find())
        adminobj = list(adminlog.find())
        secobj = list(securitylog.find())
        countsecurityusers = len(secobj)
        countadminusers = len(adminobj)
        countvis = len(visitobj)
        countact = len(activeobj)

        emailsplit = email.split("@")
        approvedby = emailsplit[0]

        for x in collection.find():
            if x['Email'] == email and x['Password'] == password and email.find("admin") != -1:
                session['emailadmin'] = email
                return render_template('Admin Dashboard.html', visitobj=visitobj, activeobj=activeobj, reqobj=reqobj,
                                       rejectobj=rejectobj, secobj=secobj, adminobj=adminobj,
                                       countsecurityusers=countsecurityusers, countadminusers=countadminusers,
                                       countvis=countvis, countact=countact)
            if x['Email'] == email and x['Password'] == password and email.find("security") != -1:
                session['emailsecurity'] = email
                return render_template('Security Dashboard.html', data={}, visitobj=visitobj, activeobj=activeobj,
                                       approvedby=approvedby)

        return render_template('login.html')


@app.route('/deletevis/<uid>', methods=['POST', 'GET'])
def deletevis(uid):
    global approvedby
    activevisitorstable.delete_one({"UID": uid})
    now1 = datetime.datetime.now()
    dt_string = now1.strftime("%d/%m/%Y %H:%M:%S")
    myquery = {"UID": uid}
    newvalues = {"$set": {"Exittime": dt_string}}
    visitorlogtable.update_one(myquery, newvalues)
    return redirect(url_for('securitydash'))


@app.route('/acceptvis/<uid>', methods=['POST', 'GET'])
def acceptvis(uid):
    # global approvedby
    element1 = reqvistable.find_one({"UID": uid},
                                    {"_id": 0, "Name": 1, "Gender": 1, "Card": 1, "UID": 1, "Date": 1, "Purpose": 1,
                                     "Email": 1, "Phone": 1, "Approvedby": 1, "Exittime": 1})
    reqvistable.delete_one({"UID": uid})
    visitorlogtable.insert_one(element1)
    activevisitorstable.insert_one(element1)
    return redirect(url_for('admindash'))


@app.route('/rejectvis/<uid>', methods=['POST', 'GET'])
def rejectvis(uid):
    # global approvedby
    element2 = reqvistable.find_one({"UID": uid},
                                    {"_id": 0, "Name": 1, "Gender": 1, "Card": 1, "UID": 1, "Date": 1, "Purpose": 1,
                                     "Email": 1, "Phone": 1, "Approvedby": 1, "Exittime": 1})
    reqvistable.delete_one({"UID": uid})
    rejectedvistable.insert_one(element2)
    return redirect(url_for('admindash'))


@app.route('/addsec', methods=['GET', 'POST'])
def addsec():
    if request.method == 'POST':
        if request.form['submit'] == 'pass':
            name1 = request.form['fullname']
            email1 = request.form['addemail']
            phone = request.form['phone']
            job1 = request.form['jobtitle']
            password = request.form['password']

            daobject = {
                "Name": name1,
                "Email": email1,
                "Phone": phone,
                "Job": job1,
                "Password": password,
            }

            collection.insert_one(daobject)
            securitylog.insert_one(daobject)
    return redirect(url_for('admindash'))


@app.route('/addadmin', methods=['GET', 'POST'])
def addadmin():
    if request.method == 'POST':
        if request.form['submit1'] == 'pass':
            name2 = request.form['fullname']
            email2 = request.form['addemail']
            phone2 = request.form['phone']
            job2 = request.form['jobtitle']
            password2 = request.form['password']

            adobject = {
                "Name": name2,
                "Email": email2,
                "Phone": phone2,
                "Job": job2,
                "Password": password2,
            }

            collection.insert_one(adobject)
            adminlog.insert_one(adobject)
    return redirect(url_for('admindash'))


@app.route('/deleteuser/<Phone>', methods=['POST', 'GET'])
def deleteuser(Phone):
    collection.delete_one({"Phone": Phone})
    securitylog.delete_one({"Phone": Phone})
    adminlog.delete_one({"Phone": Phone})
    return redirect(url_for('admindash'))


@app.route('/updateusers/<id>', methods=['POST', 'GET'])
def updateusers(id):
    users = collection.db.users
    items = users.find_one({'_id': ObjectId(id)})

    if request.method == 'POST':
        if request.form['submit'] == 'pass':
            myquery = {'_id': ObjectId(id)}

            updatelog = {"$set":
                             {"Name": request.form.get('Name'),
                              "Email": request.form.get('Email'),
                              "Phone": request.form.get('Phone'),
                              "Job": request.form.get('Job'),
                              "Password": request.files.get('Password'),
                              "date": datetime.datetime.utcnow()
                              }
                         }

    adminlog.update_one(myquery, updatelog)
    collection.update_one(myquery, updatelog)
    securitylog.update_one(myquery, updatelog)

    return redirect(url_for('admindash'))


@app.route('/visitor', methods=['GET', 'POST'])
def visitor():
    global approvedby, dobee, dataobject1
    if request.method == 'POST':
        if request.form['submit'] == 'pass':
            name = request.form['name']
            father = request.form['father']
            dob = request.form['dob']
            gender = request.form['gender']
            uid = request.form['uid']
            date = request.form['Date']
            purpose = request.form['Purpose']
            email = request.form['Email']
            phone = request.form['Phone']
            apprv = request.form['Approvedby']
            card = request.form['card']

            dataobject1 = {
                "Name": name,
                "Gender": gender,
                "Card": card,
                "UID": uid,
                "Date": date,
                "Purpose": purpose,
                "Email": email,
                "Phone": phone,
                "Approvedby": apprv,
                "Exittime": ""
            }

            reqvistable.insert_one(dataobject1)
            dobee = 1

    return redirect(url_for('securitydash'))


if __name__ == '__main__':
    app.run(debug=True)
