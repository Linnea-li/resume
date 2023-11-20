import json
from flask import Flask,request,render_template,session,redirect
import re
from utils.query import querys
from utils.homeData import *
from utils.timeData import *
from utils.rateData import *
from utils.addressData import *
from utils.typeData import *
from utils.tablesData import *
from utils.actor import *
from word_cloud_picture import get_img

import random
app = Flask(__name__)
app.secret_key = 'This is a app.secret_Key , You Know ?'

@app.route('/')
def every():
    return render_template('login.html')

@app.route("/home")
def home():
    email = session['email']
    allData = getAllData()
    maxRate = getMaxRate()
    maxCast = getMaxCast()
    typesAll = getTypesAll()
    maxLang = getMaxLang()
    types = getType_t()
    row,column = getRate_t()
    tablelist = getTableList()
    return render_template(
        "index.html",
        email=email,
        dataLen = len(allData),
        maxRate=maxRate,
        maxCast=maxCast,
        typeLen = len(typesAll),
        maxLang = maxLang,
        types=types,
        row=list(row),
        column=list(column),
        tablelist=tablelist
    )

@app.route("/login",methods=['GET','POST'])
def login():
    if request.method == 'POST':
        request.form = dict(request.form)

        def filter_fns(item):
            return request.form['email'] in item and request.form['password'] in item

        users = querys('select * from user', [], 'select')
        login_success = list(filter(filter_fns, users))
        if not len(login_success):
            return '账号或密码错误'

        session['email'] = request.form['email']
        return redirect('/home', 301)

    else:
        return render_template('./login.html')

@app.route("/registry",methods=['GET','POST'])
def registry():
    if request.method == 'POST':
        request.form = dict(request.form)
        if request.form['password'] != request.form['passwordCheked']:
            return '两次密码不符'
        else:
            def filter_fn(item):
                return request.form['email'] in item

            users = querys('select * from user', [], 'select')
            filter_list = list(filter(filter_fn, users))
            if len(filter_list):
                return '该用户名已被注册'
            else:
                querys('insert into user(email,password) values(%s,%s)',
                       [request.form['email'], request.form['password']])

        session['email'] = request.form['email']
        return redirect('/home', 301)

    else:
        return render_template('./register.html')

@app.route("/search/<int:searchId>",methods=['GET','POST'])
def search(searchId):
    email = session['email']
    allData = getAllData()
    data = []
    if request.method == 'GET':
        if searchId == 0:
            return render_template(
                'search.html',
                idData=data,
                email=email
            )

        for i in allData:
            if i[0] == searchId:
                data.append(i)
        return render_template(
                'search.html',
                data=data,
                email=email
            )
    else:
        searchWord = dict(request.form)['searchIpt']
        def filter_fn(item):
            if item[3].find(searchWord) == -1:
                return False
            else:
                return True
        data = list(filter(filter_fn,allData))
        return render_template(
            'search.html',
            data=data,
            email=email
        )

@app.route("/time_t",methods=['GET','POST'])
def time_t():
    email = session['email']
    row,column = getTimeList()
    moveTimeData = getMovieTimeList()
    return render_template(
        'time_t.html',
        email=email,
        row=list(row),
        column=list(column),
        moveTimeData=moveTimeData
    )

@app.route("/rate_t/<type>",methods=['GET','POST'])
def rate_t(type):
    email = session['email']
    typeAll = getTypesAll()
    rows,columns = getMean()
    x,y,y1 = getCountryRating()
    if type == 'all':
        row, column = getRate_t()
    else:
        row,column = getRate_tType(type)
    if request.method == 'GET':
        starts,movieName = getStart('长津湖')
    else:
        searchWord = dict(request.form)['searchIpt']
        starts,movieName = getStart(searchWord)
    return render_template(
        'rate_t.html',
        email=email,
        typeAll=typeAll,
        type=type,
        row=list(row),
        column=list(column),
        starts=starts,
        movieName=movieName,
        rows = rows,
        columns = columns,
        x=x,
        y=y,
        y1=y1
    )

@app.route("/address_t",methods=['GET','POST'])
def address_t():
    email = session['email']
    row,column = getAddressData()
    rows,columns = getLangData()
    return render_template('address_t.html',row=row,column=column,rows=rows,columns=columns,email=email)

@app.route('/type_t',methods=['GET','POST'])
def type_t():
    email = session['email']
    result = getMovieTypeData()
    return render_template('type_t.html',result=result,type_t=type_t,email=email)

@app.route('/actor_t')
def actor_t():
    email = session['email']
    x,y = getAllActorMovieNum()
    x1,y1 = getAllDirectorMovieNum()
    return render_template('actor_t.html',email=email,x=x,y=y,x1=x1,y1=y1)

@app.route("/movie/<int:id>")
def movie(id):
    allData = getAllData()
    idData = {}
    for i in allData:
        if i[0] == id:
            idData = i
    return render_template('movie.html',idData=idData)

@app.route('/tables/<int:id>')
def tables(id):
    if id == 0:
        tablelist = getTableList()
    else:
        deleteTableId(id)
        tablelist = getTableList()
    return render_template('tables.html',tablelist=tablelist)

@app.route('/title_c')
def title_c():
    return render_template('title_c.html')

@app.route('/summary_c')
def summary_c():
    return render_template('summary_c.html')

@app.route('/casts_c')
def casts_c():
    return render_template('casts_c.html')

@app.route('/comments_c',methods=['GET','POST'])
def comments_c():
    email = session['email']
    if request.method == 'GET':
        return render_template('comments_c.html', email=email)
    else:
        searchWord = dict(request.form)['searchIpt']
        randomInt = random.randint(1,10000000)
        get_img('commentContent','./static/4.jpg',f'./static/{randomInt}.jpg',searchWord)
        return render_template('comments_c.html', email=email,imgSrc=f'{randomInt}.jpg')

@app.before_request
def before_requre():
    pat = re.compile(r'^/static')
    if re.search(pat,request.path):
        return
    if request.path == "/login" :
        return
    if request.path == '/registry':
        return
    uname = session.get('email')
    if uname:
        return None

    return redirect("/login")


if __name__ == '__main__':
    app.run()
