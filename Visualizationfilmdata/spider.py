'爬取的字段信息：电影名，评分，封面图，详情url，上映时间，导演，类型，制作国家，语言，片长，电影简介，星星比例，多少人评价，预告片，前五条评论，五张详情图片'
import re
import pandas as ps
import requests
import jsonpath
import random
import time
from pymysql import *
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
import json

engine = create_engine('mysql+pymysql://root:123456@localhost:3306/doubanmovie')

def init():
    conn = connect(host='localhost',user='root',password='123456',database='doubanmovie',port=3306,charset='utf8mb4')
    sql = '''
        create table movie(
            id int primary key auto_increment,
            directors varchar(255),
            rate varchar(255),
            title varchar(255),
            casts varchar(255),
            cover varchar(255),
            year varchar(255),
            types varchar(255),
            country varchar(255),
            lang varchar(255),
            time varchar(255),
            moveiTime varchar(255),
            comment_len varchar(255),
            starts varchar(255),
            summary varchar(2555),
            comments text,
            imgList varchar(2555),
            movieUrl varchar(255)
        )
    '''
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()

def save_to_csv(df):
    df.to_csv('./datas.csv')

def save_to_sql():
    df= ps.read_csv("./datas.csv",index_col=0)
    df.to_sql('movies_copy',con=engine,index=False,if_exists ='append')

def spider(spiderTarget,start):
    # 每次调用spider获取20条数据
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36',
        'Referer':'https://movie.douban.com/tag/',
        'Cookie':'bid=6mHemZBr91A; __gads=ID=f6d1e29943a2289e-2235597b46cf0014:T=1637928036:RT=1637928036:S=ALNI_MbUV2rRDkg5u38czBTVDBFS0PLajA; ll="108305"; _vwo_uuid_v2=D05DF24F53B472D086C01A79B01735762|e5e120c8d217d8191d1303c2a5b5aa04; gr_user_id=6441c017-d74b-422f-af14-93a11a57112d; __yadk_uid=tajeNgKg6NT6nhEQczKfmecGcZqdVBXY; douban-fav-remind=1; __utmv=30149280.23512; _ga=GA1.2.1042859692.1637928038; viewed="2995812_1458367_6816154_1416697_1455695_1986653_1395176_3040149_1427374_4913064"; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1652841585%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DVlfhYZ2MEHRBSLvH1rcPwd4AYRrL-DQrWxaEeqtUjfETnWetwL98pNUbJ__vgCwN%26wd%3D%26eqid%3Da1df55aa0002864e0000000662845c64%22%5D; _pk_ses.100001.4cf6=*; ap_v=0,6.0; __utma=30149280.1042859692.1637928038.1649577909.1652841585.36; __utmb=30149280.0.10.1652841585; __utmc=30149280; __utmz=30149280.1652841585.36.12.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.1442664133.1641956556.1648885892.1652841585.22; __utmb=223695111.0.10.1652841585; __utmc=223695111; __utmz=223695111.1652841585.22.8.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; dbcl2="235123238:u9rdv3vTMd0"; ck=nl8E; _pk_id.100001.4cf6=023cbddd8ff1a247.1641956556.21.1652843267.1648885892.; push_noty_num=0; push_doumail_num=0'
    }
    params = {
        'start':start
    }
    movieAllRes = requests.get(spiderTarget,params=params,headers=headers)
    movieAllRes = movieAllRes.json()
    detailUrls = jsonpath.jsonpath(movieAllRes,'$.data..url')
    moveisInfomation = jsonpath.jsonpath(movieAllRes,'$.data')[0]
    for i,moveInfomation in enumerate(moveisInfomation):
        try:
            resultData = {}
            # 详情
            resultData['detailLink'] = detailUrls[i]
            # 导演（数组）
            resultData['directors'] = ','.join(moveInfomation['directors'])
            # 评分
            resultData['rate'] = moveInfomation['rate']
            # 影片名
            resultData['title'] = moveInfomation['title']
            # 主演（数组）
            resultData['casts'] = ','.join(moveInfomation['casts'])
            # 封面
            resultData['cover'] = moveInfomation['cover']

            # =================进入详情页====================
            detailMovieRes = requests.get(detailUrls[i], headers=headers)
            soup = BeautifulSoup(detailMovieRes.text, 'lxml')
            # 上映年份
            resultData['year'] = re.findall(r'[(](.*?)[)]',soup.find('span', class_='year').get_text())[0]
            types = soup.find_all('span',property='v:genre')
            for i,span in enumerate(types):
                types[i] = span.get_text()
            # 影片类型（数组）
            resultData['types'] = ','.join(types)
            country = soup.find_all('span',class_='pl')[4].next_sibling.strip().split(sep='/')
            for i,c in enumerate(country):
                country[i] = c.strip()
            # 制作国家（数组）
            resultData['country'] = ','.join(country)
            lang = soup.find_all('span', class_='pl')[5].next_sibling.strip().split(sep='/')
            for i, l in enumerate(lang):
                lang[i] = l.strip()
            # 影片语言（数组）
            resultData['lang'] = ','.join(lang)

            upTimes = soup.find_all('span',property='v:initialReleaseDate')
            upTimesStr = ''
            for i in upTimes:
                upTimesStr = upTimesStr + i.get_text()
            upTime = re.findall(r'\d*-\d*-\d*',upTimesStr)[0]
            # 上映时间
            resultData['time'] = upTime
            if soup.find('span',property='v:runtime'):
                # 时间长度
                resultData['moveiTime'] = re.findall(r'\d+',soup.find('span',property='v:runtime').get_text())[0]
            else:
                # 时间长度
                resultData['moveiTime'] = random.randint(39,61)
            # 评论个数
            resultData['comment_len'] = soup.find('span',property='v:votes').get_text()
            starts = []
            startAll = soup.find_all('span',class_='rating_per')
            for i in startAll:
                starts.append(i.get_text())
            # 星星比例（数组）
            resultData['starts'] = ','.join(starts)
            # 影片简介
            resultData['summary'] = soup.find('span',property='v:summary').get_text().strip()

            # 五条热评
            comments_info = soup.find_all('span', class_='comment-info')
            comments = [{} for x in range(5)]
            for i, comment in enumerate(comments_info):
                comments[i]['user'] = comment.contents[1].get_text()
                comments[i]['start'] = re.findall('(\d*)', comment.contents[5].attrs['class'][0])[7]
                comments[i]['time'] = comment.contents[7].attrs['title']
            contents = soup.find_all('span', class_='short')
            for i in range(5):
                comments[i]['content'] = contents[i].get_text()
            resultData['comments'] = json.dumps(comments)

            # 五张详情图
            imgList = []
            lis = soup.select('.related-pic-bd img')
            for i in lis:
                imgList.append(i['src'])
            resultData['imgList'] = ','.join(imgList)
            # =================详情页结束===================


            # =================进入电影页===================
            if soup.find('a',class_='related-pic-video'):
                movieUrl = soup.find('a', class_='related-pic-video').attrs['href']
                foreshowMovieRes = requests.get(movieUrl,headers=headers)
                foreshowMovieSoup = BeautifulSoup(foreshowMovieRes.text,'lxml')
                movieSrc = foreshowMovieSoup.find('source').attrs['src']
                # 电影路径
                resultData['movieUrl'] = movieSrc
            else:
                resultData['movieUrl'] = '0'

            # =================进入电影页结束===================
            result.append(resultData)
            print('已经爬取%d条数据' % len(result))
        except :
            return





def main():
    global result
    result = []
    with open('./pageNum.txt','r') as fr:
        page = int(fr.readlines()[-1])
        print('开始爬取第%s个20' % page)
        spider(spiderTarget, page * 20)

        time.sleep(5)
        df = ps.DataFrame(result)
        save_to_csv(df)
        print('导入csv成功......')
        time.sleep(5)
        save_to_sql()
        print('导入sql成功......')
        with open('./pageNum.txt','a') as fa:
            fa.write(str(page + 1) + '\n')
        result = []
        for i in range(len(result)):
            result.pop()
    main()

if __name__ == '__main__':
    print('爬虫已开始...')
    spiderTarget = 'https://movie.douban.com/j/new_search_subjects?'
    main()


