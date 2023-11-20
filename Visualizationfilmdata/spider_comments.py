import json
import csv
import requests
from lxml import etree
import re
from pymysql import *

conn = connect(host='localhost',user='root',password='root',database='doubanmovie',port=3306)
cursor = conn.cursor()

def querys(sql,params,type='no_select'):
    params = tuple(params)
    cursor.execute(sql,params)
    if type != 'no_select':
        data_list = cursor.fetchall()
        conn.commit()
        return data_list
    else:
        conn.commit()
        return '数据库语句执行成功'

def getAllData():
    def map_fn(item):
        item = list(item)
        if item[1] == None:
            item[1] = '无'
        else:
            item[1] = item[1].split(',')
        if item[4] == None:
            item[4] = '无'
        else:
            item[4] = item[4].split(',')
        item[7] = item[7].split(',')
        if item[8] == None:
            item[8] = '中国大陆'
        else:
            item[8] = item[8].split(',')
        if item[9] == None:
            item[9] = '汉语普通话'
        else:
            item[9] = item[9].split(',')
        item[13] = item[13].split(',')
        item[16] = item[16].split(',')
        item[15] = json.loads(item[15])
        return item
    allData = querys('select * from movies',[],'select')
    allData = list(map(map_fn,list(allData)))
    with open('./top25MovieId.csv', 'a', newline='') as f:
        for i in range(25):
            f.write(allData[i][18] + '\n')
    return allData

def spider_main():
    with open('./top25MovieId.csv','r') as f:
        for i in f.readlines():
            mId = re.findall('\d+',i)[0]
            for j in range(5):
                base_url = 'https://movie.douban.com/subject/{}/reviews?start={}'.format(mId, j * 20)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'
                }
                resp = requests.get(base_url, headers=headers)
                xpathHtml = etree.HTML(resp.text)
                # 电影名
                movieName = xpathHtml.xpath('//div[@class="subject-title"]/a/text()')[0][2:]
                print(movieName)
                divs = xpathHtml.xpath('//div[@class="review-list  "]/div')
                # 内容
                for div in divs:
                    content = div.xpath('.//div[@class="short-content"]/text()')
                    querys('insert into comments(movieName,commentContent) values(%s,%s)',[movieName,content[0]])


if __name__ == '__main__':
    spider_main()