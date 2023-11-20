import re

str = 'https://movie.douban.com/subject/25845392/'

print(re.findall('\d+',str))