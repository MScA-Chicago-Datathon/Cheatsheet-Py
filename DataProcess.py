#Citadel Datathon, Chicago, IL, May 12th 2017
Teammates = "Chen, Yuxiang" + "Pan, Yuanyuan" + "Shi, Zhiyin" +  "Yu, Qianfan"

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A. Import Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
#A1 From html
#Beautiful Soup Doc: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
import urllib.request
from   bs4 import BeautifulSoup

url = "hello.html"
my_html = urllib.request.urlopen(url)
my_soup = BeautifulSoup(my_html.read(), "html.parser")

my_soup.find_all()
my_soup.tag.contents[0]

#A2 From url link which automatecally downloads csv
import urllib.request
import requests

with requests.Session() as s:
    download = s.get(url_cur)
    decoded_content = download.content.decode('utf-8')
    csv_content = csv.reader(decoded_content.splitlines(), delimiter=',')

#A3 From url link which automatecally downloads zip
from zipfile import ZipFile
from io import BytesIO
import urllib.request
import pandas as pd

request = urllib.request.Request(dataUrl)
response = urllib.request.urlopen(request).read()
file = ZipFile(BytesIO(response))
NBI_txt = file.open("slubkin_992016-20170113122943.txt")
NBI_df = pd.read_csv(NBI_txt, encoding = "ISO-8859-1", low_memory = False)

#A4 From Excel to pandas df
import pandas as pd

dataStore = pd.ExcelFile(dataUrl)
dataParse = dataStore.parse("Data")

#A5 From Txt to csv
import csv

in_txt = csv.reader(open("NBI.txt","r",encoding= "ISO-8859-1"), delimiter = ',')

with open('NBI.csv', 'w') as pinapple:
    writer = csv.writer(pinapple)
    for row in in_txt: 
        writer.writerow(row)

#A6 From database to python
from sqlalchemy import *

engine = create_engine()
with engine.connect() as conn:
        res = conn.execute(query)
        return res

#A7 Read in csv
import csv

with open(train_file) as file:
    reader = csv.DictReader(file, delimiter=',')
    raw_data = [row for row in reader]


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
B. Export Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
#B1 Write csv
import csv

with open('file_name.csv', 'w') as cool:
    writer = csv.writer(cool)
    for row in file_to_be_write:
        writer.writerow(row) 

#create empty csv
with open('SDWIS.csv', 'w') as pepper:
    pass

#B2 pandas to sql db
from sqlalchemy import *

engine = create_engine()
dataNBIW.to_sql("shipping_bridge", engine, if_exists = "append", index = False)

