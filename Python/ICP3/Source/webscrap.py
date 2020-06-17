#second question web scrap
import requests,json,string
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Deep_learning"
getdata = requests.get(url)
data = (BeautifulSoup(getdata.content, "html.parser"))
file_content = []
f = open('links_file.txt','w')

# print(data.find('h1',{"class":"firstHeading"}).text)
print(data.find(id='firstHeading').text)

for link in data.find_all('a'):
    if link.has_attr('href') and link.get('href').startswith('/'):
          print(link.text + " : " + link.get('href'))
          file_content.append(link.text + " : " + link.get('href'))
f.write(str(file_content))
# print(file_content)