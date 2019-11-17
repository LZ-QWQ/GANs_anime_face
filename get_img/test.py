
import requests
import os
from bs4 import BeautifulSoup

def download(url, filename):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url, stream=True, timeout=60)
        with open(filename, 'wb') as f:
            f.write(r.content)
            f.flush()
        return filename

    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
    except Exception:
        if os.path.exists(filename):
            os.remove(filename)
        print('我也不知道什么情况~')

        raise Exception

if __name__=='__main__':

    if os.path.exists('imgs') is False:
        os.makedirs('imgs')

    startpage=1
    numpages=10

    for i in range(startpage, numpages + 1):
        url = 'http://konachan.net/post?page=%d&tags=' % i
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        for img_url in soup.find_all('a', class_="thumb"):
            target_url = 'http://konachan.net'+img_url['href']
            html2=requests.get(target_url).text
            print(html2)
            soup2 = BeautifulSoup(html2, 'html.parser')
            img =soup2.find('img', class_="image")
            img_down_url=img['src']
            filename = os.path.join('imgs',img_down_url.split('/')[-1])
            download(img_down_url, filename)
        print('%d / %d' % (i, numpages))

