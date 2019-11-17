
import requests
import os
from bs4 import BeautifulSoup
import urllib.request
import socket
socket.setdefaulttimeout(30)

def cbk(a, b, c):  
    '''回调函数 
    @a: 已经下载的数据块 
    @b: 数据块的大小 
    @c: 远程文件的大小 
    '''  
    per = 100.0 * a * b / c  
    if per > 100:  
        per = 100  
    print ('%.2f%%' % per,end='\r')#emmm就这样吧
  


def download(url, filename):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        urllib.request.urlretrieve(url,filename,cbk)
        print('done~~~~')
    except socket.timeout:
        count = 1
        while count <= 5:
            try:
               urllib.request.urlretrieve(url,filename,cbk)
               print('done~~~~')
               break
            except socket.timeout:
               err_info = 'Reloading for %d time'%count if count == 1 else 'Reloading for %d times'%count
               print(err_info)
               count += 1
        if count > 5:
            print("downloading picture fialed!")

    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt


if __name__=='__main__':

    path='imgs_no_explicit_yande'
    if os.path.exists(path) is False:
        os.makedirs(path)

    startpage=1
    numpages=10000
    
    for i in range(startpage, numpages + 1):
        proxies = { "http": "http://127.0.0.1:1080" }
        headers = {'content-type':'application/json'}
        payload = {'page': '%d'%i}
        url='https://yande.re/post.json?'
        html = requests.get(url,proxies=proxies,headers=headers,params=payload)
        #print(html.headers.get('content-type'))
        html=html.json()
        for emmm in html:
            temp=emmm['rating']
            if temp=='e':
                continue
            img_down_url = emmm['file_url']
            filename = os.path.join(path,img_down_url.split('/')[-1])
            download(img_down_url, filename)
        print('%d / %d' % (i, numpages))

