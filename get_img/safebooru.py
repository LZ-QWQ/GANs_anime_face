
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
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url,filename,cbk)
        print('done~~~~')
    except socket.timeout:
        count = 1
        if os.path.exists(filename):
            os.remove(filename)

        while count <= 5:
            try:
               opener=urllib.request.build_opener()
               opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
               urllib.request.install_opener(opener)
               urllib.request.urlretrieve(url,filename,cbk)
               print('done~~~~')
               break
            except socket.timeout:
               err_info = 'Reloading for %d time'%count if count == 1 else 'Reloading for %d times'%count
               print(err_info)
               count += 1
               if os.path.exists(filename):
                    os.remove(filename)

        if count > 5:
            print("downloading picture fialed!")

    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt

    except :
        print('未知错误')
        if os.path.exists(filename):
            os.remove(filename)

if __name__=='__main__':

    path='imgs_no_explicit_safebooru'
    if os.path.exists(path) is False:
        os.makedirs(path)

    startpage=2600000
    numpages=3500000
    
    for i in range(startpage, numpages + 1):
        proxies = { "http": "http://127.0.0.1:1080" }
        headers = {'content-type':'application/json'}
        #payload = {'login':'LZ_QAQ','api_key':'FGTgEpCuYRXkKZbhzDgR66S3','limit':'200','page':'a%d' %(i)}
        url='https://danbooru.donmai.us/posts.json?tags=bang_dream%21&page=200'
        #url='https://safebooru.donmai.us/posts.json?page=a990000'
        html = requests.get(url,proxies=proxies,headers=headers)
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

