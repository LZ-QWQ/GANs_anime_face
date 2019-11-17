#一通乱改，我也不知道这个文件的代理是什么情况了
import requests
import os
from bs4 import BeautifulSoup
import urllib.request
from urllib.request import ProxyHandler, build_opener
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
        proxy_handler = ProxyHandler({'sock5': 'localhost:1080'})
        opener = build_opener(proxy_handler)
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url,filename,cbk)
        print('done~~~~')
    except socket.timeout:
        count = 1
        if os.path.exists(filename):
            os.remove(filename)
        while count <= 5:
            try:
               proxy_handler = ProxyHandler({'sock5': 'localhost:1080'})
               opener = build_opener(proxy_handler)
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

    path='imgs_no_explicit'
    if os.path.exists(path) is False:
        os.makedirs(path)

    startpage=6295
    numpages=6388

    for i in range(startpage, numpages + 1):
        proxies = { "http": "socks5://127.0.0.1:1080" ,"https": "socks5://127.0.0.1:1080" }
        #url = 'http://konachan.com/post.json?page=%d&login=LZ_QAQ&password_hash=So-I-Heard-You-Like-Mupkids-?--lz18825463589--;limits=100' % i
        url = 'https://konachan.net/post.json?page=%d&login=LZ_QAQ&password_hash=So-I-Heard-You-Like-Mupkids-?--lz18825463589--;limits=100' % i
        while(1):#就手动打断吧
            try:
                html = requests.get(url,proxies,timeout=30).json()
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                print('又error,requests.get这里有问题~')
                continue                
        for emmm in html:
            temp=emmm['rating']
            if temp=='e':
                continue
            img_down_url = emmm['file_url']
            filename = os.path.join(path,img_down_url.split('/')[-1])
            download(img_down_url, filename)
        print('%d / %d' % (i, numpages))

