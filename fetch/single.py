import os
import urllib.request
from multiprocessing import Process


def DownloadFile(url, path, filename):
    try:
        #urllib.request.urlretrieve(url, os.path.join(path, filename))
        req = urllib.request.Request(
            url,
            data=None,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            }
        )
        resp = urllib.request.urlopen(req)
        with open(os.path.join(path, filename), 'wb') as f:
            f.write(resp.read())
        resp.close()
    except Exception as e:
        print(e)
        print("cannot download : " + url)
        print("cannot download : " + os.path.join(path, filename))


def ConcurrentDownload(url, path, filename):
    p = None
    if not os.path.isfile(os.path.join(path, filename)):
        p = Process(target=DownloadFile, args=(url, path, filename))
        p.start()
    return p