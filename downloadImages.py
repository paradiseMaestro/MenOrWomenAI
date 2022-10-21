import shutil
import time
import requests

def Q(f):
    os.mkdir(f)
    i = 0
    while (i < 10000):
        url = 'https://thispersondoesnotexist.com/image'
        response = requests.get(url, stream=True)
        with open('./'+f+'/'+str(i)+'.jpg', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        i = i + 1
        time.sleep(1)
Q('e')