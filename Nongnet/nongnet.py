import pandas as pd
import urllib.request
import json
import datetime
import os

today = datetime.datetime.today()
# yesterday = (today - datetime.timedelta(1800)).strftime('%Y%m%d')
# url = 'https://www.nongnet.or.kr/api/whlslDstrQr.do?sdate=' # sdate = 날짜

# response = urllib.request.urlopen(url+yesterday).read()
# print(response)
# response = json.loads(response)
# print(response)

# data = pd.DataFrame(response['data'])
# print(data)
# data.to_csv(f"{yesterday}.csv")
# print(data)

try:
    basedir = "Nongnet"
    for delta in range(2820, 3190):
        print(f"***** {delta} *****")
        date = (today - datetime.timedelta(delta)).strftime("%Y%m%d")
        print(date)
        url = "https://www.nongnet.or.kr/api/whlslDstrQr.do?sdate="  # sdate = 날짜
        response = urllib.request.urlopen(url + date).read()
        response = json.loads(response)

        yy_path = os.path.join(basedir, date[2:4])
        if not os.path.exists(yy_path):
            os.mkdir(yy_path)

        if len(response["data"]) > 0:
            data = pd.DataFrame(response["data"])

            print(f"Date ({date}) DATA IS SAVED")
            print(data[:2])
            dd_path = os.path.join(yy_path, f"{date}.csv")
            data.to_csv(dd_path)

except KeyboardInterrupt:
    print("Abort!")
