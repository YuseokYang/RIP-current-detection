import requests

with open("HD_PARA2_20190630_090624.jpg", "rb") as f:
    files = {"file": ("HD_PARA2_20190630_090624.jpg", f, "image/jpeg")} #해수욕장 CCTV 이미지
    res = requests.post("http://127.0.0.1:8000/predict", files=files)

print(res.json())