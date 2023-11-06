from fastapi import FastAPI
from pydantic import BaseModel # 타입스크립트의 interface 같은 애인 듯.
import emotion_dl as EMOTION
import ai_apis as APIS
import urllib
import json
from hooks import random_string
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from starlette.config import Config
from dotenv import load_dotenv

load_dotenv()
config = Config('.env') # env 설정.
IMGUR_ID = config('IMGUR_ID')

origins = [
    'http://localhost:3000'
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)
class Item(BaseModel):
    text: str
@app.get("/")
def root():
    return {"msg" : "python api의 메인 페이지."}

# ['불안', '놀람', '분노', '슬픔', '중립', '행복', '당황']
emotions = {
    '0': '불안',
    '1': '놀람',
    '2': '분노',
    '3': '슬픔',
    '4': '중립',
    '5': '행복',
    '6': '당황',
}
@app.post("/predict/emotion/")
async def emotion_classify(text: Item):
    predicted = await EMOTION.predict(text.text)
    predicted = list(predicted)
    result = emotions.get(str(predicted[0]))
    return {"result" : result}

@app.post('/predict/summary/')
async def text_summary(text: Item):
    result = await APIS.summary(text.text)
    return {"result":result}

@app.post('/predict/trans/')
async def text_trans(text: Item):
    result = await APIS.translate(text.text)
    result = json.loads(result)
    return {"result" : result['message']['result']['translatedText'] }

@app.post("/predict/img/")
async def create_img(text: Item):
    result = await APIS.create_image(text.text)
    result = Image.open(urllib.request.urlopen(result.get('images')[0].get('image')))
    str = random_string()
    result.save(f'./img-{str}.png')
    name = f'./img-{str}.png'
    upload_url = 'https://api.imgur.com/3/image'
    headers = {
        "Authorization":f"Client-ID {IMGUR_ID}"
    }
    try:
        with open(name, 'rb') as file:
            # 파일 업로드
            files = {'image': (name, file)}
            response = requests.post(upload_url, headers=headers, files=files)

            if response.status_code == 200:
                imgur_response = response.json()
                imgur_link = imgur_response['data']['link']
                # os.remove(name)
                return {"result": imgur_link}
            else:
                return {"result":"ERROR"}
    except FileNotFoundError:
        return {"result":"File Not Found"}
    except Exception as e:
        return {"result":"ERROR!"}
    