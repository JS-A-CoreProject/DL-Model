import requests
import json
import io
import base64
# import urllib
import urllib.request
from PIL import Image
from starlette.config import Config
from dotenv import load_dotenv

load_dotenv()
config = Config('.env') # env 설정.


REST_API_KEY = config('REST_API_KEY')
NAVER_CLIENT_ID = config('NAVER_CLIENT_ID')
NAVER_CLIENT_KEY = config('NAVER_CLIENT_KEY')
PAPAGO_ID = config('PAPAGO_ID')
PAPAGO_KEY = config('PAPAGO_KEY')

async def summary(prompt):
    url = 'https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize'
    headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-NCP-APIGW-API-KEY-ID': NAVER_CLIENT_ID,
            'X-NCP-APIGW-API-KEY': NAVER_CLIENT_KEY
        }
    data = {
        "document": {
        "content": prompt
        },
        "option": {
        "language": "ko",
        "model": "general",
        "tone": 0,
        "summaryCount": 1
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('UTF-8'))
    print(response)
    return response.text

async def translate(text):
    client_id = PAPAGO_ID
    client_secret = PAPAGO_KEY
    encText = urllib.parse.quote(text)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        return response_body.decode('utf-8')
    else:
        return False
    

def imageToString(img):
    img_bye_arr = io.BytesIO
    img.save(img_bye_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_bye_arr.getvalue()).decode('ascii')
    return my_encoded_img

def stringToImage(base64_string, mode='RGBA'):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata)).convert(mode)
    return img

async def create_image(text):
    r = requests.post(
        'https://api.kakaobrain.com/v2/inference/karlo/t2i',
        json = {
            'prompt': text + ', 삽화',
            'negative_prompt': 'text'
        },
        headers = {
            'Authorization': f'KakaoAK {REST_API_KEY}',
            'Content-Type': 'application/json'
        }
    )
    # 응답 JSON 형식으로 변환
    response = json.loads(r.content)
    return response