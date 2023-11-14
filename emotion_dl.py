from transformers import BertForSequenceClassification, AutoTokenizer
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # GPU를 사용하기 위해 CUDA 장치 객체를 생성
    print('GPU 사용:', torch.cuda.get_device_name(0))  # 사용 가능한 GPU의 이름 출력
else:
    device = torch.device("cpu")  # GPU를 사용할 수 없으면 CPU를 사용

loaded_model = BertForSequenceClassification.from_pretrained("./checkpoint-3082")

model_name = 'beomi/kcbert-base' # beomi/KcELECTRA-base-v2022
tokenizer = AutoTokenizer.from_pretrained(model_name)

# inputs.to('cuda')
# 모델로 예측 수행

async def predict(text):
    emotions = {
    '0': '불안',
    '1': '놀람',
    '2': '분노',
    '3': '슬픔',
    '4': '중립',
    '5': '행복',
    '6': '당황',
    }
    textList = text.split('\n')
    resultList = []
    for texts in textList:
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            predicts = torch.argmax(outputs.logits, dim=1)
            predicted_class = predicts.item()
            # r = list(predicted_class)
            resultList.append(emotions.get(str(predicted_class)))
    return resultList