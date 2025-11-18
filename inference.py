import torch
from torch import nn
from transformers import AutoTokenizer, BertModel, BartForConditionalGeneration, PreTrainedTokenizerFast

# ==========================================
# 0. Device
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================
# 1. Extractive Model 정의 (학습과 동일)
# ==========================================
class BertForExtSummarization(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ==========================================
# 2. 모델 로드
# ==========================================
# Extractive
MODEL_EXT = "./kobert"
tok_ext = AutoTokenizer.from_pretrained(MODEL_EXT)

ext_model = BertForExtSummarization(MODEL_EXT)
ext_model.load_state_dict(torch.load("best_kobert_ext.pt", map_location=device))
ext_model.to(device)
ext_model.eval()
print("Extractive model loaded.")

# Abstractive
MODEL_ABS = "./kobart"
tok_abs = PreTrainedTokenizerFast.from_pretrained(MODEL_ABS)

abs_model = BartForConditionalGeneration.from_pretrained(MODEL_ABS)
abs_model.load_state_dict(torch.load("best_kobart_abs.pt", map_location=device))
abs_model.to(device)
abs_model.eval()
print("Abstractive model loaded.")


# ==========================================
# 3. 문장 분리 (간단 rule 기반)
# 필요하면 kss 적용 가능
# ==========================================
def split_sentences(text):
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    return sents


# ==========================================
# 4. Extractive Top-3 문장 선택
# ==========================================
def extractive_top3(sentences):
    inputs = tok_ext(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = ext_model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )
        probs = torch.softmax(logits, dim=-1)[:, 1]  # extractive label=1

    probs = probs.cpu().tolist()

    # 문장 + 확률 묶기 → 상위 3개 선택
    ranked = sorted(
        [(sent, p) for sent, p in zip(sentences, probs)],
        key=lambda x: x[1],
        reverse=True
    )

    top3 = ranked[:3]
    return [s for s, p in top3]


# ==========================================
# 5. Abstractive Summary
# ==========================================
def abstractive_summary(text):
    enc = tok_abs(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = abs_model.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_length=256,
            min_length=20,
            num_beams=8,
            no_repeat_ngram_size=2
        )

    summary = tok_abs.decode(generated_ids[0], skip_special_tokens=True)
    return summary


# ==========================================
# 6. 통합 Summarizer
# ==========================================
def summarize_both(article_text):
    # 1) 문장 분리
    sents = split_sentences(article_text)

    # 2) Extractive top-3
    top3 = extractive_top3(sents)

    # 3) Abstractive 요약
    abs_summary = abstractive_summary(article_text)

    return top3, abs_summary


# ==========================================
# 7. 테스트
# ==========================================
if __name__ == "__main__":
    article = """
    ‘한국 재즈계의 대모’ 박성연이 23일 지병으로 타계했다. 
    77세. 국내 1세대 재즈 보컬리스트인 고인은 1978년 국내 첫 토종 재즈클럽 ‘야누스’를 만들고 40여년간 이끌어 왔다. 
    JNH뮤직은 23일 부고 소식을 전하며 “재즈 불모지였던 한국이 이제 여러 재즈 스타와 대규모 국제 페스티벌을 보유할 만큼 울창한 숲이 됐다”며 “야누스는 오늘의 숲이 있게 한 그 처음의 나무”라고 평했다.\n\n
    고인은 고등학교 졸업 후 미8군 무대 가수를 뽑는 오디션에 합격하며 무대에 섰다. 
    그는 생전 본지 인터뷰에서 “피아노 아르바이트를 하러 미8군에 갔다가 처음 재즈 피아노를 듣고 매력에 빠졌다”고 밝혔다. 
    숙명여대 작곡과에 진학했으나 집안 형편이 좋지 않아 클래식을 포기하고 재즈의 길로 접어들었다고. 
    그는 “화성ㆍ리듬ㆍ멜로디 등 음악의 3요소가 클래식과는 완전히 다른 신천지였다”고 표현했다.\n\n
    “실컷 재즈를 하고 싶어서” 한국인 최초로 만든 재즈 클럽은 수많은 음악인이 모여들며 한국 재즈의 산실이 됐다. 
    손님보다 연주자가 많을 정도로 재정난에 시달렸지만, 그는 신인 발굴과 연주자를 위한 무대를 포기하지 않았다. 
    신촌에서 대학로, 이화여대, 청담동을 거쳐 지금의 서초동에 정착하기까지 여러 차례 자리를 옮기면서도 연주비를 줄이지 않았다. \n\n
    2012년에는 운영 자금 마련을 위해 평생 소장해온 LP 음반을 전부 경매로 처분하기도 했다. 
    당시 사연이 전해지자 후배 뮤지션들이 그를 돕기 위해 헌정 공연 ‘땡큐, 박성연’을 열기도 했다. 
    말로ㆍ이부영ㆍ여진ㆍ써니킴ㆍ혜원ㆍ허소영ㆍ그린티 등 한국을 대표하는 재즈 보컬들이 총출동했다. 
    박성연은 “하루도 적자가 아닌 날이 없었다”면서도 “야누스를 연 것을 절대 후회하지 않는다. 
    다시 돌아가도 똑같이 할 것”이라고 밝혔다. \n \n
    오랫동안 앓던 신부전증이 악화되면서 2015년부터는 클럽 운영에서 손을 뗐다. 
    가장 아끼던 후배 재즈 보컬 말로가 이어받아 ‘디바 야누스’라는 이름으로 꾸려가고 있다. 
    박성연은 이후 서울 은평구의 한 요양병원에서 투석치료를 받으면서도 휠체어에 탄 채로 2018년 야누스 40주년 기념 특별 공연을 펼치고 지난해 9월 서울숲 재즈페스티벌 무대에 서는 등 강한 열정을 보여왔다.\n\n\n
    1985년 첫 앨범 ‘박성연과 재즈 앳 더 야누스 Vol.1’을 시작으로 1998년 ‘세상 밖에서’, 2013년 ‘박성연 위드 스트링스’ 등 틈틈이 앨범도 발표했다. 
    지난해 후배 가수 박효신과 함께 자동차 광고 모델로 출연해 화제가 되기도 했다. 
    광고 배경 음악으로 삽입된 자신의 곡 ‘바람이 부네요’를 박효신과 듀엣으로 다시 녹음한 것이 마지막 음악 기록이 됐다. \n\n
    남무성 재즈평론가는 “재즈를 미군을 상대로 하는 쇼에서 한국 대중을 위한 음악으로 옮겨온 야누스는 단순한 공간 이상의 의미를 지닌다. 
    음악공동체로서 꾸준히 교류할 수 있는 토대를 마련해줬다”며 “박성연은 독특한 음색과 창법으로 ‘한국의 빌리 홀리데이’라 불리기도 했다”고 밝혔다. 
    JNH뮤직 이주엽 대표는 “‘무대에 설 때만 비로소 존재 의미가 있다’ ‘외롭고 괴로울 때면 블루스를 더 잘 부르게 되겠구나’ 라던 모습이 특히 기억에 남는다”고 회고했다. 
    빈소는 서울대병원 장례식장, 발인은 25일이다. 장지는 경기 파주 가족묘다.
    """

    print(article)

    extractive, abstractive = summarize_both(article)

    print("\n=== Extractive Top-3 문장 ===")
    for s in extractive:
        print("•", s)

    print("\n=== Abstractive Summary ===")
    print(abstractive)
