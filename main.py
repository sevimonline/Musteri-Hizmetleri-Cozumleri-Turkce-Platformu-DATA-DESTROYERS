from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import logging
import replicate
from pydub import AudioSegment
from io import BytesIO
import whisper
import tempfile
import ssl
import replicate
from urllib.request import urlopen
import torch

ssl._create_default_https_context = ssl._create_unverified_context



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


REPLICATE_API_TOKEN = 'r8_CGgLCJgP860ZFoYeQKokSN5s0ZJqUEo2LHxFF'
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model", response_class=HTMLResponse)
def model_page(request: Request):
    return templates.TemplateResponse("model.html", {"request": request})

@app.get("/result_bot", response_class=HTMLResponse)
def result_bot_page(request: Request):
    return templates.TemplateResponse("result_bot.html", {"request": request})

def convert_to_wav(file: BytesIO, format: str):
    audio = AudioSegment.from_file(file, format=format)
    wav_io = BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io

def llama3_1_405b(prompt, temperature=0.5):
    output = replicate.run(
        "meta/meta-llama-3.1-405b-instruct",
        input={"prompt": prompt, "max_tokens": 2048, "temperature": temperature})
    return "".join(output)

def format_transcription(transcription):
    segments = transcription.split("Müşteri:")
    formatted_transcription = ""
    for i, segment in enumerate(segments):
        if i == 0:
            formatted_transcription += f"{segment.strip()}"
        else:
            sub_segments = segment.split("Müşteri Hizmetleri Temsilcisi:")
            formatted_transcription += f"\n\nMüşteri: {sub_segments[0].strip()}"
            if len(sub_segments) > 1:
                formatted_transcription += f"\nMüşteri Hizmetleri Temsilcisi: {sub_segments[1].strip()}"
    return formatted_transcription

def identify_representative(file_name):
    if "rep1" in file_name:
        return "rep1"
    elif "rep2" in file_name:
        return "rep2"
    elif "rep3" in file_name:
        return "rep3"
    elif "rep4" in file_name:
        return "rep4"
    else:
        return "unknown"

def clean_output(llm_output):
    if isinstance(llm_output, list):
        llm_output = '\n'.join(llm_output)
    cleaned_output = []
    for line in llm_output.split('\n'):
        stripped_line = line.strip()
        if stripped_line:
            cleaned_output.append(stripped_line)
    return cleaned_output

@app.post("/process_audio", response_class=HTMLResponse)
async def process_audio(request: Request, file: UploadFile = File(...)):
    try:
        file_name = file.filename
        representative = identify_representative(file_name)
        
        format = file_name.split('.')[-1]
        audio_content = await file.read()
        wav_file = convert_to_wav(BytesIO(audio_content), format)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            wav_file.seek(0)
            temp_wav_file.write(wav_file.read())
            temp_wav_path = temp_wav_file.name

        model = whisper.load_model("medium")
        result = model.transcribe(temp_wav_path)
        transcription = result["text"]

        prompt = f"""
        {transcription}

        Yukarıdaki müşteri hizmetleri diyalogundan aşağıdaki bilgileri çıkarın:

        1. Duygu Analizi (Sentiment): Olumlu - Olumsuz
        2. Sorunun Çözümü: Evet - Hayır
        3. Anahtar Kelimeler: 
        4. En Çok Tekrarlanan Kelimeler:
        5. Müşteri Hizmetleri Temsilcisinin Performansı ve Üslubu: 5 üzerinden puanlayın
        6. Müşterinin Performansı ve Üslubu: 5 üzerinden puanlayın
        7. Müşteri Hizmetleri Temsilcisinin Kullandığı Kelime Sayısı:
        8. Müşteri Hizmetleri Temsilcisinin Kullandığı Cümle Sayısı:
        9. Müşterinin Kullandığı Kelime Sayısı:
        10. Müşterinin Kullandığı Cümle Sayısı:
        11. Müşteri Memnuniyeti: Kısa bir değerlendirme yapın
        12. Müşteri Sorunu Tanımı: Kısa bir özet
        13. Temsilci Çözüm Teklifleri: Sunulan çözümleri belirtin
        14. Temsilcinin Empati Düzeyi: 5 üzerinden puanlayın
        15. Temsilcinin Profesyonellik Düzeyi: 5 üzerinden puanlayın
        16. Müşterinin Duygu Durumu: Genel duygu durumu (örneğin: sinirli, üzgün, memnun)
        17. Konuşmanın Tonu: Dilin tonu (örneğin: dostça, resmi, agresif)
        18. Çözüm Süresi: Sorunun çözümü için geçen süre veya sürenin yeterliliği
        19. Geri Dönüş Gerekliliği: Müşteriyle takip amaçlı iletişim gerekip gerekmediği
        20. Müşterinin Gelecekteki Davranışları: Olası davranışlar (örneğin: ürün/hizmet kullanmaya devam edecek mi)

        Lütfen yalnızca belirtilen bilgileri sağlayın ve başka açıklama eklemeyin.
        """

        llm_output = llama3_1_405b(prompt)
        cleaned_output = clean_output(llm_output)

        prompt2 = f"""
        '{transcription}'
        Diyalogları hatasız ve doğru bir şekilde ayırt etmek ve yeniden yazmak için aşağıdaki yönergeleri izleyin. Yalnızca belirtilen formatı kullanın ve ek açıklama veya başka metin eklemeyin:

        1.Konuşmacıları Doğru Belirleme: Her bir konuşmayı dikkatlice analiz edin ve kimin konuştuğunu net bir şekilde belirleyin. Müşteri ve müşteri hizmetleri temsilcisinin konuşmalarını kesin olarak ayırt edin.

        2.Doğru Yazım ve İmla: Konuşmaları yazarken doğru yazım ve imla kurallarını uygulayın. Konuşma metninde yanlış algılanabilecek kelimeleri kontrol edin ve gerektiğinde düzeltin.

        3.Format: Diyalogları aşağıda belirtilen formatta yeniden yazın:
        Müşteri Hizmetleri Temsilcisi: [Temsilcinin cümlesi]
        Müşteri: [Müşterinin cümlesi]

        Müşteri Hizmetleri Temsilcisi: X Şirketine hoş geldiniz. Ben ismim Mehmet, size nasıl yardımcı olabilirim?
        Müşteri: Merhaba Mehmet Bey. ...
        Müşteri Hizmetleri Temsilcisi: Sizlere Hitap edebilmem için isminizi öğrenebilir miyim ?
        Müşteri: Tabiki. İsmim Aylin.
        Müşteri Hizmetleri Temcilcisi: Aylin Hanım konuyla ilgili ...

        ...

        4.Konuşma Bütünlüğü: Her iki tarafın konuşmalarını ayrı tutun ve bir konuşmanın diğer konuşmacıya ait olduğunu varsaymayın.

        5.Yanlış Anlaşılmaları Düzeltme: Speech-to-text modelinden kaynaklanan yanlış anlamaları ve yazım hatalarını düzeltin.

        6.Ek Metin Eklememe: Yalnızca konuşmaları dahil edin; ek açıklamalar, başlıklar veya başka metinler eklemeyin.


        """

        llm_output_2 = llama3_1_405b(prompt2)
        formatted_transcription_2 = format_transcription(llm_output_2)
        
        return templates.TemplateResponse("result_bot.html", {
            "request": request,
            "transcription": formatted_transcription_2,
            "llm_output": cleaned_output,
            "representative": representative
        })

    except Exception as e:
        logger.error(f"General error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
