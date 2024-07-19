from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
from IPython.display import Audio
import numpy as np

model_name = "facebook/mms-tts-tur"
model = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_and_combine_audio(texts, combined_filename):
    sampling_rate = model.config.sampling_rate
    silence_duration = 2  
    silence = np.zeros(int(sampling_rate * silence_duration))
    audio_data = []

    for text, filename in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        sf.write(filename, output.squeeze().cpu().numpy(), sampling_rate)
        print(f"Generated {filename}")

        data, _ = sf.read(filename)
        audio_data.append(data)
        audio_data.append(silence)

    combined_audio = np.concatenate(audio_data)
    sf.write(combined_filename, combined_audio, sampling_rate)
    print(f"Generated combined audio file: {combined_filename}")
    display(Audio(combined_audio, rate=sampling_rate))

# Example 1: Telekomünikasyon
texts_telecom = [
    ("Merhaba, ABC Telekom'a hoş geldiniz. Ben Emre, size nasıl yardımcı olabilirim?", "emre_1.wav"),
    ("Merhaba Emre Bey. Telefon sinyalim evde çok zayıf, neredeyse hiç çekmiyor. Bu durumu nasıl düzeltebiliriz?", "customer_1.wav"),
    ("Anladım. Bu sorunu çözmek için birkaç adım kontrol edelim. Telefonunuzu yeniden başlattınız mı?", "emre_2.wav"),
    ("Evet, birkaç kez yeniden başlattım ama hala aynı.", "customer_2.wav"),
    ("Peki, sim kartınızı başka bir telefona takıp denediniz mi?", "emre_3.wav"),
    ("Evet, denedim ama sorun devam ediyor.", "customer_3.wav"),
    ("Anladım. Görünüşe göre bölgenizdeki baz istasyonunda bir sorun olabilir. Bu sorunu teknik ekibimize ileteceğim fakat ne zaman çözüleceği konusunda kesin bir bilgi veremiyorum. Yoğunluğa bağlı olarak biraz zaman alabilir. Başka bir konuda yardımcı olabilir miyim?.", "emre_4.wav"),
    ("Hayır, çok teşekkürler.", "customer_4.wav"),
    ("Rica ederim. İyi günler dilerim.", "emre_5.wav"),
]

generate_and_combine_audio(texts_telecom, "telecom_cozumlenmemis.wav")

# Example 2: E-ticaret
texts_ecommerce = [
    ("Merhaba, AY E-ticaret'e hoş geldiniz. Ben Berke, size nasıl yardımcı olabilirim?", "berke_1.wav"),
    ("Merhaba Berke. Geldığım ürün arızalı çıktı, iade etmek istiyorum ama sitedeki iade butonu çalışmıyor. Ne yapabilirim??", "customer_trade_1.wav"),
    ("Üzgünüm bu durum için hemen yardımcı oluyorum. Teknik bir sorun olabilir, hemen kontrol ediyorum... Maalesef şu anda sistemde bir sorun var ve çözülmesi biraz zaman alabilir. İade sürecinizi başlatmak için alternatif bir yöntem önerebilirim. E-posta yoluyla bize ulaşabilir misiniz?", "berke_2.wav"),
    ("E-posta ile ulaşmak zaman alabilir mi?", "customer_trade_2.wav"),
    ("Evet, maalesef e-posta üzerinden işlem süresi biraz daha uzun olabilir. Teknik ekibimiz sorunu çözene kadar bu yöntemle yardımcı olabilirim.", "berke_3.wav"),
    ("Tamam,peki başka bir yol yok mu?", "customer_trade_3.wav"),
    ("Şu an için başka bir yol yok, ancak size en kısa sürede geri dönüş yapacağız. Başka bir sorunuz var mı?", "berke_4.wav"),
    ("Hayır,teşekkürler.", "customer_trade_4.wav"),
    ("Rica ederim. İyi günler dilerim.", "berke_5.wav"),
]

generate_and_combine_audio(texts_ecommerce, "ecommerce_cozumlenmemis.wav")

# Example 3: Sağlık Ürünleri Satan Yer
texts_health = [
    ("Merhaba, ABC Sağlık Ürünleri'ne hoş geldiniz. Ben Mehmet, size nasıl yardımcı olabilirim?","mehmet_1.wav"),
    ("Merhaba Mehmet Bey. Aldığım tansiyon aleti doğru ölçmüyor gibi görünüyor. İade etmek istiyorum.", "customer_health_1.wav"),
    ("Üzgünüm bu durum için hemen yardımcı oluyorum. Sipariş numaranızı alabilir miyim?", "mehmet_2.wav"),
    ("Evet, sipariş numaram 987654321.", "customer_health_2.wav"),
    ("Teşekkür ederim. Maalesef ürününüzün iade süresi geçmiş durumda. Ancak teknik destek ekibimizle iletişime geçip sorununuzu çözmeye çalışabiliriz." , "mehmet_3.wav"),
    ("Ama ürünü daha yeni aldım ve ilk kullanımda sorun yaşadım.", "customer_health_3.wav"),
    ("Anladım, ancak sistemimizde iade süresi dolmuş görünüyor. Ürününüzü incelemeleri için teknik ekibimize yönlendirebiliriz ama iade yapamayabiliriz. Başka yardımcı olabileceğim bir konu var mı?", "mehmet_4.wav"),
    ("Hayır,şimdilik başka sorum yok.Teşekkürler", "customer_health_4.wav"),
    ("Rica ederim. İyi günler dilerim.", "mehmet_5.wav"),
]

generate_and_combine_audio(texts_health, "health_cozumlenmemis.wav")

# Example 4: Elektronik Mağazası
texts_electronics = [
    ("Merhaba, DEF Elektronik'e hoş geldiniz. Ben Deniz, size nasıl yardımcı olabilirim?","Deniz_1.wav"),
    ("Merhaba Deniz Bey. Geçen ay aldığım bilgisayar sürekli donuyor. İade etmek istiyorum ama hala bir geri dönüş almadım.", "customer_electronic_1.wav"),
    ("Üzgünüm bu durum için hemen kontrol ediyorum. Sipariş numaranızı alabilir miyim?", "Deniz_2.wav"),
    ("Evet, garanti numaram 543210987.", "customer_electronic_2.wav"),
    ("Teşekkür ederim. Görünüşe göre iade talebiniz işleme alınmış fakat henüz onaylanmamış. Teknik ekibimiz yoğun olduğu için biraz gecikme yaşanabilir. İade sürecinizi hızlandırmaya çalışacağım fakat kesin bir süre veremiyorum. Başka bir konuda yardımcı olabilir miyim?", "Deniz_3.wav"),
    ("Anladım, teşekkür ederim. Başka bir sorum yok.", "customer_electronic_3.wav"),
    ("Rica ederim. İyi günler dilerim.", "Deniz4_.wav"),
]

generate_and_combine_audio(texts_electronics, "electronics_cozumlenmemis.wav")

# Example 5: Bankacılık
texts_banking = [
    ("Merhaba, GHI Bankası'na hoş geldiniz. Ben Ahmet, size nasıl yardımcı olabilirim?","Ahmet_1.wav"),
    ("Merhaba Ahmet Bey. Kredi kartı başvurusu yaptım ama hala onaylanmadı. Durumunu öğrenebilir miyim?", "customer_bankacılık_1.wav"),
    ("Üzgünüm bu durum için hemen kontrol ediyorum. Lütfen T.C. kimlik numaranızı paylaşır mısınız?", "Ahmet_2.wav"),
    ("Evet, 12345678901.", "customer_bankacılık_2.wav"),
    ("Teşekkür ederim. Başvurunuz işleme alınmış fakat hala değerlendirme aşamasında görünüyor. Yoğunluk nedeniyle gecikme yaşanabilir. Tam olarak ne zaman sonuçlanacağı hakkında bilgi veremiyorum. Başka yardımcı olabileceğim bir konu var mı?", "Ahmet_3.wav"),
    ("Anladım, teşekkür ederim. Başka bir sorum yok.", "customer_bankacılık_3.wav"),
    ("Rica ederim. İyi günler dilerim.", "Ahmet_4.wav"),
]

generate_and_combine_audio(texts_banking, "banking_cozumlenmemis.wav")
