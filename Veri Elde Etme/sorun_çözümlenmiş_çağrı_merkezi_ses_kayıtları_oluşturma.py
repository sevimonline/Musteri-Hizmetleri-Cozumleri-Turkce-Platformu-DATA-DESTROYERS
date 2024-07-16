
"""#Örnek 1: Telekomünikasyon"""

from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
from IPython.display import Audio

# Model ve tokenizer'ı yükle
model_name = "facebook/mms-tts-tur"
model = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    ("Merhaba, ABC Telekom'a hoş geldiniz. Ben Emre, size nasıl yardımcı olabilirim?", "emre_1.wav"),
    ("Merhaba Emre Bey. Telefon faturam bu ay normalden çok yüksek geldi. Bu durumu inceleyebilir misiniz?", "customer_1.wav"),
    ("Tabii ki, hemen kontrol ediyorum. Hesabınıza giriş yapıyorum...Görünüşe göre bu ay yurtdışı aramalarınız oldukça fazla. Yurtdışı aramalarını kullanırken bir paketiniz var mıydı?", "emre_2.wav"),
    ("Hayır, genelde yurtdışı araması yapmıyorum ama bu ay birkaç kez aramak zorunda kaldım.", "customer_2.wav"),
    ("Anladım. Bu durum için size önerim, yurtdışı aramaları için uygun bir paket seçmeniz. Hemen bir indirim sağlayamam ama gelecekte daha düşük faturalar için paketlerden birini seçebilirsiniz. Başka yardımcı olabileceğim bir konu var mı?", "emre_3.wav"),
    ("Anladım, teşekkür ederim. Başka bir sorum yok.", "customer_3.wav"),
    ("Rica ederim. İyi günler dilerim.", "emre_4.wav"),
]

# Metinleri ses dosyalarına dönüştür ve sesleri dinle
for text, filename in texts:
    # Tokenizer ve model ile ses verisini oluştur
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    # Ses dosyasını kaydet
    sf.write(filename, output.squeeze().cpu().numpy(), model.config.sampling_rate)
    print(f"Generated {filename}")

    # Ses dosyasını dinle
    display(Audio(output.squeeze().cpu().numpy(), rate=model.config.sampling_rate))



"""## Örnek 2: E-ticaret"""

texts = [
    ("Merhaba, AY E-ticaret'e hoş geldiniz. Ben Berke, size nasıl yardımcı olabilirim?", "berke_1.wav"),
    ("Merhaba Berke. Geçen hafta bir ürün sipariş ettim ama hala elime ulaşmadı. Kargo takip numaramı kontrol edebilir misiniz?", "customer_trade_1.wav"),
    ("Tabii ki, hemen kontrol ediyorum. Lütfen kargo takip numaranızı paylaşır mısınız?", "berke_2.wav"),
    ("Evet, takip numaram 123456789..", "customer_trade_2.wav"),
    ("Teşekkür ederim. Sistemden kontrol ettiğimde kargonuzun dağıtım aşamasında olduğunu görüyorum. Ancak dağıtımda bir gecikme olmuş. Bu durumu kargo firmasıyla hemen paylaşıp hızlandırmalarını sağlayacağım. Üzgünüm, size bekletiyoruz.?", "berke_3.wav"),
    ("Tamam, teşekkür ederim. Ne zaman elime ulaşır peki?", "customer_trade_3.wav"),
    (" Muhtemelen yarın elinizde olacaktır. Size kargo firmasıyla iletişime geçtikten sonra bilgi vereceğim. Başka bir sorunuz var mıydı?", "berke_4.wav"),
    (" Hayır, bu kadar. Teşekkürler.", "customer_trade_4.wav"),
    ("  Rica ederim. İyi günler dilerim.", "berke_5.wav"),
]

for text, filename in texts:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    sf.write(filename, output.squeeze().cpu().numpy(), model.config.sampling_rate)
    print(f"Generated {filename}")

    display(Audio(output.squeeze().cpu().numpy(), rate=model.config.sampling_rate))





"""# Örnek 3: Sağlık Ürünleri Satan Yer"""

texts = [
    ("Merhaba, ABC Sağlık Ürünleri'ne hoş geldiniz. Ben Mehmet, size nasıl yardımcı olabilirim?","mehmet_1.wav"),
    ("Merhaba Mehmet Bey. Aldığım vitamin haplarının son kullanma tarihi geçmiş. Bu durumu nasıl düzeltebiliriz?", "customer_health_1.wav"),
    ("Üzgünüm bu durum için hemen yardımcı olacağım. Sipariş numaranızı alabilir miyim?", "mehmet_2.wav"),
    ("Evet, sipariş numaram 987654321.", "customer_health_2.wav"),
    ("Teşekkür ederim. Hemen kontrol ediyorum... Evet, bu siparişinizde bir hata olmuş. Size yeni bir paket göndereceğiz ve eski ürünlerinizi de geri alacağız. Kargo ücreti tamamen bize ait. Başka yardımcı olabileceğim bir konu var mı?", "mehmet_3.wav"),
    ("Çok teşekkür ederim, başka bir sorum yok.", "customer_health_3.wav"),
    (" Rica ederim. İyi günler dilerim.", "mehmet_4.wav"),

]

for text, filename in texts:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    sf.write(filename, output.squeeze().cpu().numpy(), model.config.sampling_rate)
    print(f"Generated {filename}")

    display(Audio(output.squeeze().cpu().numpy(), rate=model.config.sampling_rate))





"""#Örnek 4: Elektronik Mağazası"""

texts = [
    ("Merhaba, DEF Elektronik'e hoş geldiniz. Ben Deniz, size nasıl yardımcı olabilirim?","Deniz_1.wav"),
    ("Merhaba Deniz Bey. Geçen ay aldığım televizyon bozuldu. Garanti kapsamında mı değil mi öğrenebilir miyim?", "customer_electronic_1.wav"),
    ("Tabii ki, hemen kontrol ediyorum. Lütfen faturanızdaki garanti numarasını paylaşır mısınız?", "Deniz_2.wav"),
    ("Evet, garanti numaram 543210987.", "customer_electronic_2.wav"),
    ("Teşekkür ederim. Sistemde kontrol ettiğimde ürününüzün hala garanti kapsamında olduğunu görüyorum. Size en yakın servis noktasına yönlendirebilirim. Orada tamir veya değişim işlemleriniz yapılacaktır. Başka bir konuda yardımcı olabilir miyim?", "Deniz_3.wav"),
    ("Tamam, teşekkür ederim. Adresi alabilir miyim?", "customer_electronic_3.wav"),
    ("Tabii ki, adresi size e-posta olarak gönderiyorum. Başka bir sorunuz var mı?", "Deniz_4.wav"),
    (" Hayır, bu kadar. Teşekkürler.", "customer_electronic_4.wav"),
    (" Rica ederim. İyi günler dilerim.", "Deniz_5.wav"),
]

for text, filename in texts:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    sf.write(filename, output.squeeze().cpu().numpy(), model.config.sampling_rate)
    print(f"Generated {filename}")

    display(Audio(output.squeeze().cpu().numpy(), rate=model.config.sampling_rate))






"""#Örnek 5: Bankacılık"""

texts = [
    ("Merhaba, GHI Bankası'na hoş geldiniz. Ben Ahmet, size nasıl yardımcı olabilirim?","Ahmet_1.wav"),
    ("Merhaba Ahmet Bey. Kredi kartım kayboldu, ne yapmalıyım?", "customer_bankacılık_1.wav"),
    ("Üzgünüm bu durum için hemen yardımcı oluyorum. Hemen kartınızı iptal ediyorum ve yeni bir kart gönderilmesini sağlıyorum. Size kimlik doğrulaması için birkaç soru soracağım.", "Ahmet_2.wav"),
    ("Tabii, buyurun.", "customer_bankacılık_2.wav"),
    ("Doğum tarihinizi ve son işlem yaptığınız tarihi söyleyebilir misiniz?", "Ahmet_3.wav"),
    ("1 Ocak 1980, son işlem 15 Temmuz'da bir market alışverişiydi.", "customer_bankacılık_3.wav"),
    ("Teşekkür ederim. Kartınız iptal edildi ve yeni kartınız 3 iş günü içinde adresinize gönderilecek. Başka yardımcı olabileceğim bir konu var mı?", "Ahmet_4.wav"),
    (" Hayır, çok teşekkürler.", "customer_bankacılık_4.wav"),
    (" Rica ederim. İyi günler dilerim.", "Ahmet_5.wav"),
]

for text, filename in texts:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    sf.write(filename, output.squeeze().cpu().numpy(), model.config.sampling_rate)
    print(f"Generated {filename}")

    display(Audio(output.squeeze().cpu().numpy(), rate=model.config.sampling_rate))