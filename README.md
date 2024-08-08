# 📞 Müşteri Hizmetleri Çözümleri Türkçe Platformu: CX AI
![image](https://github.com/user-attachments/assets/dff3e20d-0141-4e33-8035-f4933ed7c030)

## Projeye Giriş
Günümüzde müşteri hizmetleri operasyonlarının verimliliğini artırmak ve müşteri memnuniyetini iyileştirmek, işletmelerin rekabet avantajı sağlaması açısından kritik bir önem taşımaktadır. Bu bağlamda, projemiz  ile birlikte yapay zeka ve türkçe doğal dil işleme teknolojilerine sunulacak katkılar sayesinde, hedef müşterilerimiz olan işletmelerin müşteri hizmetleri süreçlerini optimize etmelerine yardımcı olmayı hedefleyen çalışmalar Teknofest 2024 Doğal Dil İşleme Serbest Kategori Yarışması kapsamında bu repository'de sunulmuştur.
## EKİBİMİZ 
![image](https://github.com/user-attachments/assets/54f26d71-e534-4c3b-8fba-fa2cd9687851)

**Ekip Liderimiz Gözde Ceren Yıldırım;** Web arayüzü için internette ücretsiz olarak sunulan web sitesi template'lerinden modelimize uygun olanı bularak tüm proje çalışmamıza entegresinin yapılmasından sorumludur.
**Ekip Üyemiz Büşra Sulukan;** Planlanan proje aşamalarından biri olan Finetuning adımı için türkçe verilerin ( sektör bazlı intentler,sektör bazlı yabancı intentlerin llm kullanarak türkçeye dönüştürülmesi, atasözleri ve deyimler ile açıklamalarının elde edilmesi) toplanması, proje kapsamında  hedef sektör analizlerinin yapılması, pazarlama araştırmasının yapılması, model için en başta internet ortamında ücretsiz elde edilemeyen sesli müşteri hizmetleri konuşma verilerinin elde edilmesi için gerekli sentetik verilerin oluşturulmasından sorumludur.
**Ekip Üyemiz Berke Sevim**; Ekip olarak kararlaştırılan llm modelini,Türkçe diline katkı sunmak üzere toplanan Türkçe veriler ile finetuning etmek için gerekli gereksinimlerin araştırılmasından, FastAPI ile proje llm modelin kurulmasından sorumludur.
## 🎯 Proje Tanımı
![image](https://github.com/user-attachments/assets/3b2d2651-7a69-4cd3-9b00-ed66f4a7893d)
**Müşteri Hizmetleri Çözümleri Türkçe Platformu: CX AI**, Teknofest 2024 Doğal Dil İşleme Serbest Kategori Yarışması için geliştirilen yenilikçi bir projedir. Projemiz,  çağrı merkezi ve müşteri hizmetleri operasyonlarını iyileştirmek amacıyla doğal dil işleme (NLP) ve yapay zeka teknolojilerini kullanmaktadır. Bu platform, müşteri ve temsilci arasındaki sesli görüşmeleri metne ddönüştürerek, bu metinler üzerinde duygu analizi, tema çıkarımı ve anahtar kelime analizleri gerçekleştirmektedir.Elde edilen analiz sonuçları, işletmelerin operasyonel verimliliklerini artırmalarına ve müşteri memnuniyetini iyileştirmelerine olanak tanımaktadır. Ayrıca, bu analizler sayesinde işletmeler, daha bilinçli kararlar alabilir ve müşteri ilişkilerini optimize edebilir.Böylece operasyonel verimlilik artarken, müşteri memnuniyeti de önemli ölçüde iyileştirilir.

## 🎯 Hedef Kitlemiz 
![image](https://github.com/user-attachments/assets/fd1055f4-4342-4626-a359-d6cd1220e2e0)
Türkiye’de müşteri hizmetleri sektöründe, önemli bir istihdam kaynağı olup, 140.000’den fazla müşteri temsilcisi ve destek yönetici kadrosuyla 160.000’den fazla çalışanı bulunmaktadır. Bankacılık, finans, sigorta, telekomünikasyon, kamu hizmetleri ve e-ticaret gibi sektörlerde yoğunlaşan bu pazar, 2023 yılında 41.7 milyar TL’ye ulaşmıştır. Projemiz, Türkiye’de hızla büyüyen bu sektörde operasyonel verimliliği artırmayı ve müşteri memnuniyetini iyileştirmeyi hedeflemektedir.

Özellikle anadilin Türkçe olmasından dolayı yurtiçi hizmet veren firmalar, en büyük hedef kitlemizi oluşturmaktadır. Türkçe diline özgü anlayış ve çözümleme yeteneklerimiz sayesinde, yerel pazarın ihtiyaçlarını daha iyi karşılayabilir ve rekabet avantajı elde edebiliriz.

## 🚀Projenin Sağladığı Çözümler
![image](https://github.com/user-attachments/assets/39be0005-c589-4480-a0f1-8c12147220e5)

**📜Sesli Verilerin Diyaloğa Dönüştürülmesi**
Çağrı merkezi ses kayıtlarının karşılıklı diyalog formatına dönüştürülmesi, işletmelerin müşteri hizmetleri süreçlerini daha iyi anlamalarına ve analiz etmelerine olanak tanır. Bu süreçte, sesli veriler FFmpeg teknolojisi kullanılarak dijital metinlere dönüştürülmektedir. **FFmpeg**, ses dosyalarının işlenmesi ve dönüştürülmesinde yüksek performans ve esneklik sağlayarak, büyük veri setlerinin hızlı ve etkili bir şekilde işlenmesini mümkün kılar. Bu teknoloji sayesinde sesli veriler, analiz edilebilir metinlere dönüştürülerek daha ayrıntılı ve hassas analizlerin yapılmasına olanak tanır. Ffmpeg teknolojisi kullanılarak ses dosyasının kişi bazlı metin formatına dönüştürülerek karşılıklı diyalogların metinde ve nihai arayüzümüzde gözükmesine olanak tanır.

**🤖Doğal Dil İşleme (NLP) Analizi**
Transkripte edilen metinler üzerinde türkçe doğal dil işleme (NLP) teknikleri kullanılarak anlamlı içgörüler elde edilmektedir. Bu analizler, müşteri ve temsilci arasındaki iletişimi derinlemesine inceleyerek, kritik temaları ve anahtar kelimeleri ortaya çıkarmaktadır.

**💬Duygu Durumu Analizi**
Müşteri duygu durumunu analiz ederek etkileşim kalitesini artırma hedeflenmektedir. Bu analizler sayesinde müşteri memnuniyeti ve temsilci performansı gibi önemli metrikler değerlendirilebilir, işletmelerin müşteri deneyimini iyileştirmesi sağlanabilir.

**📈Geri Bildirim**
Çağrı merkezi süreçlerini izleme ve geri bildirim sağlama mekanizması, işletmelerin hizmet kalitesini sürekli olarak izlemelerine ve geliştirmelerine yardımcı olur. Bu süreç, müşteri geri bildirimlerini sistematik olarak toplar ve analiz eder, böylece işletmelerin hızlı ve etkili iyileştirmeler yapmalarını mümkün kılar.Çağrı merkezi ve müşteri hizmetleri süreçlerini dijitalleştirerek ve analiz ederek işletmelere önemli avantajlar sağlar.

**📊Sonuç**
Projemizde, sesli verilerin diyaloğa dönüştürülmesinden başlayarak doğal dil işleme ve duygu analizi ile anlamlı içgörüler elde edilmekte ve bu veriler geri bildirim mekanizmaları ile desteklenmektedir. Bu çözümlerden en önemli olanı, işletmelerin müşteri memnuniyetini artırmalarına ve operasyonel verimliliklerini yükseltmelerine katkı sağlamaktır.


## SWOT Analizi: Müşteri Hizmetleri Çözümleri Türkçe Platformu

![image](https://github.com/user-attachments/assets/ad08ff5d-26be-48df-a989-c2fe8b2afdec)

### Güçlü Yönler

**Yüksek Teknoloji Kullanımı:** Projemizde en güncel yapay zeka ve doğal dil işleme teknolojileri kullanılmıştır. Bu sayede, müşteri hizmetleri operasyonları yüksek doğruluk ve hızla analiz edilebilmektedir.
**Verimlilik Artışı:** Sesli görüşmeleri metne dönüştürme ve analiz etme süreçleri otomatik hale getirilerek, operasyonel verimlilik artırılmaktadır.
**Çok Yönlülük:** Platformumuz, duygu analizi, tema çıkarımı ve anahtar kelime analizi gibi çok yönlü işlevler sunarak işletmelere kapsamlı bir analiz imkanı sağlar.
**Geniş Kullanıcı Kitlesi:** Projemiz, Türkiye’deki geniş müşteri hizmetleri sektörüne hitap etmektedir.
**Rekabet Üstünlüğü:** Yerel pazarın ihtiyaçlarına özgü çözümler sunarak, yabancı rakiplere karşı rekabet avantajı sağlamaktadır.

### Zayıf Yönler

**Yüksek Teknik Beceri Gereksinimi:** Projenin teknik altyapısı ve uygulama süreçleri yüksek teknik beceri gerektirmektedir.
**Maliyet:** Gelişmiş teknolojilerin kullanımı ve sürekli güncellenmesi maliyetli olabilir.
**Bağımlılıklar:** Kullanılan teknolojiler ve platformlar dış kaynaklara bağımlılık yaratabilir.
**Özelleştirme Zorlukları:** Farklı sektörlerin özel ihtiyaçlarına göre özelleştirme yapmak zaman ve kaynak gerektirebilir.

### Fırsatlar

**Pazar Talebi:** Türkiye’de hızla büyüyen müşteri hizmetleri sektöründe yüksek bir talep bulunmaktadır.
**Teknolojik Gelişmeler:** Yapay zeka ve doğal dil işleme alanındaki hızlı gelişmeler, projemizin sürekli olarak iyileştirilmesini sağlayabilir.
**Türkiye Pazarında Yerel Oyuncu Olmak:** Yerel dil ve kültüre uygun çözümler sunarak, Türkiye pazarında güçlü bir konum elde edilebilir.
**Ortaklık ve İşbirlikleri:** Diğer teknoloji ve hizmet sağlayıcıları ile yapılacak ortaklıklar, projenin yeteneklerini artırabilir.
**Anadilin Türkçe Olması:** Türkçe diline özgü anlayış ve çözümleme yeteneklerimiz, yerel pazarda önemli bir avantaj sağlar.

### Tehditler

**Rekabet:** Hem yerel hem de uluslararası rakiplerin varlığı, pazarda rekabeti artırmaktadır.
**Hızlı Değişen Teknoloji:** Teknolojik yeniliklerin hızla değişmesi, projenin sürekli güncellenmesini gerektirir.
**Veri Güvenliği:** Müşteri verilerinin güvenliği ve gizliliği, projenin başarısı için kritik öneme sahiptir.
**Yasal ve Düzenleyici Riskler:** Veri gizliliği ve güvenliği ile ilgili yasal düzenlemeler, projenin uygulanabilirliğini etkileyebilir.

## 🛠Uygulama Mimmarisi: Kullanılan Teknolojiler 
![image](https://github.com/user-attachments/assets/4bd31531-d481-4a78-9902-2fcb010e9b20)

Projemizde temel programlama dili olarak Python kullanılmıştır. Sesli görüşmeleri metne dönüştürmek için OpenAI Whisper(medium) teknolojisinden faydalanılmıştır. Metinler üzerinde duygu analizi, tema çıkarımı ve anahtar kelime analizleri gerçekleştirmek için Llama 3.1 405 B dil modeli kullanılmıştır. Ses dosyalarının gerçek zamanlı olarak izlenebilmesi ve analiz sonuçlarının raporlanabilmesi için FastAPI ve Replicate’in bulut altyapısı kullanılmıştır. Kullanıcı arayüzü, HTML, CSS ve JavaScript gibi internetten bulunan ücretsiz templatelerle oluşturularak, proje için oluşturulan main.py ana dosyası bu web sayfasına gömülmüştür.
### 🌐 Ön Yüz
- HTML, CSS, JavaScript, Bootstrap

### 🖥 Arka Yüz
- Python, FastAPI

### 🤖 Yapay Zeka ve NLP Teknolojileri
- **Ses Tanıma:** Whisper
- **Ses Dosyalarının Kişi Bazlı Parçalara Bölünmesi** : FFmpeg
- **Dil Modeli:** Meta-Llama-3.1-405 B


## Proje İlk Model Prototipi

Proje için ücretsiz türkçe müşteri hizmetleri ses veri seti bulunamadığı için ilk olarak hugginface'den "facebook/mms-tts-tur" modeli indirilerek, sektör bazlı basit türkçe müşteri temsilcisiyle konuşma 
 metni oluşturulup, bu metin modele sokulup ses dosyaları elde edilmiştir. Oluşturulan ses verilerine buradan ulaşabilirsiniz.

 Oluşturulan ilk sentetik veri seti ile model arayüzü ve çıktıları aşağıdaki şekildedir:

https://github.com/user-attachments/assets/e9a50d6d-4cd4-4d46-8705-ae7376764ab3


## Ücretsiz Versiyon Kullanmak Zorunda Olmasaydık Yapmayı Düşündüğümüz Mimari

![image](https://github.com/user-attachments/assets/e3dc872a-ca48-4b7f-9c39-66be3d03c960)



-------AÇIKLAMA KISMI ---------



![image](https://github.com/user-attachments/assets/6a79b549-477e-443f-b396-ab91e908ba82)
![image](https://github.com/user-attachments/assets/d3ede316-ebc4-41c9-b8c8-c36001fc5f57)
![image](https://github.com/user-attachments/assets/77f2f85b-915d-4316-b583-7dd6b0388457)
![image](https://github.com/user-attachments/assets/1231b200-996f-4696-bae5-0c57f03f7156)



## 📂 Dosyalar

- `main.py`: Uygulamanın ana dosyası.
- `templates/`: Uygulamanın HTML şablonlarının dosyası.
- `static/css/`: Uygulamanın CSS stillerinin dosyası.
- `static/js/`: Uygulamanın JavaScript dosyaları.
- `requirements.txt`: Projede kullanılan Python bağımlılıklarının listesi.

## 🔧 Kurulum

1. Projeyi klonlayın:

    ```bash
    git clone https://github.com/sevimonline/Musteri-Hizmetleri-Cozumleri-Turkce-Platformu-DATA-DESTROYERS.git
    ```

2. Proje dizinine gidin:

    ```bash
    cd Musteri-Hizmetleri-Cozumleri-Turkce-Platformu-DATA-DESTROYERS
    ```

3. Gerekli paketleri yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

4. Uygulamayı başlatın:

    ```bash
    uvicorn main:app --reload
    ```

5. Tarayıcınızda `http://localhost:8000` adresine gidin ve uygulamayı kullanmaya başlayın.

## 👥 İletişim

- LinkedIn: [Berke Sevim](https://www.linkedin.com/in/berke-sevim-1565161a2/)
- LinkedIn: [Gözde Ceren Yıldız](https://www.linkedin.com/in/gözde-ceren-yıldız/)
- LinkedIn: [Büşra Sulukan](https://www.linkedin.com/in/büşra-sulukan-82299a177/)

## 📄 Lisans

Bu proje Apache 2.0 Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.
