# Müşteri Hizmetleri Çözüm Platformu

![Proje Görseli](link-to-your-image.png)

## Proje Tanımı

🎯 **Müşteri Hizmetleri Çözüm Platformu**, Teknofest 2024 Doğal Dil İşleme Serbest Kategori Yarışması için geliştirilmiştir. Bu proje, çağrı merkezi ve müşteri hizmetleri operasyonlarını iyileştirmek amacıyla doğal dil işleme (NLP) ve yapay zeka teknolojilerini kullanır. Platform, müşteri ve temsilci arasındaki sesli görüşmeleri metne dönüştürür, analiz eder ve anlamlı içgörüler sunar. Bu sayede operasyonel verimlilik artarken müşteri memnuniyeti de iyileştirilir.

## Özellikler

- **Sesli Metin Dönüşümü:** Çağrı merkezi ses dosyalarını otomatik olarak metne dönüştürme.
- **Duygu Analizi:** Görüşmelerdeki duygusal tonları analiz etme.
- **Anahtar Kelime ve Tema Çıkarımı:** Görüşmelerden öne çıkan kelimeler ve temaların tespiti.
- **Performans Değerlendirmesi:** Müşteri hizmetleri temsilcilerinin performansını değerlendirme.
- **Gerçek Zamanlı İzleme:** Canlı görüşmelerin anında analiz edilmesi ve raporlanması.

## Kullanılan Teknolojiler

### Ön Yüz
- HTML, CSS, JavaScript, Bootstrap

### Arka Yüz
- Python, FastAPI

### Yapay Zeka ve NLP Teknolojileri
- **Ses Tanıma:** Whisper
- **Dil Modeli:** Meta-Llama-3-70B

### Veritabanı
- MongoDB

## Dosyalar

- `main.py`: Uygulamanın ana dosyası.
- `templates/index.html`: Uygulamanın HTML şablon dosyası.
- `static/css/style.css`: Uygulamanın CSS stil dosyası.
- `static/js/script.js`: Uygulamanın JavaScript dosyası.
- `requirements.txt`: Projede kullanılan Python bağımlılıklarının listesi.

## Kurulum

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

## İletişim

LinkedIn: [Berke Sevim](https://www.linkedin.com/in/berke-sevim-1565161a2/)
LinkedIn: [Gözde Ceren Yıldız](https://www.linkedin.com/in/gözde-ceren-yıldız/)
LinkedIn: [Büşra Sulukan](https://www.linkedin.com/in/büşra-sulukan-82299a177/)


## 📄 Lisans

Bu proje Apache 2.0 Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.
