# Bitirme_Projesi

🚀 Projenin Çalıştırılması
1. Gerekli Python Kütüphaneleri
Aşağıdaki kütüphanelerin yüklü olduğundan emin olun:

pip install numpy pandas librosa scikit-learn keras tqdm

    İsteğe bağlı: TensorFlow backend için ayrıca TensorFlow kurulumu gerekebilir:

pip install tensorflow

2. Dataset Hazırlığı

Bu proje, MAESTRO v3.0.0 klasik müzik veri setini kullanmaktadır.

    Dataseti şu klasör yapısında yerleştirin:

datasett/
└── datasett/
    ├── maestro-v3.0.0.csv
    └── [yıl klasörleri içinde ses dosyaları (2004, 2005, ..., 2018)]

    Not: CSV dosyası ve ses dosyaları aynı kök klasörde olmalıdır.

3. Modelin Eğitilmesi

Aşağıdaki komutla kodu çalıştırabilirsiniz:

python main.py

    Varsayılan olarak main() fonksiyonu çalışacak ve:

    Özellik çıkarımı yapacak

    Veriyi ölçekleyecek

    Modeli eğitecek

    Test setinde doğruluk, classification report ve confusion matrix çıktıları verecek

    Sonuçları results/ klasörüne kaydedecektir.

4. Sonuç Dosyaları

Kod çalıştıktan sonra şu dosyalar oluşur:

    results/scaler.pkl → Test sırasında kullanılacak ölçekleme nesnesi

    results/label_encoder.pkl → Sınıf isimlerini içeren label encoder

    results/best_dense_model.keras → Eğitilmiş model

    results/results.txt → Test doğruluğu, classification report ve confusion matrix çıktıları

5. Donanım Notu

Bu proje düşük donanımlı cihazlarda geliştirilmiştir.
Eğitim süresi uzun olabilir. Küçük batch size (4) ile çalışacak şekilde optimize edilmiştir.
