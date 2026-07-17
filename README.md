# TensorFlow/Keras ile Türkçe Metin Sınıflandırma

Küçük bir Türkçe duygu veri kümesi üzerinde uçtan uca metin sınıflandırma hattı kuran TensorFlow/Keras projesi. Veri hazırlama, metin vektörleştirme, model eğitimi, değerlendirme ve tahmin adımlarını tek bir örnekte gösterir.

## Özellikler

- Türkçe örnek cümlelerden hazır eğitim veri kümesi
- `TextVectorization` ile kelime indeksleme
- Embedding tabanlı sinir ağı
- Eğitim/doğrulama ayrımı ve erken durdurma
- Komut satırından yeni cümle tahmini
- Eğitilmiş modeli `.keras` biçiminde kaydetme

## Kurulum

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Eğitim

```bash
python train.py --epochs 25
```

Eğitim sonrasında örnek tahminler ekrana yazılır ve model `artifacts/turkce_duygu.keras` konumuna kaydedilir.

## Mimari

```text
Türkçe metin → TextVectorization → Embedding → GlobalAveragePooling → Dense → Olumlu/Olumsuz
```

## Test

Veri katmanının testleri TensorFlow kurulmadan da çalışır:

```bash
python -m unittest discover -s tests -v
```

## Geliştirme fikirleri

- Daha büyük, lisansı uygun bir Türkçe veri kümesi kullanmak
- F1, kesinlik ve duyarlılık metriklerini eklemek
- Alt sözcük tokenizasyonu kullanmak
- Sonucu bir Türkçe BERT modeliyle karşılaştırmak

> Bu küçük veri kümesi yalnızca eğitim ve kod doğrulama amacı taşır; gerçek ürün kalitesini temsil etmez.

## Lisans

MIT
