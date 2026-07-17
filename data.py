"""Küçük ve dengeli Türkçe duygu analizi örnekleri."""

from __future__ import annotations

import random


SAMPLES: list[tuple[str, int]] = [
    ("Bu ürünü gerçekten çok beğendim", 1),
    ("Beklediğimden çok daha kaliteli çıktı", 1),
    ("Hızlı teslimat ve özenli paketleme", 1),
    ("Uygulamanın yeni sürümü harika çalışıyor", 1),
    ("Müşteri hizmetleri çok yardımcı oldu", 1),
    ("Fiyatına göre mükemmel bir seçim", 1),
    ("Kullanımı kolay ve tasarımı çok güzel", 1),
    ("Sonuçtan son derece memnun kaldım", 1),
    ("Kesinlikle tekrar satın alırım", 1),
    ("Performansı beni olumlu anlamda şaşırttı", 1),
    ("Kurulumu hızlı ve sorunsuz tamamladım", 1),
    ("Görüntü kalitesi çok net ve canlı", 1),
    ("Bu deneyim bütün beklentilerimi karşıladı", 1),
    ("Malzeme kalitesi oldukça başarılı", 1),
    ("Arkadaşlarıma gönül rahatlığıyla öneririm", 1),
    ("Ürün bozuk geldi ve hiç çalışmadı", 0),
    ("Beklediğim kaliteyi kesinlikle bulamadım", 0),
    ("Teslimat çok gecikti ve paket hasarlıydı", 0),
    ("Uygulama sürekli çöküyor ve donuyor", 0),
    ("Müşteri hizmetleri sorunumu çözmedi", 0),
    ("Verdiğim paraya hiç değmedi", 0),
    ("Kullanımı zor ve tasarımı çok kötü", 0),
    ("Sonuçtan hiç memnun kalmadım", 0),
    ("Bir daha kesinlikle satın almam", 0),
    ("Performansı oldukça yavaş ve yetersiz", 0),
    ("Kurulum sırasında sürekli hata aldım", 0),
    ("Görüntü kalitesi bulanık ve renksiz", 0),
    ("Bu deneyim tam bir hayal kırıklığıydı", 0),
    ("Malzeme kalitesi çok zayıf", 0),
    ("Kimseye tavsiye etmiyorum", 0),
]


def split_samples(validation_ratio: float = 0.2, seed: int = 42) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Veriyi sınıf dengesini koruyarak eğitim ve doğrulamaya ayırır."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 0 ile 1 arasında olmalıdır")
    rng = random.Random(seed)
    train: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    for label in (0, 1):
        group = [sample for sample in SAMPLES if sample[1] == label]
        rng.shuffle(group)
        cut = max(1, round(len(group) * validation_ratio))
        validation.extend(group[:cut])
        train.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation
