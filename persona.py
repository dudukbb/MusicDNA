# persona.py
from __future__ import annotations

import hashlib
import random


PERSONA_TEXT = {
    "Mutlu & Coşkulu": [
        "Sen müziği sadece dinlemiyorsun; enerji topluyorsun.",
        "Pozitif tınılar sende anında mod yükseltir.",
        "Neşeli parçalar seni hızla motive eden bir düğme gibi.",
    ],
    "Agresif & Dinamik": [
        "Müzikte güç ve sertlik seni canlı tutuyor.",
        "Enerji yükseldikçe sen daha net hissediyorsun.",
        "Sert ritimler sende bir ‘gaz’ etkisi yaratıyor.",
    ],
    "Hüzünlü & Melankolik": [
        "Müzik senin için yüzleşme alanı gibi; duygudan kaçmıyorsun.",
        "Derin ve duygusal parçalar sende daha uzun kalıyor.",
        "Herkes ritme kapılırken sen sözlere takılan taraftasın.",
    ],
    "Huzurlu & Sakin": [
        "Müzik sende gürültüyü azaltan bir filtre gibi çalışıyor.",
        "Sakin tınılar senin denge alanın.",
        "Akış ve dinginlik, zevkinin gizli anahtarı.",
    ],
    "Dans/Party": [
        "Ritmi yakaladığında ayakta durmak zor.",
        "Beat iyiyse gerisi teferruat.",
        "Playlist’in ‘kıpır kıpır’ çalışıyor.",
    ],
    "Enstrümantal/Odak": [
        "Sözlerden çok melodilerle bağ kuruyorsun.",
        "Müzik sende bazen ‘odak modu’ açıyor.",
        "Enstrümanlar konuşsun, sen dinlersin.",
    ],
    "Sözel/Rap": [
        "Sen ritim kadar mesajı da takip ediyorsun.",
        "Anlatısı olan parçalar seni daha çabuk yakalıyor.",
        "Söz ağırlığı sende bir ‘hikâye’ hissi bırakıyor.",
    ],
    "Akustik": [
        "Doğal tınılar sana daha gerçek geliyor.",
        "Akustik taraf sende ‘samimiyet’ arayışı gibi.",
        "Yapay olmayan sesler sende daha çok iz bırakıyor.",
    ],
}

ENDING_BY_MOOD = {
    "Mutlu & Coşkulu": [
        "Bu enerji seni kolay kolay aşağı çekmez.",
        "Pozitiflik senin temel frekansın.",
        "Mod yükselten tarafın oldukça baskın.",
    ],
    "Agresif & Dinamik": [
        "Güçlü hissettiğinde müzik seninle aynı yönde akıyor.",
        "Bu sertlik sende net bir duruş yaratıyor.",
    ],
    "Hüzünlü & Melankolik": [
        "Bu derinlik herkeste bulunmaz.",
        "Duygularla bu kadar yakın olmak güçlü bir taraf.",
    ],
    "Huzurlu & Sakin": [
        "Bu sakinlik senin doğal alanın.",
        "Denge arayışın müzikte çok net hissediliyor.",
    ],
}

GENERAL_ENDINGS = [
    "Bu müzik zevki sana özgü bir imza gibi.",
    "Müzik tercihlerin rastgele değil, oldukça tutarlı.",
    "Bu dinleme tarzı seni anlatmanın sessiz bir yolu.",
    "Playlist’inde bile bir karakter var.",
]


def seed_from_user_songs(user_song_list) -> int:
    key = "|".join([f"{s.get('name','')}-{s.get('artist','')}" for s in user_song_list]).lower()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def generate_dynamic_comment(cluster_label: str, seed: int | None = None, max_sentences: int = 4) -> str:
    """
    - Aynı şarkı listesi -> aynı yorum (seed sabit)
    - Farklı şarkı listesi -> farklı yorum (seed değişir)
    """
    rng = random.Random(seed)
    parts = [p.strip() for p in str(cluster_label).split("/")] if cluster_label else []

    sentences: list[str] = []
    for p in parts:
        if p in PERSONA_TEXT:
            sentences.append(rng.choice(PERSONA_TEXT[p]))

    rng.shuffle(sentences)
    sentences = sentences[: max(1, max_sentences - 1)]

    main_mood = parts[0] if parts else None
    if main_mood in ENDING_BY_MOOD:
        sentences.append(rng.choice(ENDING_BY_MOOD[main_mood]))
    else:
        sentences.append(rng.choice(GENERAL_ENDINGS))

    return "\n".join(sentences)

