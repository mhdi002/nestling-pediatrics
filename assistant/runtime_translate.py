"""Runtime FA↔EN for parent chat (not a third HF model — Google MT / glossary)."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import lru_cache
from typing import Literal

from assistant.settings import get_settings

log = logging.getLogger(__name__)

_HAS_FA = re.compile(r"[\u0600-\u06FF]")

# Small clinical glossary for robust offline fallback (both directions).
_FA_TO_EN = {
    "سلام": "hello",
    "وزن": "weight",
    "قد": "length",
    "دور سر": "head circumference",
    "پسر": "boy",
    "دختر": "girl",
    "هفته": "weeks",
    "ماه": "months",
    "کیلو": "kg",
    "سانتی‌متر": "cm",
    "سانتی متر": "cm",
    "نارس": "preterm",
    "طبیعی": "term",
    "آهن": "iron",
    "کمک": "help",
    "چطور": "how",
    "رشد": "growth",
    "نمودار": "chart",
}

_EN_TO_FA_PHRASES = {
    "Hi! I'm Nestling, your pediatric parent assistant. I can:": "سلام! من نستلینگ، دستیار والدین هستم. می‌توانم:",
    "Got it — I saved that for this chat": "متوجه شدم — برای این گفتگو ذخیره شد",
    "Chart overlay saved:": "نمودار ذخیره شد:",
    "I'm here to help with growth charts, ASQ/M-CHAT screening, and care questions.": "اینجام تا درباره رشد، غربالگری ASQ/M-CHAT و مراقبت کمک کنم.",
    "I still need these details before I can run growth tools:": "قبل از محاسبه رشد هنوز این موارد را لازم دارم:",
    "sex": "جنس",
    "measure": "شاخص",
    "weeks": "هفته",
    "value": "مقدار",
    "centile": "صدک",
    "status": "وضعیت",
    "preterm": "نارس",
    "term": "طبیعی (ترم)",
    "within_10_90": "در محدوده صدک ۱۰ تا ۹۰",
    "below_3rd_investigate": "زیر صدک ۳ — پیگیری با پزشک",
    "above_97th_investigate": "بالای صدک ۹۷ — پیگیری با پزشک",
    "outer_centile_monitor": "نزدیک حاشیه منحنی — پایش",
}


def has_persian(text: str) -> bool:
    return bool(_HAS_FA.search(text or ""))


def detect_lang(text: str) -> Literal["fa", "en"]:
    return "fa" if has_persian(text) else "en"


# One entry per direction — maxsize=1 made fa→en and en→fa evict each other.
@lru_cache(maxsize=2)
def _google(source: str, target: str):
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=source, target=target)
    except Exception as exc:
        log.info("Online translation unavailable, using the offline glossary: %s", exc)
        return None


@lru_cache(maxsize=1)
def _translate_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=max(1, get_settings().nestling_translate_workers),
        thread_name_prefix="nestling-mt",
    )


def _translate_bounded(translator, text: str) -> str | None:
    """
    Machine translation with a hard deadline.

    deep_translator issues an HTTP GET with no timeout, so a stalled upstream
    would otherwise hang the whole chat turn. On timeout the caller falls back
    to the offline glossary.
    """
    timeout = get_settings().nestling_translate_timeout
    future = _translate_pool().submit(translator.translate, text)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        future.cancel()
        log.warning("Translation timed out after %.1fs; using the offline glossary.", timeout)
    except Exception as exc:
        log.warning("Translation failed, using the offline glossary: %s", exc)
    return None


def _glossary_fa_to_en(text: str) -> str:
    out = text
    for fa, en in sorted(_FA_TO_EN.items(), key=lambda x: -len(x[0])):
        out = out.replace(fa, en)
    return out


def translate_fa_to_en(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if not has_persian(text):
        return text
    tr = _google("fa", "en")
    if tr is not None:
        translated = _translate_bounded(tr, text)
        if translated:
            return translated
    return _glossary_fa_to_en(text)


def translate_en_to_fa(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if has_persian(text):
        return text
    # Prefer phrase replacements for known templates, then MT for the rest
    patched = text
    for en, fa in sorted(_EN_TO_FA_PHRASES.items(), key=lambda x: -len(x[0])):
        patched = patched.replace(en, fa)
    if has_persian(patched) and patched != text and len(patched) > len(text) * 0.5:
        # Still may have English leftovers — try MT on full original for quality
        pass
    tr = _google("en", "fa")
    if tr is not None:
        translated = _translate_bounded(tr, text)
        if translated:
            return translated
    return patched if patched != text else text


def ensure_pure_lang(text: str, lang: str) -> str:
    """
    Force a reply into a single language.
    - en: strip Persian script tokens (no FA/EN mixing)
    - fa: if mostly English, translate whole string to FA
    """
    text = (text or "").strip()
    if not text:
        return text
    if lang == "en":
        # Remove Persian words/characters from English replies
        cleaned = re.sub(r"[\u0600-\u06FF]+", "", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned or text
    # fa
    if has_persian(text) and not re.search(r"[A-Za-z]{4,}", text):
        return text
    # Mixed or English-heavy → full EN→FA
    return translate_en_to_fa(text)


def translate_for_models(user_message: str, ui_lang: str | None = None) -> tuple[str, str]:
    """
    Returns (lang_detected_or_ui, english_message_for_agent).
    ui_lang forces reply language when set to 'fa' or 'en'.
    """
    detected = detect_lang(user_message)
    # If parent writes in Persian, keep replies Persian consistently even when UI lang
    # is still set to English.
    if detected == "fa":
        reply_lang = "fa"
    else:
        reply_lang = ui_lang if ui_lang in {"fa", "en"} else detected
    en = translate_fa_to_en(user_message) if detected == "fa" else user_message
    return reply_lang, en
