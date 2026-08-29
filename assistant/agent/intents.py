"""Intent classification (regex + prior-slot continuation)."""
from __future__ import annotations

import re

from assistant.agent.slots import extract_growth_slots

MEASURE_EXPLAIN_RE = re.compile(
    r"\b(?:what(?:'s| is| do you mean(?: by)?)?)\s+(?:the\s+)?measur|"
    r"\bby the measur|"
    r"\bmeasur(?:e|es|ement)?\s+you mean|"
    r"\bmesure|"
    r"منظورت(?:ان)?\s*از\s*(?:اندازه|مقیاس|measure)|اندازه\s*یعنی",
    re.I,
)
SHOW_CHART_RE = re.compile(
    r"\b(?:show|see|open|display|plot|draw)\b.{0,40}\bchart\b|"
    r"\b(?:child(?:'s)?|baby(?:'s)?|my)\s+chart\b|"
    r"نمودار|چارت|چارتش|نشان\s*بده.*(?:نمودار|چارت)|(?:نمودار|چارت).{0,12}(?:نشون|نشان)",
    re.I,
)
BARE_SHOW_RE = re.compile(r"^\s*(?:show(?:\s+it)?|نمایش(?:\s*بده)?)\s*[!.]?\s*$", re.I)
AFFIRM_RE = re.compile(
    r"^\s*(?:so\s+)?(?:it'?s|its)\s+(?:ok(?:ay|ey)?|fine|alright|good)\b|"
    r"^\s*(?:ok(?:ay|ey)?|alright|thanks|thank you|got it)\s*[!.]?\s*$|"
    r"^\s*(?:پس\s*)?(?:خوبه|اوکی|باشه|ممنون)\s*[!.]?\s*$",
    re.I,
)
ANALYZE_GROWTH_RE = re.compile(
    r"\b(?:analy[sz]e|interpret|explain)\b(?:\s+\w+){0,4}\s*(?:that|this|it|chart|result|growth|number|centile)?|"
    r"^\s*analy[sz]e\s*[!.]?\s*$|"
    r"\bwhat does (?:that|this|it|the (?:chart|result|number)) mean\b|"
    r"\b(?:is|am i|are we).{0,30}(?:on|in)\s+(?:a\s+|the\s+)?(?:good\s+|right\s+|correct\s+)?track\b|"
    r"\b(?:good|right|correct)\s+track\b|"
    r"\bon\s+track\b|"
    r"\bis (?:he|she|my (?:baby|child|son|daughter)|the (?:baby|child))\s+"
    r"(?:ok|okay|okey|fine|normal|healthy|alright|good)\b|"
    r"\bis (?:my |the )?(?:baby|child|he|she|son|daughter).{0,30}"
    r"(?:ok|okay|okey|fine|normal|healthy|good|alright|growing well)\b|"
    r"\bhow is (?:my |the )?(?:baby|child|growth|weight)\b|"
    r"تحلیل|تفسیر|یعنی\s*چی|وضعیت\s*رشد|مسیر\s*خوب|روی\s*خط|"
    r"رشدش\s*خوبه|وزنش\s*خوبه|خوبه\s*\?|"
    r"حالش\s*خوبه|اوکی\s*هست",
    re.I,
)
TALK_WORRY_RE = re.compile(
    r"\b(?:talk(?:ing)?|speech|language)\b.{0,30}\b(?:worr|concern|delay|ability)|"
    r"\b(?:worr|concern).{0,30}\b(?:talk(?:ing)?|speech|language|ability)|"
    r"cant?\s+talk|can'?t\s+talk|chald|child\s+cant|"
    r"talk\s+well|talking\s+well|"
    r"نگران.{0,20}(?:حرف|گفتار|صحبت)|(?:حرف|گفتار).{0,20}نگران",
    re.I,
)
# Short care follow-ups after a medical/speech turn (dada/mama, is that good, yes she…).
MEDICAL_FOLLOWUP_RE = re.compile(
    r"\b(?:dada|mama|baba|papa|mommy|daddy|mum+a|mum+y)\b|"
    r"\b(?:she|he|they|(?:the\s+)?(?:baby|child|kid))\s+says\b|"
    r"\bsays\s+(?:dada|mama|baba|papa|hi|hello|bye)\b|"
    r"^\s*yes\b.{0,80}\b(?:she|he|they|says|dada|mama|good|ok|okay|so)\b|"
    r"\bis\s+that\s+(?:good|ok(?:ay|ey)?|fine|normal|alright|encouraging)\b|"
    r"\bis\s+(?:that|this|it)\s+(?:a\s+)?good\s+(?:sign|thing)\b|"
    r"^\s*(?:so|and)\s*\?+\s*$|"
    r"\bwhat\s+about\s+(?:that|this|it)\b|"
    r"\b(?:and|but)\s+(?:she|he)\s+(?:says|said|can|does)\b|"
    r"ماما|بابا|دادا|می\s*گه|میگه|خوبه\s*\?|درسته\s*\?",
    re.I,
)

# Non-care thread labels (growth/UI) — anything else with last_medical_query is care.
_NON_CARE_TOPICS = frozenset(
    {"growth", "growth_analysis", "help", "chat", "history", "reassure", "slot_update", ""}
)
_TOPIC_SLUG_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "she",
        "he",
        "her",
        "his",
        "him",
        "they",
        "them",
        "their",
        "my",
        "our",
        "your",
        "baby",
        "child",
        "kid",
        "son",
        "daughter",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "that",
        "this",
        "it",
        "its",
        "ok",
        "okay",
        "okey",
        "fine",
        "good",
        "what",
        "how",
        "when",
        "why",
        "about",
        "does",
        "did",
        "has",
        "have",
        "had",
        "i",
        "to",
        "for",
        "in",
        "on",
        "of",
        "and",
        "or",
        "with",
        "well",
        "cant",
        "can",
        "cannot",
        "should",
        "do",
        "does",
        "me",
        "tell",
        "please",
        "just",
        "really",
        "very",
        "also",
        "so",
        "yes",
        "yeah",
        "yep",
        "no",
        "not",
        "from",
        "into",
        "there",
        "here",
        "some",
        "any",
        "much",
        "more",
        "need",
        "needs",
        "want",
        "like",
        "get",
        "got",
        "چی",
        "چه",
        "که",
        "برای",
        "این",
        "اون",
        "بچه",
        "فرزند",
        "نوزاد",
    }
)


HELP_RE = re.compile(
    r"^\s*(hi|hello|hey|salam|سلام|درود)(?:\s*[!.]*)?\s*$|"
    r"^\s*(?:hi|hello|hey)[,!.\s]+(?:how can you help(?: me)?|what can you do)\s*\??\s*$|"
    r"^\s*(how can you help(?: me)?|what can you do|who are you)\s*\??\s*$|"
    r"^\s*(help(?: me)?|کمک(?:\s*کن)?)\s*\??\s*$",
    re.I,
)
HISTORY_RE = re.compile(
    r"\b(last|previous|history|remember|remind(?:\s+me)?|summary|"
    r"child profile|show (?:my )?(?:child|baby)(?: profile| data| info| growth| record)?|"
    r"who (?:is|did) I select|what do you know about (?:my )?(?:child|baby)|"
    r"child(?:'s)? (?:data|info|information|record|profile)|"
    r"my child(?:'s)? (?:last |growth |result|summary|profile|data)|"
    r"what (?:was|were) my|"
    r"what did we just look at|"
    r"what did we just check|"
    r"we just (?:plotted|measured|saved|checked))\b|"
    r"قبلی|تاریخچه|آخرین|یادت|یادآوری|پرونده(?:\s*فرزند)?|اطلاعات (?:فرزند|کودک)|"
    r"وضعیت فرزندم را نشان بده|پروفایل|پروفیل|نشون\s*میدی|نشان\s*می\s*دهی|"
    r"بچم(?:و| را)?\s*(?:نشون|نشان)|فرزندم(?:و| را)?\s*(?:نشون|نشان)",
    re.I,
)
CONCERN_RE = re.compile(
    r"\b(can'?t talk|cannot talk|doesn'?t talk|not talking|speech delay|late (?:to )?talk|"
    r"won'?t speak|no words|language delay|developmental (?:delay|concern)|"
    r"can'?t walk|cannot walk|doesn'?t walk|not walking|late (?:to )?walk|"
    r"won'?t walk|not (?:yet )?walking|still (?:not |can'?t )?walk|"
    r"worried|problem|wrong|abnormal|delay|"
    r"fever|rash|vomit|cough|cry(?:ing)? a lot|"
    r"scar|wound|cut|scrape|injury|bruise|burn|blister|lesion|redness|"
    r"what should i do)\b|"
    r"حرف\s*نمی\s*زن|حرف نمیزن|صحبت نمی|گفتار|تاخیر|نگران|مشکل|چی\s*کار|چه\s*کار|"
    r"چرا\s*.*(?:حرف|صحبت)|نمیتواند\s*حرف|کی\s*حرف|چه\s*موقع\s*حرف|کی\s*صحبت|"
    r"راه\s*نمی\s*ره|راه نمیره|نمیتواند\s*راه|کی\s*راه|"
    r"زخم|جراحت|خراش|کبودی|سوختگی|راش|جوش",
    re.I,
)
MEDICAL_RE = re.compile(
    r"\b(iron|sleep|vitamin|breast|feed|feeding|food|foods|eat|eaten|nutrition|vaccine|fever|colic|"
    r"teething|teeth|tooth|earache|constipation|diarrhea|diarrhoea|reflux|congestion|"
    r"complementary|weaning|sids|milestone|development|speech|talk(?:ing)?|language|"
    r"walk(?:ing|s|ed)?|crawl(?:ing|s|ed)?|cruise|cruising|stand(?:ing|s)?|sitting|"
    r"gross\s+motor|fine\s+motor|motor\s+skill|"
    r"scar|wound|cut|scrape|injury|bruise|burn|blister|lesion|redness|rash|eczema|"
    r"dada|mama|baba|papa|"
    r"آهن|خواب|شیر|رشد|تغذیه|غذا|خوراک|واکسن|تب|کولیک|دندان|"
    r"حرف|گفتار|صحبت|راه\s*رفتن|خزیدن|زخم|جراحت|خراش|کبودی|سوختگی|"
    r"چی\s*به(?:ش|ش)?\s*بدم\s*بخوره|چی\s*بدم\s*بخوره|چی\s*بخوره|"
    r"به(?:ش|ش)?\s*چی\s*بدم|چه\s*غذایی\s*بدم|چی\s*بخوره\s*که\s*رشد)\b|"
    r"غذا\s*باید\s*چی|بچم\s*غذا|تغذیه(?:\s*بچ|\s*کودک|\s*نوزاد)?|"
    r"چی\s*به(?:ش|ش)?\s*بدم|چه\s*چیزی\s*بخوره|"
    r"\btell me about\b|\bwhat about\b|\bhow about\b|\bexplain\b|\bwhy (?:is|does|can'?t)\b|"
    r"\bwhat should (?:be eaten|my (?:baby|child) eat|i do)\b|\bshould be eaten\b|"
    r"\b(?:she|he|they|(?:the\s+)?(?:baby|child))\s+says\b|"
    r"\bfoods?.{0,30}(?:year|month)\b|"
    r"مشکل چی|چی کار کنم|چه باید|کی\s*حرف|چه\s*موقع",
    re.I,
)

# Hard care domains — switching between these replaces last_medical_query (not soft follow).
_CARE_TOPIC_FAMILY_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "feeding",
        re.compile(
            r"\b(?:food|foods|feed(?:ing)?|eat(?:en|ing)?|nutrition|weaning|complementary|"
            r"formula|breast(?:feed(?:ing)?)?|solids?|milk\s+(?:oz|ounce)|iron)\b|"
            r"تغذیه|غذا|خوراک|بخوره|آهن",
            re.I,
        ),
    ),
    (
        "motor",
        re.compile(
            r"\b(?:walk(?:ing|s|ed)?|crawl(?:ing|s|ed)?|cruise|cruising|stand(?:ing|s)?|"
            r"sitting|gross\s+motor|fine\s+motor|motor\s+skill|pull(?:ing)?\s+to\s+stand)\b|"
            r"راه\s*رفتن|خزیدن",
            re.I,
        ),
    ),
    (
        "skin",
        re.compile(
            r"\b(?:scar|wound|cut|scrape|injury|bruise|burn|blister|lesion|redness|rash|eczema)\b|"
            r"زخم|جراحت|خراش|کبودی|سوختگی|راش|جوش",
            re.I,
        ),
    ),
    (
        "speech",
        re.compile(
            r"\b(?:talk(?:ing)?|speech|language|words?|babbl|dada|mama|baba|papa)\b|"
            r"حرف|گفتار|صحبت|ماما|بابا|دادا",
            re.I,
        ),
    ),
    (
        "sleep",
        re.compile(r"\b(?:sleep|nap|sids)\b|خواب", re.I),
    ),
    (
        "illness",
        re.compile(
            r"\b(?:fever|vomit|cough|colic|teething|teeth|earache|constipation|"
            r"diarrhea|diarrhoea|reflux|congestion|vaccine)\b|"
            r"تب|کولیک|دندان|واکسن",
            re.I,
        ),
    ),
)


def care_topic_family(text: str) -> str | None:
    """Return a coarse care domain for hard topic-switch detection, or None."""
    for name, rx in _CARE_TOPIC_FAMILY_RES:
        if rx.search(text or ""):
            return name
    return None
GROWTH_COMPUTE_RE = re.compile(
    r"\b(overlay|plot|percentile|z-?score|compute|calculate|check growth|"
    r"on the chart|growth chart|show(?:\s+me)?(?:\s+the)?\s+chart|the chart|نمودار)\b|"
    r"\b(?:show|see|open|display).{0,40}\bchart\b|"
    r"\b(?:child(?:'s)?|baby(?:'s)?)\s+chart\b|"
    r"\b(weight|length|height|head circumference|وزن|قد|دور\s*سر)\b.+\b\d+(?:\.\d+)?|"
    r"\b\d+(?:\.\d+)?\s*(?:kg|cm|کیلو(?:گرم)?)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:w|weeks?|هفته)\b.+\b\d+(?:\.\d+)?|"
    r"\b(?:age|pma|pna|سن)\s*[:=]?\s*\d+|"
    r"\bvalue\s*[:=]|"
    r"چارت|نمودار|چارتش",
    re.I,
)
SCREEN_RE = re.compile(r"\b(asq|m-?chat|autism screen|غربال|سنین و مراحل)\b", re.I)


def topic_slug_from_query(query: str) -> str:
    """Free-text topic label from first meaningful tokens (not a closed enum)."""
    tokens = re.findall(r"[a-zA-Z\u0600-\u06FF]+", (query or "").lower())
    meaningful = [t for t in tokens if t not in _TOPIC_SLUG_STOPWORDS and len(t) > 2][:4]
    return "-".join(meaningful) if meaningful else "medical"


def _looks_like_growth_ask(msg: str) -> bool:
    return bool(
        SHOW_CHART_RE.search(msg or "")
        or re.search(
            r"\b(?:chart|overlay|percentile|z-?score|growth\s+chart|centile)\b|نمودار|چارت",
            msg or "",
            re.I,
        )
    )


def _topic_looks_speech(text: str) -> bool:
    return bool(
        re.search(
            r"talk|speech|language|dada|mama|baba|papa|words?|babbl|حرف|گفتار|صحبت|ماما|بابا|دادا",
            text or "",
            re.I,
        )
    )


def _prior_medical_thread(prior_slots: dict | None) -> bool:
    """True when the previous turn left an active medical/speech care thread."""
    if not prior_slots:
        return False
    last = prior_slots.get("last_intents") or []
    if isinstance(last, str):
        last = [last]
    if set(last) & {"medical", "screening"}:
        return True
    # Free-text last_topic + stored query means an open care thread.
    topic = str(prior_slots.get("last_topic") or "").lower()
    if prior_slots.get("last_medical_query") and topic not in _NON_CARE_TOPICS:
        return True
    return False


def _has_new_care_content(msg: str) -> bool:
    """
    True when the user states a care concern (any topic), not only a soft probe.
    Used so 'she cant talk… is that okey?' is a new medical turn, not a soft follow-up.
    """
    if TALK_WORRY_RE.search(msg or "") or CONCERN_RE.search(msg or ""):
        return True
    # Remove soft-followup surface forms, then see if medical cues remain.
    stripped = MEDICAL_FOLLOWUP_RE.sub(" ", msg or "")
    stripped = re.sub(
        r"\b(?:yes|yeah|yep|yup|ok|okay|okey|so|right|and|but|also|"
        r"is\s+that|what\s+about\s+(?:that|this|it)|"
        r"how\s+long|how\s+often|how\s+much)\b",
        " ",
        stripped,
        flags=re.I,
    )
    return bool(MEDICAL_RE.search(stripped))


def _is_soft_followup(
    msg: str,
    *,
    followup_hit: bool,
    affirm: bool,
    analyze: bool,
    prior_query: str | None = None,
) -> bool:
    """Yes/no / is-that-good / so? clarifications without a new care concern."""
    if _has_new_care_content(msg):
        return False
    # Hard domain switch (feeding→motor/skin/speech/…) is never a soft follow-up.
    cur_fam = care_topic_family(msg)
    prior_fam = care_topic_family(prior_query or "")
    if cur_fam and prior_fam and cur_fam != prior_fam:
        return False
    if followup_hit:
        return True
    if affirm and not analyze:
        return True
    if analyze:
        return False
    if len(msg.split()) > 14:
        return False
    return bool(
        re.search(
            r"^\s*(?:yes|yeah|yep|yup|ok|okay|okey|so|right|and|but|also)\b|"
            r"\bis\s+that\b|\bwhat\s+about\s+(?:that|this|it)\b|"
            r"^\s*(?:how\s+long|how\s+often|how\s+much)\b|"
            r"^\s*so\s*\?",
            msg,
            re.I,
        )
    )


from assistant.settings import get_settings


def classify_intent(user_message: str, prior_slots: dict | None = None) -> set[str]:
    """Classify the *current* user turn only (never history)."""
    msg = (user_message or "").strip()
    intents: set[str] = set()
    if not msg:
        return {"help"}

    concern = bool(CONCERN_RE.search(msg)) or bool(TALK_WORRY_RE.search(msg))
    medical_hit = bool(MEDICAL_RE.search(msg)) or bool(TALK_WORRY_RE.search(msg))
    followup_hit = bool(MEDICAL_FOLLOWUP_RE.search(msg))
    prior_medical = _prior_medical_thread(prior_slots)
    prior_query = str((prior_slots or {}).get("last_medical_query") or "").strip()
    measure_q = bool(MEASURE_EXPLAIN_RE.search(msg))
    show_chart = bool(SHOW_CHART_RE.search(msg))
    bare_show = bool(BARE_SHOW_RE.search(msg))
    affirm = bool(AFFIRM_RE.search(msg))
    analyze = bool(ANALYZE_GROWTH_RE.search(msg))
    soft_followup = _is_soft_followup(
        msg,
        followup_hit=followup_hit,
        affirm=affirm,
        analyze=analyze,
        prior_query=prior_query,
    )
    # Hard switch: any new medical concern (not a soft follow-up) while a care
    # thread is open. No closed topic taxonomy required.
    new_medical_concern = bool(
        (medical_hit or concern)
        and not soft_followup
        and not show_chart
        and not bare_show
        and not GROWTH_COMPUTE_RE.search(msg)
        and not measure_q
        and not _looks_like_growth_ask(msg)
    )
    topic_switch = bool(prior_medical and new_medical_concern)

    # Pure greeting / capability question only
    if (
        HELP_RE.search(msg)
        and not concern
        and not medical_hit
        and not followup_hit
        and not GROWTH_COMPUTE_RE.search(msg)
        and not measure_q
        and not show_chart
        and not analyze
    ):
        intents.add("help")
        return intents

    # Soft follow-up continues whatever last_medical_query was (any care topic).
    if (
        prior_medical
        and not topic_switch
        and not show_chart
        and not bare_show
        and not GROWTH_COMPUTE_RE.search(msg)
        and not measure_q
        and soft_followup
    ):
        # Don't hijack a clear new growth-only analysis ask with no care cues
        if analyze and not followup_hit and not medical_hit and not concern:
            pass
        else:
            intents.add("medical")
            speech_ctx = prior_query + " " + msg
            if _topic_looks_speech(speech_ctx):
                intents.add("screening")
            return intents

    # Care / injury / nutrition questions win over growth-analysis heuristics
    # (e.g. scar/wound after a chart turn must never become "analyze that").
    if (concern or medical_hit or followup_hit) and not show_chart and not GROWTH_COMPUTE_RE.search(msg):
        # Bare "is that good?" without prior medical / speech cues stays non-medical
        # so growth analysis / chat can handle it.
        if followup_hit and not concern and not medical_hit and not prior_medical:
            if not _topic_looks_speech(msg) and not re.search(
                r"\b(?:dada|mama|baba|papa|says|scar|wound|iron|sleep|feed|food)\b",
                msg,
                re.I,
            ):
                pass
            else:
                intents.add("medical")
                if _topic_looks_speech(msg):
                    intents.add("screening")
                if intents.intersection({"medical", "screening"}):
                    return intents
        elif not re.search(
            r"\b(?:show|open|view)\s+my\s+child\b|child profile|پرونده|اطلاعات فرزند",
            msg,
            re.I,
        ):
            intents.add("medical")
            if _topic_looks_speech(msg):
                intents.add("screening")
            if intents.intersection({"medical", "screening"}):
                return intents

    if analyze:
        intents.add("growth_analysis")
        return intents

    if affirm and not concern and not show_chart and not GROWTH_COMPUTE_RE.search(msg):
        intents.add("reassure")
        return intents

    if measure_q or show_chart:
        intents.add("growth")

    if bare_show and prior_slots and (
        prior_slots.get("want_overlay")
        or prior_slots.get("value") is not None
        or prior_slots.get("child_id")
    ):
        intents.add("growth")

    # Mixed turns (measurement + care) that skipped the early medical return
    if (concern or medical_hit) and not show_chart:
        if not re.search(
            r"\b(?:show|open|view)\s+my\s+child\b|child profile|پرونده|اطلاعات فرزند",
            msg,
            re.I,
        ):
            intents.add("medical")
        if _topic_looks_speech(msg):
            intents.add("screening")

    if (
        HISTORY_RE.search(msg)
        and not concern
        and not GROWTH_COMPUTE_RE.search(msg)
        and not measure_q
        and not show_chart
        and not bare_show
    ):
        intents.add("history")

    if GROWTH_COMPUTE_RE.search(msg):
        intents.add("growth")
    if SCREEN_RE.search(msg):
        intents.add("screening")

    slots = extract_growth_slots(msg)
    # Continue a chart only when THIS turn adds real growth facts (not just boy/girl
    # from a sentence like «پسرم کی حرف میزنه»).
    growth_progress = any(k in slots for k in ("measure", "weeks", "age_months", "value"))
    if slots.get("want_overlay"):
        intents.add("growth")
        intents.discard("slot_update")
    elif (
        prior_slots
        and prior_slots.get("want_overlay")
        and growth_progress
        and not concern
        and "medical" not in intents
    ):
        intents.add("growth")
        intents.discard("slot_update")

    # Care / speech questions must never reuse leftover chart tools
    if ("medical" in intents or "screening" in intents or concern) and not (
        show_chart or bare_show or slots.get("want_overlay") or GROWTH_COMPUTE_RE.search(msg)
    ):
        intents.discard("growth")

    if "growth" in intents and (show_chart or bare_show or slots.get("want_overlay")):
        intents.discard("history")

    if "history" in intents and show_chart:
        intents.discard("history")

    if (
        slots
        and not intents.intersection(
            {"growth", "medical", "history", "screening", "help", "reassure", "growth_analysis"}
        )
        and not concern
        and not measure_q
        and len(msg.split()) <= 6
        and set(slots) - {"want_overlay", "chart_standard", "sex"}
    ):
        intents.add("slot_update")
    if not intents:
        # Nothing matched. The keyword lists can never enumerate every way a
        # parent describes a concern -- "her urine is yellow" and
        # "ادرارش زرده" matched nothing, so a real symptom got the generic
        # "I'm listening" menu instead of an answer, even though the corpus
        # has content on it. Anything long enough to be a real statement or
        # question is therefore sent to retrieval, which is equipped to judge
        # relevance (and, where enabled, to fall back to a web search). Short
        # utterances stay conversational so greetings and one-word replies do
        # not trigger a pointless search.
        # A bare clarification ("is that good", "so?") belongs to whatever
        # thread is already open -- routing it to retrieval would answer a
        # question the parent did not ask. Only genuinely new content falls
        # through to retrieval.
        # A bare clarification ("is that good", "so?") belongs to whatever
        # thread is already open; routing it to retrieval would answer a
        # question the parent did not ask. Detected with the existing
        # follow-up patterns rather than a care-keyword check: keyword lists
        # do not know every symptom, so "ادرارش زرده" would be misread as
        # having no new content and silently swallowed.
        prior_topic = str((prior_slots or {}).get("last_topic") or "").strip()
        if prior_topic and (MEDICAL_FOLLOWUP_RE.search(msg) or AFFIRM_RE.search(msg)):
            intents.add("chat")
            return intents

        _s = get_settings()
        # Length is measured in characters as well as words: Persian says in
        # two words ("ادرارش زرده") what English needs four for, so a
        # word-count-only threshold silently ignores real concerns in the
        # more compact language.
        if (
            len(msg.split()) >= _s.nestling_retrieval_fallback_min_words
            or len(msg.strip()) >= _s.nestling_retrieval_fallback_min_chars
        ):
            intents.add("medical")
        else:
            intents.add("chat")
    return intents

