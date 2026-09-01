"""Generated emergency and ordinary parent turns, in English and Persian.

The six emergency phrasings and twelve ordinary questions already written into
tests/test_audience_scope.py are the ones the feature was measured against
when it was switched off. Reusing them would only prove that the new
classifier fits the same eighteen sentences.

So the turns here are composed instead: a subject form, a predicate, and a
tail, drawn from vocabularies the detector was not written against, across
several ages, conditions and care topics. The ordinary half deliberately
includes the hard cases -- questions that NAME a danger sign ("what are the
danger signs I should watch for", "what should I do if my baby has a fit"),
symptom reports that are not emergencies ("she has a mild fever since
yesterday"), and words that contain a red-flag word inside them ("my toddler
is limping", "he has a runny nose"). Those are what a keyword list gets wrong.

Seeded, so a failure names the seed that produced it and can be replayed.
"""

from __future__ import annotations

import random

# --- emergencies -------------------------------------------------------

SUBJECTS_EN = [
    "my baby",
    "my newborn",
    "my son",
    "my daughter",
    "my 3 month old",
    "my 6 week old",
    "the baby",
    "she",
    "he",
    "my little girl",
]

# One phrasing per red-flag class, several ways of saying each. None of these
# is copied from the corpus or from the previous test list.
EMERGENCY_PREDICATES_EN = [
    "is not breathing",
    "has stopped breathing",
    "cannot breathe",
    "is having difficulty breathing",
    "is having convulsions",
    "is having a fit right now",
    "just had a seizure",
    "is convulsing",
    "is unconscious",
    "is unresponsive",
    "has passed out",
    "went limp",
    "is floppy",
    "has gone blue",
    "has blue lips",
    "is turning blue",
    "is gasping",
    "is grunting",
    "is choking",
    "is not moving at all",
    "will not wake",
    "is bleeding heavily",
    "is not responding",
]

TAILS_EN = ["", " what do I do", " please help", " right now", " help!!", " I am scared"]

SUBJECTS_FA = [
    "بچم",
    "نوزادم",
    "پسرم",
    "دخترم",
    "کودکم",
    "بچه‌ام",
]

EMERGENCY_PREDICATES_FA = [
    "نفس نمی‌کشد",
    "نفس نمیکشه",
    "نفسش بند آمده",
    "تشنج کرده",
    "تشنج می‌کند",
    "بیهوش شده",
    "بیهوش است",
    "از حال رفته",
    "کبود شده",
    "لبهایش کبود است",
    "شل شده و تکان نمی‌خورد",
    "تکان نمی‌خورد",
    "بیدار نمی‌شود",
    "خونریزی شدید دارد",
    "خفه شده",
]

TAILS_FA = ["", " چیکار کنم", " کمک", " الان", " خیلی می‌ترسم"]

# --- ordinary turns ----------------------------------------------------
#
# Composed the same way, plus a block of hand-written hard cases that a naive
# keyword match gets wrong.

ORDINARY_TOPICS_EN = [
    "what foods suit a {age} month old",
    "how often should I feed a {age} month old",
    "is {name} sleeping enough at {age} months",
    "when should {name} start solids",
    "how much milk does a {age} month old need",
    "when do babies start {milestone}",
    "should I be worried that {name} is not {milestone} yet",
    "my child has {condition}, what can I do at home",
    "how do I treat {condition} in a {age} month old",
    "is {condition} contagious",
    "what vitamins does a {age} month old need",
    "how many wet nappies a day is normal at {age} months",
    "when is {name} due the next vaccine",
    "is {name} gaining enough weight",
    "how do I know if {name} is teething",
    "what is a normal head circumference at {age} months",
    "can I give a {age} month old cow's milk",
    "how do I get {name} to nap in the cot",
]

ORDINARY_TOPICS_FA = [
    "برای بچه {age} ماهه چه غذایی خوب است",
    "کودک {age} ماهه چقدر باید بخوابد",
    "چه زمانی غذای کمکی را شروع کنم",
    "{condition_fa} را چطور درمان کنم",
    "بچه {age} ماهه چند بار باید شیر بخورد",
    "وزن بچه {age} ماهه چقدر باید باشد",
    "کی باید واکسن بعدی را بزنم",
    "چطور بفهمم دندان در می‌آورد",
    "آیا {condition_fa} مسری است",
    "برای رشد قد چه کار کنم",
]

NAMES = ["Arman", "Bita", "Kian", "Roya", "Nika", "Sina", "Yalda", "Omid", "مریم", "علی"]
MILESTONES = ["walking", "crawling", "talking", "sitting up", "babbling", "pointing"]
CONDITIONS = [
    "cradle cap",
    "colic",
    "reflux",
    "thrush",
    "impetigo",
    "constipation",
    "conjunctivitis",
    "roseola",
    "nappy rash",
    "eczema",
]
CONDITIONS_FA = ["کولیک", "یبوست", "زردی", "اگزما", "برفک", "سرماخوردگی"]
AGES = [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 24, 36]

# Ordinary turns that contain red-flag vocabulary on purpose. These are the
# ones the previous score-based trigger and any keyword list get wrong.
ORDINARY_HARD_EN = [
    "what are the danger signs I should watch for",
    "what should I do if my baby has a seizure",
    "when should I worry about my baby's breathing",
    "how do I know if my baby is breathing too fast",
    "is it normal for a newborn to breathe irregularly",
    "should I call an ambulance if she has a fit",
    "what does a febrile convulsion look like",
    "how would I tell if he was unconscious",
    "my toddler is limping after a fall",
    "he has a runny nose and a cough",
    "she has a mild fever since yesterday",
    "his gums are bleeding a little from teething",
    "her eyes are blue like her father's",
    "my baby is breathing loudly when he sleeps",
    "he moves a lot in his sleep",
    "she has a small cut on her finger that stopped bleeding",
    "when do babies stop waking at night",
    "how do I wake her for a feed",
    "my son is not eating much today",
    "she refuses the bottle sometimes",
    "her urine looks a bit yellow today",
    "my baby spits up after every feed",
    "is a blue blanket safe in the cot",
    "he did not sleep well last night",
]

ORDINARY_HARD_FA = [
    "علائم خطر در نوزاد چیست",
    "اگر بچه تشنج کند چه کار کنم",
    "کی باید نگران نفس کشیدن بچه باشم",
    "بچه‌ام کمی تب دارد چه کار کنم",
    "لثه‌اش موقع دندان درآوردن کمی خون می‌آید",
    "چطور بفهمم بچه بیهوش است",
    "بچه‌ام شب‌ها بیدار می‌شود چه کنم",
    "بچه‌ام امروز کم غذا خورد",
    "چشم‌هایش آبی است",
    "برای سرفه بچه چه کار کنم",
]


def emergencies(count: int = 240, *, seed: int = 0) -> list[str]:
    """Generated emergency reports, English and Persian."""
    rng = random.Random(seed)
    out: list[str] = []
    for subject in SUBJECTS_EN:
        for predicate in EMERGENCY_PREDICATES_EN:
            out.append(f"{subject} {predicate}{rng.choice(TAILS_EN)}")
    for subject in SUBJECTS_FA:
        for predicate in EMERGENCY_PREDICATES_FA:
            out.append(f"{subject} {predicate}{rng.choice(TAILS_FA)}")
    rng.shuffle(out)
    return out[:count] if count and count < len(out) else out


def ordinary(count: int = 240, *, seed: int = 0) -> list[str]:
    """Generated everyday parent questions, English and Persian."""
    rng = random.Random(seed + 1)
    out: list[str] = list(ORDINARY_HARD_EN) + list(ORDINARY_HARD_FA)
    for template in ORDINARY_TOPICS_EN:
        for _ in range(6):
            out.append(
                template.format(
                    age=rng.choice(AGES),
                    name=rng.choice(NAMES),
                    milestone=rng.choice(MILESTONES),
                    condition=rng.choice(CONDITIONS),
                )
            )
    for template in ORDINARY_TOPICS_FA:
        for _ in range(6):
            out.append(
                template.format(
                    age=rng.choice(AGES),
                    condition_fa=rng.choice(CONDITIONS_FA),
                )
            )
    rng.shuffle(out)
    return out[:count] if count and count < len(out) else out
