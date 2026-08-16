from indic_transliteration.sanscript import DEVANAGARI, IAST, transliterate
import unicodedata

samples = [
    "kl̥tyasya",
    "yádhyā̐",
    "n eṣṇana | ū́d dāhat | vyr̥̀k | vā́kṣam̐ yajñáis | váiṃ kl̥tyasya tát tvāt pátir yāt téktyāt | tā̀sās | tád dhyāmi | yádhyā̐ vārṣuṭhásyas | íti | agnís |",
]

SELECTIVE_PRE_RULES = {
    "ŕ": "r\u0301",
    "Ŕ": "R\u0301",
    # Candidate normalization for vocalic l represented as l + COMBINING RING BELOW.
    "l\u0325": "ḷ",
}

SELECTIVE_POST_RULES = {
    # Only replace a candrabindu combining mark if it survives transliteration.
    # A correctly handled source mark will already have been converted and is untouched.
    "\u0310": "\u0901",
}


def prepare_iast(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for source, target in SELECTIVE_PRE_RULES.items():
        text = text.replace(source, target)
    return text


def repair_residual_devanagari(text: str) -> str:
    for source, target in SELECTIVE_POST_RULES.items():
        text = text.replace(source, target)
    return unicodedata.normalize("NFC", text)


def unexpected_combining_marks(text: str):
    return [
        (
            ch,
            f"U+{ord(ch):04X}",
            unicodedata.name(ch, "UNKNOWN"),
        )
        for ch in text
        if ord(ch) in {0x0310, 0x0325}
    ]


for original in samples:
    prepared = prepare_iast(original)
    baseline_dev = transliterate(prepared, IAST, DEVANAGARI)
    repaired_dev = repair_residual_devanagari(baseline_dev)

    print("\n" + "=" * 90)
    print("ORIGINAL:")
    print(original)

    print("\nPREPARED:")
    print(prepared)

    print("\nDEVANAGARI BEFORE RESIDUAL REPAIR:")
    print(baseline_dev)

    print("\nRESIDUAL U+0310/U+0325 BEFORE:")
    residual_before = unexpected_combining_marks(baseline_dev)
    print(residual_before if residual_before else "None")

    print("\nDEVANAGARI AFTER RESIDUAL REPAIR:")
    print(repaired_dev)

    print("\nRESIDUAL U+0310/U+0325 AFTER:")
    residual_after = unexpected_combining_marks(repaired_dev)
    print(residual_after if residual_after else "None")
