from indic_transliteration.sanscript import DEVANAGARI, IAST, transliterate
import unicodedata

samples = [
    "asmāŕrkṣam̐s",
    "vā́ṣṭhākham",
    "agnī́mā",
    "nr̥ṇyāt",
]

SELECTIVE_CANONICAL_DECOMPOSITION = {
    "ŕ": "r\u0301",
    "Ŕ": "R\u0301",
}


def transliteration_input(text: str) -> str:
    # Keep standard IAST characters in NFC because indic-transliteration
    # recognizes precomposed forms such as ā, ī, ṣ, ṭ and ṇ correctly.
    text = unicodedata.normalize("NFC", text)

    # Decompose only the known unsupported precomposed R-with-acute.
    # This is Unicode-equivalent decomposition, not a linguistic correction.
    return "".join(
        SELECTIVE_CANONICAL_DECOMPOSITION.get(ch, ch)
        for ch in text
    )


for original in samples:
    prepared = transliteration_input(original)
    devanagari = transliterate(prepared, IAST, DEVANAGARI)

    print("\nORIGINAL :", original)
    print("PREPARED :", prepared)
    print("DEVANAGARI:", devanagari)

    print("Prepared codepoints:")
    for ch in prepared:
        if ch in {"ŕ", "Ŕ"} or unicodedata.combining(ch):
            print(
                f"  {ch!r} U+{ord(ch):04X}",
                unicodedata.name(ch, "UNKNOWN"),
            )
