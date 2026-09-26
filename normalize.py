"""
normalize.py

Text normalization for the Business Entity Resolution Challenge.

Handles the noise patterns actually observed in the data:
- Cross-script transliteration (Devanagari / Tamil / Kannada -> Latin)
- Legal-suffix variation (Ltd/Pvt/Corp/Inc/LLC/Private Limited)
- Domain-style names (foo.com -> foo)
- Address abbreviation variants (Rd/Road, St/Street, Ave/Avenue...)
- Missing values (NaN business_address is common -- ~3% of rows)

Install the one extra dependency this needs:
    pip install anyascii
"""

import re
import unicodedata
from dataclasses import dataclass

from anyascii import anyascii

# ---------------------------------------------------------------------------
# Legal / business suffixes (checked as whole tokens after normalization)
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "l l c", "llp",
    "l l p", "ltd", "limited", "pvt", "private", "co", "company", "pc",
    "plc", "lp", "l p", "enterprises", "group", "holdings",
}

# Longest-match-first so "private limited" doesn't get chopped into two
# separate single-word hits.
LEGAL_SUFFIX_PHRASES = sorted(
    {"private limited", "pvt ltd", "pvt limited", "private ltd"} | LEGAL_SUFFIXES,
    key=len,
    reverse=True,
)

NAME_ABBREVIATIONS = {
    r"\bcorp\b": "corporation",
    r"\bcorpn\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bco\b": "company",
    r"\bllp\b": "limited liability partnership",
    r"\bllc\b": "limited liability company",
    r"\b&\b": "and",
    r"\bdba\b": "",  # "doing business as" -- drop the marker, keep both names separately if needed
}

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bavenue\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bcir\b": "circle",
    r"\bapt\b": "unit",
    r"\bunit\b": "unit",
    r"\bfl\b": "floor",
    r"\bfloor\b": "floor",
    r"\bpo box\b": "pobox",
    r"\bp o box\b": "pobox",
    r"\bhwy\b": "highway",
    r"\bhno\b": "house number",
    r"\bh no\b": "house number",
    r"\bno\b": "number",
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_DOMAIN_RE = re.compile(r"^(www\.)?([a-z0-9\-]+)\.(com|net|org|co|in|biz)$")


def _transliterate(text: str) -> str:
    """Romanize non-Latin scripts (Devanagari, Tamil, Kannada, etc.)."""
    return anyascii(text)


def _strip_domain(text: str) -> str:
    m = _DOMAIN_RE.match(text.strip())
    if m:
        return m.group(2).replace("-", " ")
    return text


def _apply_replacements(text: str, replacements: dict) -> str:
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def _romanize_lower(text: str) -> str:
    """Transliterate + fold to ASCII lowercase, WITHOUT stripping punctuation.

    Domain-stripping needs the literal '.' still present, so this stops
    short of the punctuation-removal step that _base_clean used to do
    up front (that was the bug: punctuation was gone before the domain
    regex ever saw it).
    """
    text = _transliterate(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower().strip()


def _base_clean(text: str) -> str:
    text = _romanize_lower(text)
    text = _strip_domain(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


STOPWORDS = {"the", "and", "of", "a", "an"}


@dataclass
class NormalizedName:
    raw: str
    cleaned: str          # fully expanded, suffix intact -- use this for similarity features
    core_tokens: tuple     # sorted tuple of tokens with legal suffixes removed
    has_legal_suffix: bool
    sorted_key: str        # suffix-stripped tokens sorted + joined -- one blocking key
    prefix_key: str        # first 2 significant tokens, IN ORDER -- a second, more robust
                            # blocking key that survives garbled/phonetic-transliterated
                            # trailing suffixes (e.g. Tamil "LLP" -> "elelpi")


def normalize_name(name) -> NormalizedName:
    if name is None or (isinstance(name, float)):  # NaN
        name = ""
    raw = str(name)

    cleaned = _base_clean(raw)
    cleaned = _strip_domain(cleaned)
    cleaned = _apply_replacements(cleaned, NAME_ABBREVIATIONS)
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    tokens = cleaned.split()
    has_suffix = False
    core = []
    i = 0
    # strip known legal-suffix phrases (longest first) from anywhere in the token list
    joined = " " + " ".join(tokens) + " "
    for phrase in LEGAL_SUFFIX_PHRASES:
        pat = f" {phrase} "
        if pat in joined:
            has_suffix = True
            joined = joined.replace(pat, " ")
    core_tokens = tuple(sorted(t for t in joined.split() if t))

    sorted_key = "".join(core_tokens)

    significant = [t for t in tokens if t not in STOPWORDS]
    prefix_key = "".join(significant[:2])

    return NormalizedName(
        raw=raw,
        cleaned=cleaned,
        core_tokens=core_tokens,
        has_legal_suffix=has_suffix,
        sorted_key=sorted_key,
        prefix_key=prefix_key,
    )


@dataclass
class NormalizedAddress:
    raw: str
    cleaned: str
    tokens: tuple
    is_missing: bool


def normalize_address(addr) -> NormalizedAddress:
    is_missing = addr is None or (isinstance(addr, float))  # NaN
    if is_missing:
        return NormalizedAddress(raw="", cleaned="", tokens=tuple(), is_missing=True)

    raw = str(addr)
    cleaned = _base_clean(raw)
    cleaned = _apply_replacements(cleaned, ADDRESS_ABBREVIATIONS)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    tokens = tuple(sorted(cleaned.split()))

    return NormalizedAddress(raw=raw, cleaned=cleaned, tokens=tokens, is_missing=is_missing)


if __name__ == "__main__":
    # quick smoke test against a few rows straight out of the sample output
    examples = [
        ("Maure Williams Colombier Inc", "Maure Wilblims Colombier Inc"),
        ("Raj Investments LLP", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"),
        ("Payne Enterprises", "Payne Énterprises"),
        ("maurewilliamscolombier.com", "Maure Williams Inc Center"),
    ]
    for a, b in examples:
        na, nb = normalize_name(a), normalize_name(b)
        print(f"{a!r:45} -> {na.cleaned!r:35} key={na.sorted_key!r} prefix={na.prefix_key!r}")
        print(f"{b!r:45} -> {nb.cleaned!r:35} key={nb.sorted_key!r} prefix={nb.prefix_key!r}")
        print()
