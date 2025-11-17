import re
from typing import Set, Dict, List

from bs4 import BeautifulSoup


def remove_urls_hashtags_emojis_mentions_emails(text: str) -> str:
    """
    Light cleaning: remove URLs (other patterns can be enabled if needed).
    Mirrors the notebook logic (hashtags/mentions/emails/emojis
    are currently left untouched).
    """
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # The patterns below are defined but not applied in the notebook;
    # we keep the same default behavior (no additional removal).
    # To enable them, simply uncomment.
    #
    # text = re.sub(r"#\w+", "", text)   # hashtags
    # text = re.sub(r"@\w+", "", text)   # mentions
    # text = re.sub(
    #     r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
    # )
    #
    # emoji_pattern = re.compile(
    #     "["                       # emoticons, pictos, etc.
    #     u"\U0001F600-\U0001F64F"
    #     u"\U0001F300-\U0001F5FF"
    #     u"\U0001F680-\U0001F6FF"
    #     u"\U0001F700-\U0001F77F"
    #     u"\U0001F780-\U0001F7FF"
    #     u"\U0001F800-\U0001F8FF"
    #     u"\U0001F900-\U0001F9FF"
    #     u"\U0001FA00-\U0001FA6F"
    #     u"\U0001FA70-\U0001FAFF"
    #     u"\U00002702-\U000027B0"
    #     u"\U000024C2-\U0001F251"
    #     "]+",
    #     flags=re.UNICODE,
    # )
    # text = emoji_pattern.sub(r"", text)

    return text


def transform_text(text: str) -> str:
    """
    Simple text normalization: spaces, newlines, inclusive writing markers.
    """
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # text = re.sub(r'[-–—‑‒−]', ' ', text)  # optionnel
    text = re.sub(r"\s+", " ", text)

    # Inclusive writing markers (from the notebook)
    text = text.replace("(e)", "")
    text = text.replace("(E)", "")
    text = text.replace(".e.", "")
    text = text.replace(".E.", "")

    return text.strip()


def get_text_from_tag(tag) -> str:
    """
    Concatenate all strings from a BeautifulSoup tag.
    """
    return "".join(tag.strings)


def extract_information(header: BeautifulSoup, selector: str) -> str:
    """
    Extract text from a Europresse header using a CSS selector.
    Mirrors the notebook logic:
      - if there are multiple elements, join them with '////'
      - replace ';' with ',' to avoid breaking CSV files.
    """
    if header is None:
        return "N/A"

    elements = header.select(selector)
    if elements:
        return "////".join(
            [get_text_from_tag(el).replace(";", ",") for el in elements]
        )
    else:
        return "N/A"


def normalize_journal(t: str, web_paper_differentiation: bool = False) -> str:
    """
    Normalize a journal / site name:
      - remove content inside parentheses,
      - cut after the first comma,
      - cut after ' - ...',
      - cut after ≥ 3 spaces,
      - optionally remove 'www.' and domain extensions (.fr, .com, ...),
      - strip and lowercase.
    """
    t = t.strip()

    # Remove everything inside parentheses
    t = re.sub(r"\(.*?\)", "", t)

    # Remove everything after the first comma
    t = re.sub(r",.*", "", t)

    # Remove everything after the first dash preceded by a space
    t = re.sub(r" -.*", "", t)

    # Remove everything after two (or more) consecutive spaces
    # (local editions: "le parisien  paris", "ouest-france  lannion", etc.)
    t = re.sub(r"\s{2,}.*", "", t)

    if not web_paper_differentiation:
        # Remove "www." prefixes
        t = re.sub(r"^www\.", "", t)

        # Remove domain extensions ('.fr', '.com', '.org', etc.)
        t = re.sub(r"(\.\w{2,3})+$", "", t)

    t = t.strip()
    return t.lower()


def extract_date_info(date_text: str, language: str = "fr") -> str:
    """
    Extract a human‑readable date from a header
    (e.g. \"mardi 18 octobre 2022\") for FR/EN.
    If nothing matches, return the original text.
    """
    if language == "fr":
        regex = (
            r"([1-3]?[0-9]\s("
            r"janvier|février|mars|avril|mai|juin|juillet|août|"
            r"septembre|octobre|novembre|décembre"
            r")\s20[0-2][0-9])"
        )
    elif language == "en":
        regex = (
            r"([1-3]?[0-9]\s("
            r"January|February|March|April|May|June|July|August|"
            r"September|October|November|December"
            r")\s20[0-2][0-9])"
        )
    else:
        # fallback simple
        regex = r"([1-3]?[0-9]\s\w+\s20[0-2][0-9])"

    date_text_clean = re.search(regex, date_text)
    return date_text_clean.group() if date_text_clean else date_text


def normalise_date(date_text: str) -> str | None:
    """
    Normalize a multilingual date string (FR/EN/ES/DE) into 'YYYY-MM-DD'.
    Mirrors the notebook logic.
    """
    month_dict = {
        # English
        "january": "01",
        "jan": "01",
        "february": "02",
        "feb": "02",
        "march": "03",
        "mar": "03",
        "april": "04",
        "apr": "04",
        "may": "05",
        "june": "06",
        "jun": "06",
        "july": "07",
        "jul": "07",
        "august": "08",
        "aug": "08",
        "september": "09",
        "sep": "09",
        "sept": "09",
        "october": "10",
        "oct": "10",
        "november": "11",
        "nov": "11",
        "december": "12",
        "dec": "12",
        # French
        "janvier": "01",
        "janv.": "01",
        "janv": "01",
        "février": "02",
        "févr.": "02",
        "févr": "02",
        "fevrier": "02",
        "fevr": "02",
        "mars": "03",
        "avril": "04",
        "avr.": "04",
        "avr": "04",
        "mai": "05",
        "juin": "06",
        "juillet": "07",
        "juil.": "07",
        "juil": "07",
        "août": "08",
        "aout": "08",
        "aôut": "08",
        "septembre": "09",
        "octobre": "10",
        "novembre": "11",
        "décembre": "12",
        "déc.": "12",
        "déc": "12",
        "decembre": "12",
        # Spanish
        "enero": "01",
        "ene.": "01",
        "ene": "01",
        "febrero": "02",
        "marzo": "03",
        "mar.": "03",
        "abril": "04",
        "abr.": "04",
        "abr": "04",
        "mayo": "05",
        "junio": "06",
        "julio": "07",
        "agosto": "08",
        "septiembre": "09",
        "setiembre": "09",
        # German
        "januar": "01",
        "februar": "02",
        "märz": "03",
        "maerz": "03",
        "marz": "03",
        "april": "04",
        "mai": "05",
        "juni": "06",
        "juli": "07",
        "august": "08",
        "september": "09",
        "oktober": "10",
        "november": "11",
        "dezember": "12",
    }

    date_text = (date_text or "").lower().strip()

    # Date patterns
    date_formats = [
        # 19 novembre 2021, 19 de noviembre de 2021, etc.
        r"(?:\b\w+\b,\s+)?(\d{1,2})(?:\.|\s+de|\s+)?\s*([\w\.\-]+)(?:\s+de)?\s+(\d{4})",
        # novembre 19, 2021
        r"(?:\b\w+\b,\s+)?([\w\.\-]+)\s+(\d{1,2}),?\s+(\d{4})",
        # 19/11/2021
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
        # 19-11-2021
        r"(\d{1,2})-(\d{1,2})-(\d{4})",
        # 2021-11-19
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        # 2021/11/19
        r"(\d{4})/(\d{1,2})/(\d{1,2})",
        # 19.11.2021
        r"(\d{1,2})\.(\d{1,2})\.(\d{4})",
    ]

    for pattern in date_formats:
        match = re.search(pattern, date_text, re.IGNORECASE)
        if not match:
            continue

        groups = match.groups()

        # Depending on the pattern, the order (day, month, year) varies:
        if pattern.startswith(r"(?:\b\w+\b,\s+)?(\d{1,2})"):
            day, month, year = groups
        elif pattern.startswith(r"(?:\b\w+\b,\s+)?([\w\.\-]+)"):
            month, day, year = groups
        elif pattern.startswith(r"(\d{1,2})/(\d{1,2})/"):
            first, second, year = groups
            if int(first) > 12:
                day, month = first, second
            elif int(second) > 12:
                month, day = first, second
            else:
                day, month = first, second
            day = day.zfill(2)
            month = month.zfill(2)
            return f"{year}-{month}-{day}"
        elif pattern.startswith(r"(\d{1,2})-(\d{1,2})-"):
            first, second, year = groups
            if int(first) > 12:
                day, month = first, second
            elif int(second) > 12:
                month, day = first, second
            else:
                day, month = first, second
            day = day.zfill(2)
            month = month.zfill(2)
            return f"{year}-{month}-{day}"
        elif pattern.startswith(r"(\d{4})-(\d{1,2})-(\d{1,2})"):
            year, month, day = groups
        elif pattern.startswith(r"(\d{4})/(\d{1,2})/(\d{1,2})"):
            year, month, day = groups
        elif pattern.startswith(r"(\d{1,2})\.(\d{1,2})\.(\d{4})"):
            day, month, year = groups
        else:
            continue

        month = month.lower().replace(".", "").strip()
        day = day.zfill(2)

        # Convert month to numeric form
        if month.isdigit():
            month_num = month.zfill(2)
        elif month in month_dict:
            month_num = month_dict[month]
        else:
            # Unrecognized month
            return None

        return f"{year}-{month_num}-{day}"

    # No recognized format
    return None


def standardize_name(name: str) -> str:
    """
    Sort the words of a name so comparison is order‑insensitive.
    """
    words = name.split()
    words.sort()
    return " ".join(words)


def split_names(s: str) -> list[str]:
    """
    Split a string containing several names into a list of names.
    Mirrors the heuristic logic from the notebook.
    """
    words = s.split()
    if len(words) == 4:
        return [" ".join(words[:2]), " ".join(words[2:])]
    elif len(words) == 6:
        return [
            " ".join(words[0:2]),
            " ".join(words[2:4]),
            " ".join(words[4:6]),
        ]
    elif len(words) == 8:
        return [
            " ".join(words[0:2]),
            " ".join(words[2:4]),
            " ".join(words[4:6]),
            " ".join(words[6:8]),
        ]
    return [s]


def extract_names(line: str) -> Set[str] | None:
    """
    Extract person names from a raw line (authors block).
    Mirrors the notebook pipeline (parentheses, domains, '////', etc.).
    """
    if not line or len(line) > 150:
        return None

    # Remove everything inside parentheses
    line = re.sub(r"\(.*?\)", "", line)

    # Ignore lines that contain domains or 'N/A'
    if re.search(r"(\.fr|\.com|n/a)", line, re.IGNORECASE):
        return None

    # Basic cleaning
    line = re.sub(r"\s?@\w+", "", line)
    line = line.replace(".", "")
    line = line.replace('"', "")
    line = line.replace("«", "")
    line = line.replace("»", "")
    line = re.sub(r"\s+", " ", line).strip()

    # If the line contains '////', keep what comes after
    if "////" in line:
        line = line.split("////", 1)[1].strip()

    line = line.replace(",", ", ")
    line = re.sub(r"\s+", " ", line).strip()

    names: list[str] = []
    if len(line.split()) > 3:
        parts = re.split(",| et", line)
        for part in parts:
            part = part.strip()
            if part:
                names.extend(split_names(part))
    else:
        line = line.replace(",", "")
        names.extend(split_names(line.strip()))

    cleaned = {n for n in names if n}
    return cleaned or None


def extract_metadata_from_soup(article_soup: BeautifulSoup, raw_text: str) -> Dict[str, str]:
    """
    Extract a metadata dictionary for a single Europresse article:
      - Title
      - Authors (normalized)
      - Raw_authors (raw line)
      - Journal_normalized
      - Date_normalized
      - Num_characters (length of raw text)
    """
    header = article_soup.header

    if header is None:
        return {
            "Title": "N/A",
            "Authors": "None",
            "Raw_authors": "N/A",
            "Journal_normalized": "N/A",
            "Date_normalized": "N/A",
            "Num_characters": str(len(raw_text or "")),
        }

    # Title
    title_text = extract_information(header, ".titreArticle p")

    # Journal
    journal_text = extract_information(header, ".rdp__DocPublicationName")
    journal_norm = normalize_journal(journal_text)

    # Date
    date_text = extract_information(header, ".DocHeader")
    date_text_clean = extract_date_info(date_text)
    normalized_date = normalise_date(date_text_clean)
    if normalized_date is not None:
        date_normalized = normalized_date.replace(";", "").replace("&", "")
    else:
        date_normalized = date_text_clean

    # Authors
    names_raw = extract_information(header, ".sm-margin-bottomNews")

    # Fallback if nothing is found with the historical selector
    if not names_raw or names_raw.strip().lower() in {"n/a", "na", "none"}:
        # Heuristic: look for a short block containing "par " in the header
        for tag in header.find_all(["p", "span", "div"]):
            candidate = get_text_from_tag(tag).strip()
            if not candidate:
                continue
            if len(candidate) > 150:
                continue
            if "par " in candidate.lower():
                names_raw = candidate
                break

    names = extract_names((names_raw or "").lower())
    if names:
        actual_names = [standardize_name(name) for name in names]
        # Remove subsets (names entirely contained in other names)
        filtered_names = [
            name
            for name in actual_names
            if not any(
                other_name != name
                and set(name.split()) < set(other_name.split())
                for other_name in actual_names
            )
        ]
        all_names = filtered_names
    else:
        all_names = None

    authors_str = "None" if all_names is None else ", ".join(map(str, all_names))

    return {
        "Title": title_text.replace(";", ""),
        "Authors": authors_str,
        "Raw_authors": names_raw,
        # Journal name as extracted from HTML, without normalization
        "Journal_original": journal_text.replace(";", ""),
        "Journal_normalized": journal_norm.replace(";", ""),
        "Date_normalized": date_normalized,
        "Num_characters": str(len(raw_text or "")),
    }


def extract_republication_sources(
    article_soup: BeautifulSoup,
    web_paper_differentiation: bool = False,
) -> List[str]:
    """
    Extract the list of other journals / sites where the article was published
    (republications) from Europresse HTML.

    Logic:
      - if a block with CSS class 'apd-wrapper' is present, it contains
        republication information;
      - inside this block, individual sources appear in elements with the
        class 'source-name-APD';
      - we extract their text, then apply `normalize_journal` to obtain
        normalized journal names;
      - we return a list of normalized, non‑empty names with duplicates removed.

    If no 'apd-wrapper' block is present, the function returns an empty list.
    """
    if article_soup is None:
        return []

    repub_names: List[str] = []

    # Find all republication containers
    wrappers = article_soup.find_all(class_="apd-wrapper")
    for wrapper in wrappers:
        # In each block, source names live in elements
        # (span, div, etc.) with class 'source-name-APD'
        source_tags = wrapper.find_all(class_="source-name-APD")
        for tag in source_tags:
            raw_name = get_text_from_tag(tag).strip()
            if not raw_name:
                continue
            norm_name = normalize_journal(
                raw_name,
                web_paper_differentiation=web_paper_differentiation,
            )
            if not norm_name:
                continue
            repub_names.append(norm_name)

    # Remove duplicates while preserving order
    seen: Set[str] = set()
    unique_repubs: List[str] = []
    for name in repub_names:
        if name in seen:
            continue
        seen.add(name)
        unique_repubs.append(name)

    return unique_repubs


