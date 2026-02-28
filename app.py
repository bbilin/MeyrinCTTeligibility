# app.py
# Meyrin CTT Eligibility Checker (Swiss click-tt)
# Implements: 50.4.5 / 50.4.6 / 50.4.7 + strict "no playing down" + strict "lock after 3 subs in higher team"
# Data sources:
# - clubTeams for team list + league links
# - clubLicenceMembersPage for player lookup within Meyrin
# - clubPools for "nominated team" (regular registration)
# - team pages (resolved from league/group pages) for appearances (Einsätze) per team (more reliable than clubPortraitTT)

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import streamlit as st

BASE = "https://www.click-tt.ch/cgi-bin/WebObjects/nuLigaTTCH.woa/wa"
MEYRIN_CLUB_ID = 33165

# Substitute markers (may vary; extend if you see other codes on team pages)
SUB_MARKERS = {"S", "E", "V"}

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}

session = requests.Session()
session.headers.update(
    {"User-Agent": "Meyrin-Eligibility-Checker/2.0 (+club tool; contact admin if issues)"}
)


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class TeamInfo:
    team_no: int
    name: str
    league_label: str          # label for selected phase
    league_url: str            # url for selected phase
    league_level_rank: int     # phase-agnostic level rank (computed from label)

@dataclass(frozen=True)
class PlayerPick:
    display_name: str
    portrait_url: str


@dataclass(frozen=True)
class PlayerKey:
    last: str
    first: str


# -----------------------------
# Utilities
# -----------------------------
def _abs_url(href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://www.click-tt.ch" + href
    return f"{BASE}/{href}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _parse_team_no_from_name(name: str) -> Optional[int]:
    # "Meyrin VI" -> 6 ; "Meyrin 6" -> 6
    m = re.search(r"\bMeyrin\s+([IVX]+|\d+)\b", name, flags=re.I)
    if not m:
        return None
    token = m.group(1).upper()
    if token.isdigit():
        return int(token)
    # roman numerals
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    return roman.get(token)


def league_rank(label: str) -> int:
    s = (label or "").lower()

    # strip phase/group noise
    s = re.sub(r"\bphase\s*[ab]\b", " ", s)
    s = re.sub(r"\bgr\.?\s*\d+\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # National leagues
    if "nla" in s:
        return 100
    if "nlb" in s:
        return 90
    if "nlc" in s:
        return 80

    # "Ligue 5" or "Liga 5"
    m = re.search(r"\b(?:ligue|liga)\s*(\d+)\b", s)
    if m:
        n = int(m.group(1))
        return 70 - n

    # "5ème ligue" / "5eme ligue" / "5. liga"
    m = re.search(r"\b(\d+)\s*(?:ème|eme|\.|)\s*(?:ligue|liga)\b", s)
    if m:
        n = int(m.group(1))
        return 70 - n

    return 0

#def league_rank(label: str) -> int:
#    """
#    Higher number == higher league.
#    Supports:
#      - NLA/NLB/NLC
#      - "Ligue 5", "Liga 5"
#      - "5ème ligue", "5eme ligue", "5. liga"
#    """
#    s = (label or "").lower()
#
#    # National leagues
#    if "nla" in s:
#        return 100
#    if "nlb" in s:
#        return 90
#    if "nlc" in s:
#        return 80
#
#    # "Ligue 5" or "Liga 5"
#    m = re.search(r"\b(?:ligue|liga)\s*(\d+)\b", s)
#    if m:
#        n = int(m.group(1))
#        return 70 - n  # Ligue 1 higher than Ligue 5
#
#    # "5ème ligue" / "5eme ligue" / "5. liga"
#    m = re.search(r"\b(\d+)\s*(?:ème|eme|\.|)\s*(?:ligue|liga)\b", s)
#    if m:
#        n = int(m.group(1))
#        return 70 - n
#
#    return 0


def roman_to_int(token: str) -> Optional[int]:
    token = token.upper()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    return roman.get(token)


# -----------------------------
# Click-tt scraping
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_meyrin_team_entries() -> Dict[int, List[Tuple[str, str]]]:
    """
    Returns dict: team_no -> list of (league_label, league_url) entries.
    This captures both Phase A and Phase B if present.
    """
    url = f"{BASE}/clubTeams?club={MEYRIN_CLUB_ID}"
    r = session.get(url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    entries: Dict[int, List[Tuple[str, str]]] = {}

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        team_label = _norm(tds[0].get_text(" ", strip=True))
        if not team_label:
            continue
        tl = team_label.lower()
        if not (tl.startswith("men") or tl.startswith("hommes") or tl.startswith("herren")):
            continue

        # team_no: Men/Hommes/Herren = 1; Hommes II = 2; etc.
        team_no = 1
        m = re.search(r"\b([ivx]+|\d+)\b$", team_label, flags=re.I)
        if m:
            tok = m.group(1)
            if tok.isdigit():
                team_no = int(tok)
            else:
                val = roman_to_int(tok)
                if val:
                    team_no = val

        a = tds[1].find("a")
        league_label = _norm(a.get_text(" ", strip=True)) if a else _norm(tds[1].get_text(" ", strip=True))
        league_url = _abs_url(a["href"]) if a and a.get("href") else ""

        if not league_label:
            continue

        entries.setdefault(team_no, []).append((league_label, league_url))

    return entries

def pick_phase_entry(options: List[Tuple[str, str]], phase: str) -> Tuple[str, str]:
    """
    phase: "A" or "B"
    Prefer label containing "Phase A"/"Phase B". If not found, fall back to first.
    """
    phase_key = f"phase {phase}".lower()
    for label, url in options:
        if phase_key in label.lower():
            return label, url
    return options[0]

#@st.cache_data(ttl=3600)
#def fetch_meyrin_teams() -> List[TeamInfo]:
#    """
#    Parses Meyrin clubTeams page where teams are listed as:
#      Men, Hommes II, Hommes III, ... (plus O40, Jeunesse, Cup, etc.)
#    Keeps only men's teams and returns a normalized name "Meyrin N".
#    """
#    url = f"{BASE}/clubTeams?club={MEYRIN_CLUB_ID}"
#    r = session.get(url, timeout=25)
#    r.raise_for_status()
#    soup = BeautifulSoup(r.text, "html.parser")
#
#    found: Dict[int, TeamInfo] = {}
#
#    for tr in soup.find_all("tr"):
#        tds = tr.find_all("td")
#        if len(tds) < 2:
#            continue
#
#        team_label = _norm(tds[0].get_text(" ", strip=True))
#        if not team_label:
#            continue
#
#        tl = team_label.lower()
#        if not (tl.startswith("men") or tl.startswith("hommes") or tl.startswith("herren")):
#            continue
#
#        # Exclude Cup-ish rows
#        league_text = _norm(tds[1].get_text(" ", strip=True)).lower()
#        if "hauptrunde" in league_text or "cup" in league_text:
#            continue
#
#        # team_no: Men/Hommes/Herren = 1; Hommes II = 2; etc.
#        team_no = 1
#        m = re.search(r"\b([ivx]+|\d+)\b$", team_label, flags=re.I)
#        if m:
#            tok = m.group(1)
#            if tok.isdigit():
#                team_no = int(tok)
#            else:
#                val = roman_to_int(tok)
#                if val:
#                    team_no = val
#
#        a = tds[1].find("a")
#        league_label = _norm(a.get_text(" ", strip=True)) if a else _norm(tds[1].get_text(" ", strip=True))
#        league_url = _abs_url(a["href"]) if a and a.get("href") else ""
#
#        candidate = TeamInfo(team_no=team_no, name=f"Meyrin {team_no}", league_label=league_label, league_url=league_url)
#
#        # Dedup: keep the higher league label if duplicates
#        if team_no not in found or league_rank(candidate.league_label) > league_rank(found[team_no].league_label):
#            found[team_no] = candidate
#
#    return [found[k] for k in sorted(found.keys())]


def search_player_in_meyrin_club(last: str, first: str) -> List[PlayerPick]:
    """
    Uses Meyrin 'Licenced players' page which contains direct links to player portraits.
    """
    url = f"{BASE}/clubLicenceMembersPage"
    params = {"club": str(MEYRIN_CLUB_ID), "preferredLanguage": "German"}
    r = session.get(url, params=params, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    last_l = last.strip().lower()
    first_l = first.strip().lower()

    picks: List[PlayerPick] = []
    for a in soup.find_all("a", href=True):
        name = _norm(a.get_text(" ", strip=True))
        if "," not in name:
            continue
        n_l = name.lower()
        if last_l and last_l not in n_l:
            continue
        if first_l and first_l not in n_l:
            continue
        href = a["href"]
        if "playerPortrait" in href and "person=" in href:
            picks.append(PlayerPick(display_name=name, portrait_url=_abs_url(href)))

    # de-dup by url
    seen = set()
    out = []
    for p in picks:
        if p.portrait_url in seen:
            continue
        seen.add(p.portrait_url)
        out.append(p)
    return out


def infer_player_name_from_portrait(portrait_url: str) -> PlayerKey:
    """
    Reads playerPortrait and returns (last, first) if possible.
    """
    r = session.get(portrait_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    text = soup.get_text("\n", strip=True)
    for line in text.split("\n")[:60]:
        if "," in line and len(line) < 80:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                return PlayerKey(last=parts[0], first=parts[1])
    return PlayerKey(last="", first="")


def fetch_regular_registration_nominated_team(
    season_name: str,
    contest_type_token: str,
    player: PlayerKey,
) -> Optional[int]:
    """
    Robustly parses clubPools and any linked groupPools pages to find "X.Y Last, First" entries.
    Returns X as nominated team number.
    """
    target_last = player.last.strip().lower()
    target_first = player.first.strip().lower()

    def row_has_player_text(txt: str) -> bool:
        t = _norm(txt).lower()
        return target_last in t and target_first in t

    def scan_html_for_rank_and_name(html: str) -> Optional[int]:
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.find_all("tr"):
            row_text = _norm(tr.get_text(" ", strip=True))
            if not row_has_player_text(row_text):
                continue
            m = re.search(r"\b(\d+)\.\d+\b", row_text)
            if m:
                return int(m.group(1))
        return None

    clubpools_url = f"{BASE}/clubPools"

    for display_typ in ("vorrunde", "rueckrunde"):
        params = {
            "club": str(MEYRIN_CLUB_ID),
            "contestType": contest_type_token,
            "displayTyp": display_typ,
            "preferredLanguage": "German",
            "seasonName": season_name,
        }
        r = session.get(clubpools_url, params=params, timeout=25)
        r.raise_for_status()
        html = r.text

        team_no = scan_html_for_rank_and_name(html)
        if team_no is not None:
            return team_no

        # Follow links to groupPools if clubPools is an index page
        soup = BeautifulSoup(html, "html.parser")
        group_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "groupPools" in href:
                group_links.append(_abs_url(href))
        group_links = list(dict.fromkeys(group_links))

        for link in group_links[:40]:
            try:
                r2 = session.get(link, timeout=25)
                r2.raise_for_status()
            except Exception:
                continue
            team_no = scan_html_for_rank_and_name(r2.text)
            if team_no is not None:
                return team_no

    return None


# -----------------------------
# Team-page based appearances (reliable)
# -----------------------------
def find_team_portrait_url(team: TeamInfo) -> Optional[str]:
    """
    Opens the league/group page (team.league_url) and finds the link to the specific team.
    Then returns the team portrait URL if available (teamPortrait / teamPortraitTT).
    """
    if not team.league_url:
        return None

    r = session.get(team.league_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    roman = ROMAN.get(team.team_no, str(team.team_no))
    labels = [
        f"Meyrin {roman}",
        f"Meyrin {team.team_no}",
        f"Hommes {roman}",
        f"Men {roman}",
        f"Herren {roman}",
    ]

    team_link = None
    for a in soup.find_all("a", href=True):
        txt = _norm(a.get_text(" ", strip=True))
        for lab in labels:
            if lab.lower() in txt.lower():
                team_link = _abs_url(a["href"])
                break
        if team_link:
            break

    if not team_link:
        return None

    # Open team page and find teamPortrait link (if any)
    r2 = session.get(team_link, timeout=25)
    r2.raise_for_status()
    soup2 = BeautifulSoup(r2.text, "html.parser")

    for a in soup2.find_all("a", href=True):
        href = a["href"]
        if "teamPortrait" in href or "teamPortraitTT" in href:
            return _abs_url(href)

    if "teamPortrait" in team_link or "teamPortraitTT" in team_link:
        return team_link

    # As a fallback, sometimes the team_link itself contains a roster table already
    return team_link


def fetch_team_apps_from_team_page(team_page_url: str, player: PlayerKey) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (apps, sub_apps) for that specific team.
    sub_apps is returned if the row is marked as substitute (S/E/V).
    """
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    target_last_first = f"{player.last}, {player.first}".lower()
    target_first_last = f"{player.first} {player.last}".lower()

    for tr in soup.find_all("tr"):
        row_text = _norm(tr.get_text(" ", strip=True)).lower()
        if target_last_first not in row_text and target_first_last not in row_text:
            continue

        tds = tr.find_all("td")
        if not tds:
            continue

        first_cell = _norm(tds[0].get_text(" ", strip=True)).upper()
        ints = [int(x) for x in re.findall(r"\b(\d+)\b", tr.get_text(" ", strip=True))]
        if not ints:
            continue

        apps = ints[-1]
        sub = apps if first_cell in SUB_MARKERS else None
        return apps, sub

    return None, None


def fetch_apps_via_team_pages(
    player: PlayerKey,
    nominated_team_no: Optional[int],
    target_team: TeamInfo,
    teams_by_no: Dict[int, TeamInfo],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    """
    Queries only the nominated team and target team pages.
    Returns:
      apps, sub_apps, debug_team_pages (team_no->url used)
    """
    apps: Dict[int, int] = {}
    sub_apps: Dict[int, int] = {}
    debug_team_pages: Dict[int, str] = {}

    team_nos: List[int] = []
    if nominated_team_no is not None and nominated_team_no in teams_by_no:
        team_nos.append(nominated_team_no)
    if target_team.team_no in teams_by_no and target_team.team_no not in team_nos:
        team_nos.append(target_team.team_no)

    for tno in team_nos:
        team = teams_by_no[tno]
        team_page = find_team_portrait_url(team)
        if not team_page:
            continue
        debug_team_pages[tno] = team_page
        a, s = fetch_team_apps_from_team_page(team_page, player)
        if a is not None:
            apps[tno] = a
        if s is not None:
            sub_apps[tno] = s

    return apps, sub_apps, debug_team_pages


# -----------------------------
# Rules engine (strict season behavior)
# -----------------------------
def check_eligibility(
    target_team: TeamInfo,
    teams_by_no: Dict[int, TeamInfo],
    nominated_team_no: Optional[int],
    apps: Dict[int, int],
    sub_apps: Dict[int, int],
) -> Tuple[bool, List[str]]:

    reasons: List[str] = []

    # league ranks for all Meyrin teams
    lr: Dict[int, int] = {tno: league_rank(info.league_label) for tno, info in teams_by_no.items()}

    def higher(a: int, b: int) -> bool:
        return lr.get(a, 0) > lr.get(b, 0)

    # Determine aligned teams in this season: appearances + nominated team
    played_teams = sorted([t for t, n in apps.items() if n > 0])
    if nominated_team_no is not None:
        played_teams = sorted(set(played_teams + [nominated_team_no]))

    # ---- Strict lock: 3 substitute appearances in a higher team -> locked to that team ----
    locked_team_no: Optional[int] = None
    if nominated_team_no is not None:
        base = nominated_team_no
        for tno, nsubs in sub_apps.items():
            if higher(tno, base) and nsubs >= 3:
                if locked_team_no is None or lr.get(tno, 0) > lr.get(locked_team_no, 0):
                    locked_team_no = tno

    if locked_team_no is not None and target_team.team_no != locked_team_no:
        return False, [
            "NOT ELIGIBLE (strict lock): player became titulaire in a higher team after 3 substitute matches and is locked to that team.",
            f"Locked team: {locked_team_no} ({teams_by_no[locked_team_no].league_label}). Target: {target_team.team_no} ({target_team.league_label}).",
        ]

    # ---- Strict no playing down: if aligned (nominated or played) in higher league -> cannot play in lower league ----
    target_rank = lr.get(target_team.team_no, 0)

    if nominated_team_no is not None and lr.get(nominated_team_no, 0) > target_rank:
        return False, [
            "NOT ELIGIBLE (strict no playing down): player is nominated in a higher league team for this season.",
            f"Nominated team: {nominated_team_no} ({teams_by_no[nominated_team_no].league_label}) > Target: {target_team.team_no} ({target_team.league_label}).",
        ]

    for tno in played_teams:
        if lr.get(tno, 0) > target_rank:
            return False, [
                "NOT ELIGIBLE (strict no playing down): player has already played in a higher league team this season.",
                f"Higher team played: {tno} ({teams_by_no[tno].league_label}) > Target: {target_team.team_no} ({target_team.league_label}).",
            ]

    # ---- 50.4.5: at most two teams, and must be in different leagues ----
    prospective = sorted(set(played_teams + [target_team.team_no]))
    if len(prospective) > 2:
        return False, [
            "NOT ELIGIBLE (50.4.5): a player may be aligned in at most two teams (season-wide, both phases).",
            f"Already aligned teams: {played_teams}; adding team {target_team.team_no} would make {prospective}.",
        ]

    if len(prospective) == 2:
        t1, t2 = prospective
        l1 = teams_by_no[t1].league_label if t1 in teams_by_no else "?"
        l2 = teams_by_no[t2].league_label if t2 in teams_by_no else "?"
        if league_rank(l1) == league_rank(l2):
            return False, [
                "NOT ELIGIBLE (50.4.5): if two teams, they must be in different leagues.",
                f"Teams would be {t1} ({l1}) and {t2} ({l2}) which are the same league level.",
            ]

    reasons.append("Passed 50.4.5 (max two teams, different leagues).")

    # Helper for substitute count in target team (based on what we found)
    def next_sub_count(team_no: int) -> int:
        return sub_apps.get(team_no, 0) + 1

    # ---- 50.4.7: non-nominated + first appearance in lowest team -> titulaire immediately ----
    lowest_team_no = min(lr.keys(), key=lambda k: lr[k]) if lr else target_team.team_no

    if nominated_team_no is None:
        if target_team.team_no == lowest_team_no and apps.get(lowest_team_no, 0) == 0:
            reasons.append(f"50.4.7: first alignment in lowest team {lowest_team_no} makes player titulaire there.")
            return True, ["ELIGIBLE"] + reasons

    # ---- 50.4.6 nominated case ----
    if nominated_team_no is not None:
        base = nominated_team_no
        if target_team.team_no == base:
            reasons.append(f"Player is nominated in team {base}.")
            return True, ["ELIGIBLE"] + reasons

        if higher(target_team.team_no, base):
            if not apps and not sub_apps:
                reasons.append("Warning: could not find season appearances on click-tt; assuming 0 prior substitute matches.")
            n = next_sub_count(target_team.team_no)
            if n <= 2:
                reasons.append(
                    f"50.4.6: nominated in lower team {base}; this is substitute appearance #{n} in higher team {target_team.team_no} (allowed up to 2)."
                )
                return True, ["ELIGIBLE"] + reasons
            else:
                reasons.append(
                    f"50.4.6: this is substitute appearance #{n} in higher team {target_team.team_no}; player becomes titulaire there and is then locked to that team."
                )
                return True, ["ELIGIBLE (but triggers titular lock)"] + reasons

        reasons.append("50.4.6: target team is not a higher league than nominated team; no extra restriction here.")
        return True, ["ELIGIBLE"] + reasons

    # ---- 50.4.6 non-nominated ----
    current_apps = apps.get(target_team.team_no, 0)
    n_after = current_apps + 1
    if n_after >= 3:
        reasons.append(f"50.4.6: this would be appearance #{n_after} in team {target_team.team_no}; player becomes titulaire of that team.")
        return True, ["ELIGIBLE (but becomes titulaire of the team)"] + reasons

    reasons.append(f"50.4.6: not nominated; this would be appearance #{n_after} in team {target_team.team_no}.")
    return True, ["ELIGIBLE"] + reasons


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Meyrin CTT – Eligibility Checker", layout="wide")
st.title("Meyrin CTT – Eligibility Checker (50.4.5 / 50.4.6 / 50.4.7)")

#teams = fetch_meyrin_teams()
#teams_by_no = {t.team_no: t for t in teams}

def infer_default_phase() -> str:
    # Simple heuristic: Aug-Dec => A, Jan-Jul => B
    import datetime
    m = datetime.date.today().month
    return "A" if 8 <= m <= 12 else "B"


team_entries = fetch_meyrin_team_entries()

with st.sidebar:
    phase = st.selectbox(
        "Phase (for dropdown)",
        options=["A", "B"],
        index=["A", "B"].index(infer_default_phase())
    )

teams: List[TeamInfo] = []
for team_no, opts in sorted(team_entries.items()):
    label, url = pick_phase_entry(opts, phase)
    teams.append(
        TeamInfo(
            team_no=team_no,
            name=f"Meyrin {team_no}",
            league_label=label,
            league_url=url,
            league_level_rank=league_rank(label),  # rank ignores phase/group (next section)
        )
    )

teams_by_no = {t.team_no: t for t in teams}
target = st.selectbox("Meyrin team", options=teams, format_func=lambda t: f"{t.name} — {t.league_label}")


with st.sidebar:
    st.header("Season / Series")
    season_name = st.text_input("Season (format 2025/26)", value="2025/26")
    contest_type = st.selectbox(
        "Series (contestType)",
        options=[
            ("Men (Herren)", "${herren}"),
            ("Women (Damen)", "${damen}"),
            ("O40", "${o40}"),
            ("Junior (Jeunesse)", "${jugend}"),
            ("U19", "${u19}"),
            ("U15", "${u15}"),
            ("U13", "${u13}"),
        ],
        format_func=lambda x: x[0],
    )[1]

    st.header("Target team")
    if teams:
        target = st.selectbox("Meyrin team", options=teams, format_func=lambda t: f"{t.name} — {t.league_label}")
    else:
        st.error("No teams found. Check click-tt connectivity.")
        st.stop()

    st.markdown("### Debug")
    if st.button("Test click-tt"):
        test_url = f"{BASE}/clubTeams?club={MEYRIN_CLUB_ID}"
        try:
            resp = session.get(test_url, timeout=20)
            st.write("Status:", resp.status_code)
            st.write("Length:", len(resp.text))
            st.text(resp.text[:300])
        except Exception as e:
            st.write("Request failed:", repr(e))

st.divider()
st.subheader("Player")

c1, c2 = st.columns(2)
with c1:
    last = st.text_input("Last name", value="")
with c2:
    first = st.text_input("First name", value="")

if st.button("Check eligibility"):
    if not last.strip() or not first.strip():
        st.error("Please enter both last name and first name.")
        st.stop()

    picks = search_player_in_meyrin_club(last.strip(), first.strip())
    if not picks:
        st.error("Player not found in Meyrin licensed players list.")
        st.stop()

    pick = picks[0]
    if len(picks) > 1:
        pick = st.radio(
            "Multiple matches found — pick one:",
            options=picks,
            format_func=lambda p: f"{p.display_name}",
        )

    player_key = infer_player_name_from_portrait(pick.portrait_url)
    if not player_key.last or not player_key.first:
        player_key = PlayerKey(last=last.strip(), first=first.strip())

    nominated_team_no = fetch_regular_registration_nominated_team(season_name, contest_type, player_key)

    # NEW: fetch appearances via nominated+target team pages
    apps, sub_apps, debug_team_pages = fetch_apps_via_team_pages(player_key, nominated_team_no, target, teams_by_no)

    ok, reasons = check_eligibility(
        target_team=target,
        teams_by_no=teams_by_no,
        nominated_team_no=nominated_team_no,
        apps=apps,
        sub_apps=sub_apps,
    )

    st.markdown("### Result")
    if ok:
        st.success(reasons[0])
    else:
        st.error(reasons[0])

    st.markdown("### Details")
    st.write(
        {
            "player": f"{player_key.last}, {player_key.first}",
            "portrait": pick.portrait_url,
            "nominated_team_no": nominated_team_no,
            "apps": apps,
            "sub_apps": sub_apps,
            "target_team": f"{target.name} ({target.league_label})",
            "Bilans found teams": sorted(apps.keys()),
            "team_pages_used": debug_team_pages,
        }
    )

    if not apps and not sub_apps:
        st.warning("Could not find season appearances on click-tt team pages; result may be incomplete.")

    st.markdown("### Rule reasoning")
    for r in reasons[1:]:
        st.write(f"- {r}")
