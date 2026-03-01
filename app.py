
# app.py
# Meyrin CTT Eligibility Checker (Swiss click-tt)
#
# What this version does (per your latest requirements):
# - Dropdown shows teams for a selected Phase (A/B) (current phase for UI).
# - Keeps a season history for a player: ALL teams he appeared in, league label at time (phase),
#   and how many times he played in each team (per phase + total).
# - History is built primarily by crawling match details (authoritative) and optionally cross-checks
#   with a lightweight parse of the player portrait page (best-effort).
# - Applies rules:
#   * 50.4.5: max 2 teams; if 2, must be in different leagues (league-level only; ignores phase/group)
#   * 50.4.6 + 50.4.7: nominated/non-nominated alignment behavior (counts based on season history)
#   * STRICT (season-wide, both phases): no playing down (if aligned in higher league, cannot play lower)
#   * STRICT: if nominated in lower team and appears 3 times in a higher team -> becomes titulaire there
#            and is locked to that higher team for the season (both phases)
#
# Notes:
# - click-tt HTML varies; match-link discovery is intentionally broad but capped + cached.
# - This tool only uses public pages.

import re
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import requests
from bs4 import BeautifulSoup
import streamlit as st


BASE = "https://www.click-tt.ch/cgi-bin/WebObjects/nuLigaTTCH.woa/wa"
MEYRIN_CLUB_ID = 33165

# Substitute markers are not reliably present in match details; match crawl counts "appearances".
# For 50.4.6, appearances in a higher team (different from nominated team) are treated as "substitute alignments".
SUB_MARKERS = {"S", "E", "V"}  # kept for potential future use

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}

session = requests.Session()
session.headers.update(
    {"User-Agent": "Meyrin-Eligibility-Checker/3.0 (+club tool; public pages only)"}
)


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class TeamInfo:
    team_no: int
    name: str
    league_label: str     # phase-specific label for UI/history
    league_url: str       # phase-specific url for resolving matches
    league_level_rank: int  # phase-agnostic rank for comparisons (ignores group/phase)


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


def roman_to_int(token: str) -> Optional[int]:
    token = token.upper()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    return roman.get(token)


def infer_default_phase() -> str:
    # Simple heuristic: Aug-Dec => A, Jan-Jul => B
    m = datetime.date.today().month
    return "A" if 8 <= m <= 12 else "B"


def league_rank(label: str) -> int:
    """
    Higher number == higher league.
    Phase/group are ignored.
    Supports:
      - NLA/NLB/NLC
      - "Ligue 5", "Liga 5"
      - "5ème ligue", "5eme ligue", "5. liga"
    """
    s = (label or "").lower()

    # strip noise that doesn't affect level
    s = re.sub(r"\bphase\s*[ab]\b", " ", s)
    s = re.sub(r"\bgr\.?\s*\d+\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if "nla" in s:
        return 100
    if "nlb" in s:
        return 90
    if "nlc" in s:
        return 80

    m = re.search(r"\b(?:ligue|liga)\s*(\d+)\b", s)
    if m:
        n = int(m.group(1))
        return 70 - n  # Ligue 1 higher than Ligue 5

    m = re.search(r"\b(\d+)\s*(?:ème|eme|\.|)\s*(?:ligue|liga)\b", s)
    if m:
        n = int(m.group(1))
        return 70 - n

    return 0


def _find_links(soup: BeautifulSoup, include: List[str]) -> List[str]:
    urls: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(k in href for k in include):
            urls.append(_abs_url(href))
    return list(dict.fromkeys(urls))


# -----------------------------
# Click-tt scraping (teams + players)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_meyrin_team_entries() -> Dict[int, List[Tuple[str, str]]]:
    """
    Returns dict: team_no -> list of (league_label, league_url) entries.
    Captures both Phase A and Phase B entries if present on the clubTeams page.
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

        # Determine team number: Men/Hommes/Herren => 1; Hommes II => 2; etc.
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

        # Skip Cup-ish entries
        if "hauptrunde" in league_label.lower() or "cup" in league_label.lower():
            continue

        if league_label:
            entries.setdefault(team_no, []).append((league_label, league_url))

    return entries


def pick_phase_entry(options: List[Tuple[str, str]], phase: str) -> Tuple[str, str]:
    phase_key = f"phase {phase}".lower()
    for label, url in options:
        if phase_key in (label or "").lower():
            return label, url
    # fallback: if no explicit Phase label exists, take first
    return options[0]


def build_phase_team_map(team_entries: Dict[int, List[Tuple[str, str]]]) -> Dict[int, Dict[str, TeamInfo]]:
    """
    Returns: team_no -> {"A": TeamInfo, "B": TeamInfo} (if present).
    If only one entry exists, we expose it as both phases (best-effort).
    """
    out: Dict[int, Dict[str, TeamInfo]] = {}
    for team_no, opts in team_entries.items():
        phase_map: Dict[str, TeamInfo] = {}

        # detect explicit A/B entries if possible
        a_opts = [x for x in opts if "phase a" in (x[0] or "").lower()]
        b_opts = [x for x in opts if "phase b" in (x[0] or "").lower()]

        if a_opts:
            label, url = a_opts[0]
            phase_map["A"] = TeamInfo(team_no, f"Meyrin {team_no}", label, url, league_rank(label))
        if b_opts:
            label, url = b_opts[0]
            phase_map["B"] = TeamInfo(team_no, f"Meyrin {team_no}", label, url, league_rank(label))

        # fallback if not labeled
        if "A" not in phase_map and opts:
            label, url = opts[0]
            phase_map["A"] = TeamInfo(team_no, f"Meyrin {team_no}", label, url, league_rank(label))
        if "B" not in phase_map and opts:
            label, url = opts[-1]
            phase_map["B"] = TeamInfo(team_no, f"Meyrin {team_no}", label, url, league_rank(label))

        out[team_no] = phase_map

    return out


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

    seen = set()
    out = []
    for p in picks:
        if p.portrait_url in seen:
            continue
        seen.add(p.portrait_url)
        out.append(p)
    return out


def infer_player_name_from_portrait(portrait_url: str) -> PlayerKey:
    r = session.get(portrait_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    text = soup.get_text("\n", strip=True)
    for line in text.split("\n")[:80]:
        if "," in line and len(line) < 90:
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
    Finds nominated team number from clubPools / groupPools by scanning rows containing:
      'X.Y ... Last, First'
    Returns X.
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

        soup = BeautifulSoup(html, "html.parser")
        group_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "groupPools" in href:
                group_links.append(_abs_url(href))
        group_links = list(dict.fromkeys(group_links))

        for link in group_links[:50]:
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
# Team page resolution + match crawl (authoritative season history)
# -----------------------------
def find_team_page_from_league(league_url: str, team_no: int) -> Optional[str]:
    """
    Starting from a league/group page, find the click-tt page for the specific team (Meyrin VI etc.).
    Returns a URL that should lead to team details (from which we can discover match links).
    """
    if not league_url:
        return None

    r = session.get(league_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    roman = ROMAN.get(team_no, str(team_no))
    labels = [
        f"Meyrin {roman}",
        f"Meyrin {team_no}",
        f"Hommes {roman}",
        f"Men {roman}",
        f"Herren {roman}",
    ]

    for a in soup.find_all("a", href=True):
        txt = _norm(a.get_text(" ", strip=True))
        for lab in labels:
            if lab.lower() in txt.lower():
                return _abs_url(a["href"])

    return None


def discover_match_detail_links(team_page_url: str, cap: int = 40) -> List[str]:
    """
    Given a team page, try to discover match detail links.
    We:
      - look for "teamMeetings/teamMatches" list pages
      - also collect direct match/meeting/groupMeeting links
    """
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # potential pages listing meetings
    list_pages = _find_links(soup, include=["teamMeetings", "teamMatches", "teamMeeting", "groupMeetings", "groupMatches"])
    if not list_pages:
        list_pages = [team_page_url]

    match_links: List[str] = []

    # scan a few list pages
    for page in list_pages[:3]:
        rr = session.get(page, timeout=25)
        rr.raise_for_status()
        ss = BeautifulSoup(rr.text, "html.parser")
        match_links.extend(_find_links(ss, include=["groupMeeting", "teamMeeting", "meeting", "match"]))

    match_links = list(dict.fromkeys(match_links))
    return match_links[:cap]


@st.cache_data(ttl=3600)
def build_player_season_history(
    player: PlayerKey,
    season_name: str,
    contest_type_token: str,
    team_entries_by_no: Dict[int, Dict[str, TeamInfo]],
    max_matches_per_team_phase: int = 35,
) -> Dict[int, Any]:
    """
    Authoritative history by crawling match details.

    Returns:
      history[team_no] = {
        "A": {"league": label, "matches": count} or None,
        "B": {"league": label, "matches": count} or None,
        "total": countA+countB
      }
    """
    target_last_first = f"{player.last}, {player.first}".lower()
    target_first_last = f"{player.first} {player.last}".lower()

    history: Dict[int, Any] = {}

    for team_no, phases in team_entries_by_no.items():
        team_rec = {"A": None, "B": None, "total": 0}

        for phase in ["A", "B"]:
            team = phases.get(phase)
            if not team or not team.league_url:
                continue

            # Resolve team page from league page for this phase
            team_page = find_team_page_from_league(team.league_url, team_no)
            if not team_page:
                continue

            match_links = discover_match_detail_links(team_page, cap=max_matches_per_team_phase)
            if not match_links:
                continue

            count = 0
            for murl in match_links:
                try:
                    mr = session.get(murl, timeout=25)
                    mr.raise_for_status()
                except Exception:
                    continue

                # fast text check (HTML as text)
                txt = mr.text.lower()
                if target_last_first in txt or target_first_last in txt:
                    count += 1

            if count > 0:
                team_rec[phase] = {"league": team.league_label, "matches": count}
                team_rec["total"] += count

        if team_rec["total"] > 0:
            history[team_no] = team_rec

    return history


# -----------------------------
# Optional: cross-check from player portrait (best-effort)
# -----------------------------
def fetch_portrait_crosscheck(portrait_url: str, team_entries_by_no: Dict[int, Dict[str, TeamInfo]]) -> Dict[int, int]:
    """
    Best-effort crosscheck. This is NOT authoritative.
    We try to spot occurrences of "Meyrin VI" / "Hommes VI" and nearby counts.
    Returns approximate team_no -> appearances.
    """
    r = session.get(portrait_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)

    approx: Dict[int, int] = {}
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    for team_no in team_entries_by_no.keys():
        roman = ROMAN.get(team_no, str(team_no))
        pats = [
            re.compile(rf"\bMeyrin\s*{roman}\b", re.I),
            re.compile(rf"\bMeyrin\s*{team_no}\b", re.I),
            re.compile(rf"\bHommes\s*{roman}\b", re.I),
            re.compile(rf"\bMen\s*{roman}\b", re.I),
            re.compile(rf"\bHerren\s*{roman}\b", re.I),
        ]

        for i, ln in enumerate(lines):
            if any(p.search(ln) for p in pats):
                window = " ".join(lines[i : i + 4])
                m = re.search(r"\b(?:Einsätze|Matchs|Rencontres)\b\D{0,10}(\d+)\b", window, re.I)
                if m:
                    approx[team_no] = max(approx.get(team_no, 0), int(m.group(1)))

    return approx


# -----------------------------
# Rules engine (uses authoritative history)
# -----------------------------
def check_eligibility(
    target_team: TeamInfo,
    teams_by_no: Dict[int, TeamInfo],
    nominated_team_no: Optional[int],
    history: Dict[int, Any],
) -> Tuple[bool, List[str]]:
    """
    Uses season history totals (both phases) as appearance counts per team.
    """
    reasons: List[str] = []

    # phase-agnostic ranks for comparisons
    lr: Dict[int, int] = {tno: info.league_level_rank for tno, info in teams_by_no.items()}

    def higher(a: int, b: int) -> bool:
        return lr.get(a, 0) > lr.get(b, 0)

    # appearances by team
    apps: Dict[int, int] = {tno: rec.get("total", 0) for tno, rec in history.items()}

    # aligned teams this season: any played team + nominated team (even if 0 apps)
    played_teams = sorted([t for t, n in apps.items() if n > 0])
    if nominated_team_no is not None:
        played_teams = sorted(set(played_teams + [nominated_team_no]))

    # ----- STRICT lock after 3 in higher team (nominated lower -> 3 in higher => titulaire + locked) -----
    locked_team_no: Optional[int] = None
    if nominated_team_no is not None:
        base = nominated_team_no
        for tno, n in apps.items():
            if higher(tno, base) and n >= 3:
                if locked_team_no is None or lr.get(tno, 0) > lr.get(locked_team_no, 0):
                    locked_team_no = tno

    if locked_team_no is not None and target_team.team_no != locked_team_no:
        return False, [
            "NOT ELIGIBLE (strict lock): player reached 3 alignments in a higher team and is locked to that team for the season.",
            f"Locked team: {locked_team_no}. Target: {target_team.team_no}.",
        ]

    # ----- STRICT no playing down (season-wide, both phases) -----
    target_rank = lr.get(target_team.team_no, 0)

    if nominated_team_no is not None and lr.get(nominated_team_no, 0) > target_rank:
        return False, [
            "NOT ELIGIBLE (strict no playing down): player is nominated in a higher league team for this season.",
            f"Nominated team {nominated_team_no} is higher than target team {target_team.team_no}.",
        ]

    for tno in played_teams:
        if lr.get(tno, 0) > target_rank:
            return False, [
                "NOT ELIGIBLE (strict no playing down): player already played in a higher league team this season.",
                f"Higher team played: {tno} > target {target_team.team_no}.",
            ]

    # ----- 50.4.5: at most two teams; if two, must be in different leagues -----
    prospective = sorted(set(played_teams + [target_team.team_no]))
    if len(prospective) > 2:
        return False, [
            "NOT ELIGIBLE (50.4.5): player may be aligned in at most two teams this season.",
            f"Aligned teams: {played_teams}; adding {target_team.team_no} -> {prospective}.",
        ]

    if len(prospective) == 2:
        t1, t2 = prospective
        if lr.get(t1, 0) == lr.get(t2, 0):
            return False, [
                "NOT ELIGIBLE (50.4.5): if two teams, they must be in different leagues.",
                f"Teams: {t1} and {t2} are same league level.",
            ]

    reasons.append("Passed 50.4.5 (max two teams, different leagues).")

    # ----- 50.4.7 lowest team immediate titulaire for non-nominated -----
    lowest_team_no = min(lr.keys(), key=lambda k: lr[k]) if lr else target_team.team_no

    # Count for target if they play now (prospective appearance count)
    current_target_apps = apps.get(target_team.team_no, 0)
    next_target_apps = current_target_apps + 1

    if nominated_team_no is None:
        if target_team.team_no == lowest_team_no and current_target_apps == 0:
            reasons.append(f"50.4.7: first alignment in lowest team {lowest_team_no} makes player titulaire there.")
            return True, ["ELIGIBLE"] + reasons

        # 50.4.6 non-nominated: third appearance makes titulaire of that team
        if next_target_apps >= 3:
            reasons.append(f"50.4.6: this would be appearance #{next_target_apps} in team {target_team.team_no}; becomes titulaire there.")
            return True, ["ELIGIBLE (but becomes titulaire)"] + reasons

        reasons.append(f"50.4.6: not nominated; this would be appearance #{next_target_apps} in team {target_team.team_no}.")
        return True, ["ELIGIBLE"] + reasons

    # ----- 50.4.6 nominated case -----
    base = nominated_team_no
    if target_team.team_no == base:
        reasons.append(f"Player is nominated in team {base}.")
        return True, ["ELIGIBLE"] + reasons

    if higher(target_team.team_no, base):
        # appearances in higher team count as substitute alignments for 50.4.6 purposes
        n = apps.get(target_team.team_no, 0) + 1
        if n <= 2:
            reasons.append(f"50.4.6: nominated in lower team {base}; this is substitute alignment #{n} in higher team {target_team.team_no} (allowed up to 2).")
            return True, ["ELIGIBLE"] + reasons
        reasons.append(f"50.4.6: this is substitute alignment #{n} in higher team {target_team.team_no}; becomes titulaire there and locked thereafter.")
        return True, ["ELIGIBLE (but triggers titular lock)"] + reasons

    reasons.append("50.4.6: target team is not a higher league than nominated team; no extra restriction here.")
    return True, ["ELIGIBLE"] + reasons


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Meyrin CTT – Eligibility Checker", layout="wide")
st.title("Meyrin CTT – Eligibility Checker (season-wide, both phases)")

team_entries = fetch_meyrin_team_entries()
team_entries_by_no = build_phase_team_map(team_entries)

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

    st.header("Dropdown phase (UI)")
    ui_phase = st.selectbox("Phase", options=["A", "B"], index=["A", "B"].index(infer_default_phase()))

# Build the dropdown team list for UI phase, but keep phase-agnostic rank
teams: List[TeamInfo] = []
for team_no, phase_map in sorted(team_entries_by_no.items()):
    chosen = phase_map.get(ui_phase)
    if not chosen:
        continue
    teams.append(chosen)
teams_by_no: Dict[int, TeamInfo] = {t.team_no: t for t in teams}

with st.sidebar:
    st.header("Target team")
    if not teams:
        st.error("No teams found from click-tt clubTeams page.")
        st.stop()

    target = st.selectbox("Meyrin team", options=teams, format_func=lambda t: f"{t.name} — {t.league_label}")

    st.markdown("### Debug")
    if st.button("Test click-tt"):
        test_url = f"{BASE}/clubTeams?club={MEYRIN_CLUB_ID}"
        resp = session.get(test_url, timeout=20)
        st.write("Status:", resp.status_code)
        st.write("Length:", len(resp.text))
        st.text(resp.text[:250])

st.divider()
st.subheader("Player")

c1, c2 = st.columns(2)
with c1:
    last = st.text_input("Last name", value="")
with c2:
    first = st.text_input("First name", value="")

advanced = st.checkbox("Advanced / debug", value=False)
max_matches = st.slider("Max matches per team+phase to scan", min_value=10, max_value=60, value=35, step=5)

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
        pick = st.radio("Multiple matches found — pick one:", options=picks, format_func=lambda p: p.display_name)

    player_key = infer_player_name_from_portrait(pick.portrait_url)
    if not player_key.last or not player_key.first:
        player_key = PlayerKey(last=last.strip(), first=first.strip())

    nominated_team_no = fetch_regular_registration_nominated_team(season_name, contest_type, player_key)

    # Authoritative season history (both phases)
    history = build_player_season_history(
        player=player_key,
        season_name=season_name,
        contest_type_token=contest_type,
        team_entries_by_no=team_entries_by_no,
        max_matches_per_team_phase=max_matches,
    )

    # Optional cross-check from portrait
    portrait_approx = fetch_portrait_crosscheck(pick.portrait_url, team_entries_by_no) if advanced else {}

    ok, reasons = check_eligibility(
        target_team=target,
        teams_by_no=teams_by_no,
        nominated_team_no=nominated_team_no,
        history=history,
    )

    st.markdown("### Result")
    if ok:
        st.success(reasons[0])
    else:
        st.error(reasons[0])

    # Build "played teams summary" you requested
    played_summary = []
    for tno, rec in sorted(history.items()):
        row = {"team_no": tno, "total": rec.get("total", 0)}
        if rec.get("A"):
            row["phase_A_league"] = rec["A"]["league"]
            row["phase_A_matches"] = rec["A"]["matches"]
        if rec.get("B"):
            row["phase_B_league"] = rec["B"]["league"]
            row["phase_B_matches"] = rec["B"]["matches"]
        played_summary.append(row)

    st.markdown("### Player season history (authoritative: match crawl)")
    if played_summary:
        st.json(played_summary)
    else:
        st.info("No appearances found via match crawl for this season (could be none yet, or links differ for this competition).")

    st.markdown("### Details")
    st.write(
        {
            "player": f"{player_key.last}, {player_key.first}",
            "portrait": pick.portrait_url,
            "nominated_team_no": nominated_team_no,
            "target_team": f"{target.name} ({target.league_label})",
            "teams_played_this_season": sorted(history.keys()),
            "history_raw": history if advanced else "(enable Advanced to view)",
            "portrait_crosscheck_approx": portrait_approx if advanced else "(enable Advanced to view)",
        }
    )

    st.markdown("### Rule reasoning")
    for r in reasons[1:]:
        st.write(f"- {r}")

    if not history:
        st.warning(
            "We could not build a season appearance history from match pages. "
            "If you believe the player already played, enable Advanced and increase the match scan cap, "
            "or share one team page URL so we can tighten match-link discovery."
        )
