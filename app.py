
# app.py — Meyrin CTT Eligibility Checker (match-day counting + 50.4.4 + last-48h debug)
#
# ✅ Appearances are counted as TEAM MATCH-DAYS (rencontres), NOT individual games.
#    A match-day can include 3 singles + 1 double → counts as 1 appearance.
#
# Simplified rules:
# Case 1: Licensed but never played this season (category) -> can play any team.
# Case 2: Player has played in a league -> cannot play another team of the same league or below,
#         EXCEPT: they may still play in their own/base lower team if they only played 1–2 times above it.
# Case 3: If playing up (target higher than base), 3rd appearance in that higher team -> WARNING:
#         becomes titulaire there and loses right to play base team (still allowed today, but warned).
#         IMPORTANT: Case 3 is evaluated BEFORE Case 2 blocks.
# Case 4: If eligible, check last 48h matches of OTHER teams in same league level where results may be unpublished -> WARNING.
#
# Additional rule at the very end (only if target is a "replacement" team i.e., not base team):
# 50.4.4: Le classement du joueur remplaçant ne peut être supérieur à celui du joueur titulaire qu'il remplace.
# We can't automatically know who is replaced. We therefore display:
#   - Replacement player ranking
#   - Target team roster players and their rankings
#   - Guidance/warnings to check the replaced titulaire's ranking.
#
# Debug:
# - Last 48h scan shows teams scanned + matches found + score status.
#
# Deploy: Streamlit Cloud compatible.

import re
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import requests
from bs4 import BeautifulSoup
import streamlit as st

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None


BASE = "https://www.click-tt.ch/cgi-bin/WebObjects/nuLigaTTCH.woa/wa"
MEYRIN_CLUB_ID = 33165
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}

session = requests.Session()
session.headers.update({"User-Agent": "Meyrin-Eligibility-Checker/fast-2.2 (+club tool; public pages only)"})


# -----------------------------
# Models
# -----------------------------
@dataclass(frozen=True)
class TeamInfo:
    team_no: int
    name: str
    league_label: str
    league_url: str
    league_level_rank: int


@dataclass(frozen=True)
class PlayerPick:
    display_name: str
    portrait_url: str


@dataclass(frozen=True)
class PlayerKey:
    last: str
    first: str


# -----------------------------
# Helpers
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
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    return mapping.get(token)


def infer_default_phase() -> str:
    m = datetime.date.today().month
    return "A" if 8 <= m <= 12 else "B"


def league_rank(label: str) -> int:
    """
    Higher number == higher league level.
    Ignore phase/group text. Recognize Nationalliga A/B/C by words (not only NLA/NLB/NLC).
    """
    s = (label or "").lower()
    s = re.sub(r"\bphase\s*[ab]\b", " ", s)
    s = re.sub(r"\bgr\.?\s*\d+\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # National leagues
    if "nationalliga a" in s or "nla" in s:
        return 100
    if "nationalliga b" in s or "nlb" in s:
        return 90
    if "nationalliga c" in s or "nlc" in s:
        return 80

    # Regional leagues
    m = re.search(r"\b(\d+)\s*(?:ème|eme|\.|)\s*(?:ligue|liga)\b", s)
    if m:
        n = int(m.group(1))
        return 70 - n

    m = re.search(r"\b(?:ligue|liga)\s*(\d+)\b", s)
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


def now_ch() -> datetime.datetime:
    if ZoneInfo:
        return datetime.datetime.now(ZoneInfo("Europe/Zurich"))
    return datetime.datetime.now()


# -----------------------------
# Teams (phase-aware dropdown)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_meyrin_team_entries() -> Dict[int, List[Tuple[str, str]]]:
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

        # ignore cup-ish
        if "hauptrunde" in league_label.lower() or "cup" in league_label.lower():
            continue

        entries.setdefault(team_no, []).append((league_label, league_url))

    return entries


def pick_phase_entry(options: List[Tuple[str, str]], phase: str) -> Tuple[str, str]:
    key = f"phase {phase}".lower()
    for label, url in options:
        if key in (label or "").lower():
            return label, url
    return options[0]


def build_teams_for_phase(team_entries: Dict[int, List[Tuple[str, str]]], phase: str) -> List[TeamInfo]:
    teams: List[TeamInfo] = []
    for team_no, opts in sorted(team_entries.items()):
        label, url = pick_phase_entry(opts, phase)
        teams.append(TeamInfo(team_no, f"Meyrin {team_no}", label, url, league_rank(label)))
    return teams


def build_teams_for_both_phases(team_entries: Dict[int, List[Tuple[str, str]]]) -> List[TeamInfo]:
    teams_all: List[TeamInfo] = []
    for ph in ["A", "B"]:
        teams_all.extend(build_teams_for_phase(team_entries, ph))
    # de-dup by (team_no, league_url)
    seen = set()
    out = []
    for t in teams_all:
        key = (t.team_no, t.league_url)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# -----------------------------
# Player lookup + ranking
# -----------------------------
def search_player_in_meyrin_club(last: str, first: str) -> List[PlayerPick]:
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

    # de-dup
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


def _parse_ranking_token(s: str) -> Optional[str]:
    """
    Best-effort parse of ranking/class (Swiss: e.g., D1/C8/B12/A16, etc.)
    We only display it; we do NOT compute ordering.
    """
    s = _norm(s).upper()
    # Common patterns like A16, B12, C10, D5, E1
    m = re.search(r"\b([A-E])\s*([0-9]{1,2})\b", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    # Sometimes just letters like "D" (less common)
    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1)
    return None


@st.cache_data(ttl=3600)
def fetch_player_ranking_from_portrait(portrait_url: str) -> Optional[str]:
    """
    Extracts player's ranking/class from portrait page (best-effort).
    """
    r = session.get(portrait_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    # Look for keywords around ranking (multilingual)
    # We'll just scan for first plausible token.
    tok = _parse_ranking_token(text)
    return tok


# -----------------------------
# Nominated team from registration (best-effort)
# -----------------------------
def fetch_regular_registration_nominated_team(season_name: str, contest_type_token: str, player: PlayerKey) -> Optional[int]:
    target_last = player.last.strip().lower()
    target_first = player.first.strip().lower()

    def scan_html(html: str) -> Optional[int]:
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.find_all("tr"):
            row = _norm(tr.get_text(" ", strip=True)).lower()
            if target_last not in row or target_first not in row:
                continue
            m = re.search(r"\b(\d+)\.\d+\b", row)
            if m:
                return int(m.group(1))
        return None

    url = f"{BASE}/clubPools"
    for display_typ in ("vorrunde", "rueckrunde"):
        params = {
            "club": str(MEYRIN_CLUB_ID),
            "contestType": contest_type_token,
            "displayTyp": display_typ,
            "preferredLanguage": "German",
            "seasonName": season_name,
        }
        r = session.get(url, params=params, timeout=25)
        r.raise_for_status()
        team_no = scan_html(r.text)
        if team_no is not None:
            return team_no

        soup = BeautifulSoup(r.text, "html.parser")
        group_links = []
        for a in soup.find_all("a", href=True):
            if "groupPools" in a["href"]:
                group_links.append(_abs_url(a["href"]))
        group_links = list(dict.fromkeys(group_links))

        for link in group_links[:50]:
            rr = session.get(link, timeout=25)
            rr.raise_for_status()
            team_no = scan_html(rr.text)
            if team_no is not None:
                return team_no

    return None


# -----------------------------
# Resolve team page + match list
# -----------------------------
@st.cache_data(ttl=3600)
def find_team_page_from_league(league_url: str, team_no: int) -> Optional[str]:
    """
    From league/group page, find link to the specific Meyrin team page.
    """
    if not league_url:
        return None
    r = session.get(league_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    roman = ROMAN.get(team_no, str(team_no))
    labels = [f"Meyrin {roman}", f"Meyrin {team_no}"]

    for a in soup.find_all("a", href=True):
        txt = _norm(a.get_text(" ", strip=True))
        if any(lab.lower() in txt.lower() for lab in labels):
            return _abs_url(a["href"])
    return None


def find_team_match_list_url(team_page_url: str) -> str:
    """
    Prefer a dedicated match list page (teamMeetings/teamMatches) if linked; else team_page_url.
    """
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for href in _find_links(soup, include=["teamMeetings", "teamMatches", "groupMeetings", "groupMatches"]):
        return href
    return team_page_url


# -----------------------------
# Match-day counting (correct appearances)
# -----------------------------
@st.cache_data(ttl=900)
def count_player_matchdays_for_team(team_page_url: str, player: PlayerKey, max_meetings: int = 40) -> int:
    """
    Counts appearances as TEAM MATCH-DAYS by scanning meeting pages.
    """
    target1 = f"{player.last}, {player.first}".lower()
    target2 = f"{player.first} {player.last}".lower()

    list_url = find_team_match_list_url(team_page_url)
    r = session.get(list_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # meeting sheet links
    meeting_links = _find_links(soup, include=["groupMeeting", "teamMeeting", "meeting"])
    meeting_links = meeting_links[:max_meetings]

    count = 0
    for url in meeting_links:
        try:
            mr = session.get(url, timeout=25)
            mr.raise_for_status()
        except Exception:
            continue
        txt = mr.text.lower()
        if target1 in txt or target2 in txt:
            count += 1

    return count


@st.cache_data(ttl=900)
def fetch_player_apps_across_meyrin_teams_matchdays(
    player: PlayerKey,
    teams: List[TeamInfo],
    max_meetings_per_team: int = 40,
) -> Dict[int, int]:
    """
    Counts match-day appearances per team, summed across phase pages (teams list may have same team_no twice).
    Returns team_no -> matchdays_count (>0 only).
    """
    out: Dict[int, int] = {}
    for t in teams:
        team_page = find_team_page_from_league(t.league_url, t.team_no)
        if not team_page:
            continue
        n = count_player_matchdays_for_team(team_page, player, max_meetings=max_meetings_per_team)
        if n > 0:
            out[t.team_no] = out.get(t.team_no, 0) + n
    return out


# -----------------------------
# 50.4.4 roster listing (best-effort)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_team_roster_with_rankings(team_page_url: str) -> List[Dict[str, str]]:
    """
    Best-effort: parse a team page (or linked teamPortrait) and extract player names + ranking tokens.
    Returns list of dicts: {"name": "Last, First", "rank": "D1"} (rank may be None).
    """
    # Try to find teamPortrait link first
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    candidate_urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "teamPortrait" in href or "teamPortraitTT" in href:
            candidate_urls.append(_abs_url(href))
    candidate_urls = candidate_urls[:1] or [team_page_url]

    roster: List[Dict[str, str]] = []
    seen_names = set()

    for url in candidate_urls:
        rr = session.get(url, timeout=25)
        rr.raise_for_status()
        ss = BeautifulSoup(rr.text, "html.parser")

        # Heuristic: scan table rows; find something like "Last, First" and a ranking token in row
        for tr in ss.find_all("tr"):
            row_txt = _norm(tr.get_text(" ", strip=True))
            if "," not in row_txt:
                continue
            # pick the first "Last, First" chunk from the row
            m = re.search(r"([A-Za-zÀ-ÿ'\- ]+,\s*[A-Za-zÀ-ÿ'\- ]+)", row_txt)
            if not m:
                continue
            name = _norm(m.group(1))
            if name in seen_names:
                continue
            seen_names.add(name)

            rank = _parse_ranking_token(row_txt)
            roster.append({"name": name, "rank": rank or ""})

    return roster


# -----------------------------
# Recent pending results (last 48h) + debug
# -----------------------------
def parse_date_from_text(s: str) -> Optional[datetime.datetime]:
    s = _norm(s)
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})(?:\s+(\d{1,2}):(\d{2}))?\b", s)
    if not m:
        return None
    d = int(m.group(1))
    mo = int(m.group(2))
    y = int(m.group(3))
    if y < 100:
        y += 2000
    hh = int(m.group(4)) if m.group(4) else 0
    mm = int(m.group(5)) if m.group(5) else 0
    if ZoneInfo:
        return datetime.datetime(y, mo, d, hh, mm, tzinfo=ZoneInfo("Europe/Zurich"))
    return datetime.datetime(y, mo, d, hh, mm)


def _match_has_score(txt: str) -> bool:
    t = txt.replace(" ", "")
    if "-:-" in t or ":-" in t:
        return False
    return re.search(r"\b\d+\s*:\s*\d+\b", txt) is not None


def pending_results_last_48h_with_debug(
    teams_same_league: List[TeamInfo],
    exclude_team_no: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns:
      warnings: list of warning strings
      debug: {team_name: {"list_url":..., "found": [{"text":..., "dt":..., "has_score":...}, ...]}}
    """
    now = now_ch()
    cutoff = now - datetime.timedelta(hours=48)

    warnings: List[str] = []
    debug: Dict[str, Any] = {}

    for t in teams_same_league:
        if t.team_no == exclude_team_no:
            continue

        debug[t.name] = {"list_url": None, "found": []}

        team_page = find_team_page_from_league(t.league_url, t.team_no)
        if not team_page:
            continue

        list_url = find_team_match_list_url(team_page)
        debug[t.name]["list_url"] = list_url

        rr = session.get(list_url, timeout=25)
        rr.raise_for_status()
        soup = BeautifulSoup(rr.text, "html.parser")

        for tr in soup.find_all("tr"):
            txt = _norm(tr.get_text(" ", strip=True))
            dt = parse_date_from_text(txt)
            if not dt:
                continue
            if dt < cutoff or dt > now + datetime.timedelta(hours=2):
                continue

            has_score = _match_has_score(txt)
            debug[t.name]["found"].append({"text": txt, "dt": dt.isoformat(), "has_score": has_score})

            if not has_score:
                warnings.append(f"Team {t.name}: match within last 48h may have unpublished result: {txt}")

    warnings = list(dict.fromkeys(warnings))[:10]
    return warnings, debug


# -----------------------------
# Eligibility logic (Case 3 checked before Case 2 blocks)
# -----------------------------
def decide_eligibility(
    target: TeamInfo,
    nominated_team_no: Optional[int],
    teams_by_no: Dict[int, TeamInfo],
    apps_by_team: Dict[int, int],
) -> Tuple[bool, List[str], int]:
    """
    Returns (ok, messages, base_team_no).
    base_team_no is used later to decide whether 50.4.4 should be shown (replacement scenario).
    """
    msgs: List[str] = []
    total_apps = sum(apps_by_team.values())

    # Case 1: never played
    if total_apps == 0:
        return True, [
            "ELIGIBLE (Case 1): player has not played yet this season.",
            "First appearance may be in any team.",
        ], -1

    played_teams = [tno for tno, n in apps_by_team.items() if n > 0 and tno in teams_by_no]
    if not played_teams:
        return True, ["ELIGIBLE (fallback): no valid played teams detected."], -1

    # base team = nominated if possible, else team with most appearances
    if nominated_team_no is not None and nominated_team_no in teams_by_no:
        base_team_no = nominated_team_no
    else:
        base_team_no = max(played_teams, key=lambda t: apps_by_team.get(t, 0))

    base_rank = teams_by_no[base_team_no].league_level_rank
    target_rank = target.league_level_rank
    max_played_rank = max(teams_by_no[t].league_level_rank for t in played_teams)

    already_in_target = apps_by_team.get(target.team_no, 0)
    next_count = already_in_target + 1

    higher_than_base_apps = sum(
        apps_by_team[t]
        for t in played_teams
        if teams_by_no[t].league_level_rank > base_rank
    )

    # --- Case 3 first: playing UP relative to base is allowed, with 3rd appearance warning ---
    if target_rank > base_rank:
        if next_count == 3:
            msgs.append(
                f"WARNING (Case 3): this would be appearance #3 in higher team {target.team_no}. "
                f"Player becomes titulaire there and loses right to play base team {base_team_no}."
            )
        elif next_count > 3:
            msgs.append(
                f"WARNING (Case 3): player already has {already_in_target} appearances in higher team {target.team_no}. "
                f"They should no longer play base team {base_team_no}."
            )
        else:
            msgs.append(
                f"Case 3 info: appearance #{next_count} in higher team {target.team_no} (titulaire switch at #3)."
            )
        return True, ["ELIGIBLE"] + msgs, base_team_no

    # --- Case 2 (same league or down) ---
    if target_rank <= max_played_rank:
        # Exception: allow base team if only 1–2 appearances above base
        if target.team_no == base_team_no and higher_than_base_apps <= 2:
            msgs.append(
                f"Case 2 exception: only {higher_than_base_apps} appearance(s) above base team {base_team_no}; "
                "still allowed to play base team."
            )
            return True, ["ELIGIBLE"] + msgs, base_team_no

        return False, [
            "NOT ELIGIBLE (Case 2): player already played in same league level or higher; "
            "cannot play another team of same league or below.",
        ], base_team_no

    msgs.append("Passed Case 2: target is higher than any league played so far.")
    return True, ["ELIGIBLE"] + msgs, base_team_no


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Meyrin CTT – Eligibility Checker", layout="wide")
st.title("Meyrin CTT – Eligibility Checker (match-day counting)")

team_entries = fetch_meyrin_team_entries()

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

teams_ui = build_teams_for_phase(team_entries, ui_phase)
teams_by_no = {t.team_no: t for t in teams_ui}

with st.sidebar:
    st.header("Target team")
    if not teams_ui:
        st.error("No teams found from click-tt clubTeams page.")
        st.stop()
    target = st.selectbox("Meyrin team", options=teams_ui, format_func=lambda t: f"{t.name} — {t.league_label}")

st.divider()
st.subheader("Player")

c1, c2 = st.columns(2)
with c1:
    last = st.text_input("Last name", value="")
with c2:
    first = st.text_input("First name", value="")

run_pending_check = st.checkbox("Also check last 48h pending results (warnings)", value=True)
show_last48_debug = st.checkbox("Show last 48h debug (teams + matches found)", value=False)
max_meetings = st.slider("Max match-days to scan per team+phase (for appearance counting)", min_value=10, max_value=60, value=40, step=5)
show_5044 = st.checkbox("Show 50.4.4 roster/ranking check at end (when replacement)", value=True)

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

    # Count match-day appearances across BOTH phases so you don't miss Phase A vs Phase B substitutes
    teams_both = build_teams_for_both_phases(team_entries)
    apps_by_team = fetch_player_apps_across_meyrin_teams_matchdays(player_key, teams_both, max_meetings_per_team=max_meetings)

    ok, messages, base_team_no = decide_eligibility(target, nominated_team_no, teams_by_no, apps_by_team)

    st.markdown("### Result")
    if ok:
        st.success(messages[0])
    else:
        st.error(messages[0])

    st.markdown("### Reasoning / Warnings")
    for m in messages[1:]:
        st.write(f"- {m}")

    st.markdown("### Details")
    st.write(
        {
            "player": f"{player_key.last}, {player_key.first}",
            "portrait": pick.portrait_url,
            "player_ranking": fetch_player_ranking_from_portrait(pick.portrait_url),
            "nominated_team_no": nominated_team_no,
            "base_team_no": base_team_no if base_team_no != -1 else None,
            "apps_by_team_matchdays": apps_by_team,
            "target_team": f"{target.name} ({target.league_label})",
        }
    )

    # Case 4 warning + debug
    last48_debug_data = None
    if ok and run_pending_check:
        same_level_teams = [t for t in teams_ui if t.league_level_rank == target.league_level_rank]
        warnings, dbg = pending_results_last_48h_with_debug(same_level_teams, exclude_team_no=target.team_no)
        last48_debug_data = dbg
        if warnings:
            st.warning(
                "Recent matches (last 48h) in other teams of the same league may have unpublished results. "
                "Verify with the other team:"
            )
            for w in warnings:
                st.write(f"- {w}")

    if ok and run_pending_check and show_last48_debug:
        st.markdown("### Debug — last 48 hours scan")
        if not last48_debug_data:
            st.info("No debug data collected (either no teams scanned or no matches within 48h).")
        else:
            st.json(last48_debug_data)

    # 50.4.4 at the very end (only if replacement: target team != base team)
    if show_5044 and ok:
        is_replacement = (base_team_no != -1 and target.team_no != base_team_no)
        if is_replacement:
            st.markdown("### 50.4.4 — Remplacement / Ranking check (manual verification)")
            st.write(
                "Rule 50.4.4: **Le classement du joueur remplaçant ne peut être supérieur à celui du joueur titulaire qu'il remplace.**"
            )

            player_rank = fetch_player_ranking_from_portrait(pick.portrait_url) or ""
            st.write(f"- Replacement player: **{player_key.last}, {player_key.first}** — ranking: **{player_rank or 'unknown'}**")

            # list target team roster + rankings
            team_page = find_team_page_from_league(target.league_url, target.team_no)
            roster = fetch_team_roster_with_rankings(team_page) if team_page else []
            if roster:
                st.write(f"Target team roster (best-effort from click-tt page): **{target.name}**")
                st.dataframe(roster, use_container_width=True)

                # Strong warning if we can compare lexicographically only? We can't reliably order ranks
                # (A/B/C/D with numbers often invert depending on federation), so we avoid auto decision.
                st.info(
                    "!Please ensure the **specific titulaire being replaced** has a ranking **at least as strong** as the replacement player."
                )
            else:
                st.warning(
                    "Could not extract the target team roster/rankings from click-tt automatically. "
                    "Please open the team page and verify the titulaire being replaced."
                )
