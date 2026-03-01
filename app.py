# app.py — FAST simplified eligibility checker for Meyrin CTT (with both-phase scan + match-sheet fallback)
#
# Simplified rules:
# Case 1: Licensed but never played this season (category) -> can play any team.
# Case 2: Player has played in a league -> cannot play another team of the same league or below,
#         EXCEPT: they may still play in their own/base lower team if they only played 1–2 times above it.
# Case 3: If playing up (target higher than base), 3rd appearance in that higher team -> WARNING:
#         becomes titulaire there and loses right to play base team (still allowed today, but warned).
# Case 4: If eligible, check last 48h matches of OTHER teams in same league level where results may be unpublished -> WARNING.
#
# Data strategy (fast + accurate enough):
# - Player lookup via clubLicenceMembersPage (reliable)
# - Nominated/base team via clubPools/groupPools (best-effort)
# - Appearances by team via team roster/bilan table pages across BOTH phases (A+B), summed
# - Fallback: if target team is missing, scan last N match sheets for target team only (fast)
#
# Deploy: Streamlit Cloud compatible.

import re
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
session.headers.update({"User-Agent": "Meyrin-Eligibility-Checker/fast-1.1 (+club tool; public pages only)"})


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
    Ignore phase/group text.
    """
    s = (label or "").lower()
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
        return 70 - n

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
# Player lookup
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
# FAST appearances scan (team roster/bilan), across BOTH phases
# -----------------------------
@st.cache_data(ttl=3600)
def find_team_page_from_league(league_url: str, team_no: int) -> Optional[str]:
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


def _pick_best_roster_like_url(team_page_url: str) -> str:
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "teamPortrait" in href or "teamPortraitTT" in href:
            return _abs_url(href)
    return team_page_url


def extract_player_apps_from_team_page(team_roster_url: str, player: PlayerKey) -> int:
    r = session.get(team_roster_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    target1 = f"{player.last}, {player.first}".lower()
    target2 = f"{player.first} {player.last}".lower()

    for tr in soup.find_all("tr"):
        row_text = _norm(tr.get_text(" ", strip=True)).lower()
        if target1 not in row_text and target2 not in row_text:
            continue
        ints = [int(x) for x in re.findall(r"\b(\d+)\b", tr.get_text(" ", strip=True))]
        if ints:
            return ints[-1]
        return 0

    return 0


@st.cache_data(ttl=900)
def fetch_player_apps_across_meyrin_teams(player: PlayerKey, teams: List[TeamInfo]) -> Dict[int, int]:
    """
    Sums appearances across phase pages (teams list may contain same team_no with different league_url).
    """
    out: Dict[int, int] = {}
    for t in teams:
        team_page = find_team_page_from_league(t.league_url, t.team_no)
        if not team_page:
            continue
        roster_url = _pick_best_roster_like_url(team_page)
        apps = extract_player_apps_from_team_page(roster_url, player)
        if apps > 0:
            out[t.team_no] = out.get(t.team_no, 0) + apps
    return out


# -----------------------------
# Match-sheet fallback for missing target team
# -----------------------------
def find_team_match_list_url(team_page_url: str) -> str:
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for href in _find_links(soup, include=["teamMeetings", "teamMatches", "groupMeetings", "groupMatches"]):
        return href
    return team_page_url


def count_player_in_last_matches(team_page_url: str, player: PlayerKey, max_matches: int = 12) -> int:
    target1 = f"{player.last}, {player.first}".lower()
    target2 = f"{player.first} {player.last}".lower()

    list_url = find_team_match_list_url(team_page_url)
    r = session.get(list_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    match_links = _find_links(soup, include=["groupMeeting", "teamMeeting", "meeting", "match"])
    match_links = match_links[:max_matches]

    count = 0
    for murl in match_links:
        try:
            mr = session.get(murl, timeout=25)
            mr.raise_for_status()
        except Exception:
            continue
        txt = mr.text.lower()
        if target1 in txt or target2 in txt:
            count += 1
    return count


# -----------------------------
# Recent pending results (last 48h) check
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


def pending_results_last_48h(teams_same_league: List[TeamInfo], exclude_team_no: int) -> List[str]:
    now = now_ch()
    cutoff = now - datetime.timedelta(hours=48)
    warnings: List[str] = []

    for t in teams_same_league:
        if t.team_no == exclude_team_no:
            continue

        team_page = find_team_page_from_league(t.league_url, t.team_no)
        if not team_page:
            continue

        list_url = find_team_match_list_url(team_page)
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

            if "-:-" in txt or ":-" in txt:
                warnings.append(f"Team {t.name}: match within last 48h might have unpublished result: {txt}")
            else:
                if not re.search(r"\b\d+\s*:\s*\d+\b", txt):
                    warnings.append(f"Team {t.name}: recent match may not have published score yet: {txt}")

    return list(dict.fromkeys(warnings))[:10]


# -----------------------------
# Simplified eligibility logic (with Case 2 exception)
# -----------------------------
def decide_eligibility(
    target: TeamInfo,
    nominated_team_no: Optional[int],
    teams_by_no: Dict[int, TeamInfo],
    apps_by_team: Dict[int, int],
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, messages).
    """
    msgs: List[str] = []
    total_apps = sum(apps_by_team.values())

    # Case 1: never played
    if total_apps == 0:
        return True, [
            "ELIGIBLE (Case 1): player is licensed but has not played yet this season in this category.",
            "They may play in any team for their first match.",
        ]

    played_teams = sorted([tno for tno, n in apps_by_team.items() if n > 0 and tno in teams_by_no])
    if not played_teams:
        return True, ["ELIGIBLE (fallback): no played teams detected."]

    # base/own team: nominated if available, else team with most appearances
    if nominated_team_no is not None and nominated_team_no in teams_by_no:
        base_team_no = nominated_team_no
    else:
        base_team_no = max(played_teams, key=lambda tno: apps_by_team.get(tno, 0))

    base_rank = teams_by_no[base_team_no].league_level_rank
    target_rank = target.league_level_rank
    max_played_rank = max(teams_by_no[tno].league_level_rank for tno in played_teams)

    # how many appearances ABOVE base team
    higher_than_base_apps = sum(
        apps_by_team[tno]
        for tno in played_teams
        if teams_by_no[tno].league_level_rank > base_rank
    )

    # Case 2: generally cannot play same league or below relative to highest played league
    if target_rank <= max_played_rank:
        # Exception: they may still play base team if only 1–2 appearances above base
        if target.team_no == base_team_no and higher_than_base_apps <= 2:
            msgs.append(
                f"Case 2 exception: player has only {higher_than_base_apps} appearance(s) above base team {base_team_no}, "
                "so they may still play in their base team."
            )
        else:
            max_rank_team = None
            for tno in played_teams:
                if teams_by_no[tno].league_level_rank == max_played_rank:
                    max_rank_team = tno
                    break
            return False, [
                "NOT ELIGIBLE (Case 2): player already played in same league level or higher; cannot play another team of same league or below.",
                f"Highest league played includes team {max_rank_team} ({teams_by_no[max_rank_team].league_label}); "
                f"target is team {target.team_no} ({target.league_label}).",
                f"Base team is {base_team_no} ({teams_by_no[base_team_no].league_label}).",
            ]
    else:
        msgs.append("Passed Case 2: target is in a higher league than any league the player has played so far.")

    # Case 3: 3rd appearance in higher team => titulaire warning
    if target_rank > base_rank:
        already_in_target = apps_by_team.get(target.team_no, 0)
        next_count = already_in_target + 1
        if next_count == 3:
            msgs.append(
                f"WARNING (Case 3): this would be the 3rd appearance in higher team {target.team_no} this season. "
                f"Player becomes titulaire of the higher team and loses right to play base team {base_team_no}."
            )
        elif next_count > 3:
            msgs.append(
                f"WARNING (Case 3): player already has {already_in_target} appearances in higher team {target.team_no}. "
                f"They should no longer play in base team {base_team_no}."
            )
        else:
            msgs.append(
                f"Case 3 info: appearance #{next_count} in higher team {target.team_no} (titulaire switch happens at #3)."
            )

    return True, ["ELIGIBLE"] + msgs


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Meyrin CTT – Eligibility Checker (Fast)", layout="wide")
st.title("Meyrin CTT – Eligibility Checker (Fast simplified)")

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
fallback_match_scan = st.checkbox("Fallback: scan last matches if target team not found", value=True)
fallback_max_matches = st.slider("Fallback max matches to scan (target team only)", 5, 25, 12, 1)

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

    # Scan BOTH phases to not miss substitutes in other phase
    teams_both = build_teams_for_both_phases(team_entries)
    apps_by_team = fetch_player_apps_across_meyrin_teams(player_key, teams_both)

    # Fallback: if target team missing, scan last match sheets for target team
    fallback_used = False
    fallback_found = 0
    if fallback_match_scan and target.team_no not in apps_by_team:
        team_page = find_team_page_from_league(target.league_url, target.team_no)
        if team_page:
            found = count_player_in_last_matches(team_page, player_key, max_matches=fallback_max_matches)
            if found > 0:
                apps_by_team[target.team_no] = found
                fallback_used = True
                fallback_found = found

    ok, messages = decide_eligibility(target, nominated_team_no, teams_by_no, apps_by_team)

    st.markdown("### Result")
    if ok:
        st.success(messages[0])
    else:
        st.error(messages[0])

    st.markdown("### Reasoning / Warnings")
    for m in messages[1:]:
        st.write(f"- {m}")
    if fallback_used:
        st.info(f"Fallback used: found {fallback_found} appearance(s) for target team {target.team_no} by scanning last match sheets.")

    st.markdown("### Details")
    st.write(
        {
            "player": f"{player_key.last}, {player_key.first}",
            "portrait": pick.portrait_url,
            "nominated_team_no": nominated_team_no,
            "apps_by_team": apps_by_team,
            "target_team": f"{target.name} ({target.league_label})",
        }
    )

    # Case 4 warning
    if ok and run_pending_check:
        same_level_teams = [t for t in teams_ui if t.league_level_rank == target.league_level_rank]
        warnings = pending_results_last_48h(same_level_teams, exclude_team_no=target.team_no)
        if warnings:
            st.warning(
                "Recent matches (last 48h) in other teams of the same league may have unpublished results. "
                "Verify with the other team:"
            )
            for w in warnings:
                st.write(f"- {w}")
