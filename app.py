# app.py — FAST simplified eligibility checker for Meyrin CTT
#
# Simplified rules (season-wide, both phases implicitly):
# Case 1) Licensed but never played in this category this season -> can play in any team.
# Case 2) If player has played in a league, they cannot play another team of the same league or below.
#         (i.e., they may only move UP in league level relative to their highest league played so far.)
# Case 3) Replacements up: if nominated in a lower team, and they play in a higher league team:
#         - 3rd appearance in that higher team -> WARNING (becomes titulaire there; loses right to play initial lower team).
#         (We still allow the match, but warn.)
# Case 4) If everything is OK, scan last 48h matches of OTHER Meyrin teams in same league level as target
#         that might have un-published results -> WARNING to verify with the other team.
#
# Data strategy (fast):
# - Player selection via clubLicenceMembersPage (reliable)
# - Nominated team via clubPools/groupPools (best-effort but usually works)
# - Player appearances per team via each team page roster/bilan table (7–8 pages max, cached)
# - Recent pending results via each team match list page (not match details), only for same-league teams.

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
session.headers.update({"User-Agent": "Meyrin-Eligibility-Checker/fast-1.0 (+club tool; public pages only)"})


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

    # "Ligue 4" / "Liga 4"
    m = re.search(r"\b(?:ligue|liga)\s*(\d+)\b", s)
    if m:
        n = int(m.group(1))
        return 70 - n

    # "4ème ligue" / "4eme ligue" / "4. liga"
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
    # fallback
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
# FAST appearance scan (per-team roster/bilan table)
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
    """
    Some team pages link to a specific team portrait page containing roster/bilan tables.
    If we find such a link, prefer it; else parse the team_page_url directly.
    """
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # common endpoints
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "teamPortrait" in href or "teamPortraitTT" in href:
            return _abs_url(href)
    return team_page_url


def extract_player_apps_from_team_page(team_roster_url: str, player: PlayerKey) -> int:
    """
    Returns an integer appearances count found for this player in this team page.
    Best-effort heuristic: find the row containing "Last, First" and take the last integer on the row.
    """
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


@st.cache_data(ttl=900)  # shorter cache; changes as season progresses
def fetch_player_apps_across_meyrin_teams(
    player: PlayerKey,
    teams: List[TeamInfo],
) -> Dict[int, int]:
    """
    Fast scan: for each team in current phase list (7–8 teams),
    resolve team page and extract player's appearances.
    Returns team_no -> apps (>0 only).
    """
    out: Dict[int, int] = {}
    for t in teams:
        team_page = find_team_page_from_league(t.league_url, t.team_no)
        if not team_page:
            continue
        roster_url = _pick_best_roster_like_url(team_page)
        apps = extract_player_apps_from_team_page(roster_url, player)
        if apps > 0:
            out[t.team_no] = apps
    return out


# -----------------------------
# Recent pending results (last 48h) check
# -----------------------------
def parse_date_from_text(s: str) -> Optional[datetime.datetime]:
    """
    Try parse common click-tt date formats: dd.mm.yyyy or dd.mm.yy (with time optional)
    """
    s = _norm(s)
    # dd.mm.yyyy hh:mm
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


def find_team_match_list_url(team_page_url: str) -> str:
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Prefer explicit list pages
    for href in _find_links(soup, include=["teamMeetings", "teamMatches", "groupMeetings", "groupMatches"]):
        return href
    return team_page_url


def pending_results_last_48h(
    teams_same_league: List[TeamInfo],
    exclude_team_no: int,
) -> List[str]:
    """
    Looks at other Meyrin teams in same league level; if they have matches in last 48h with no final result,
    return warnings lines.
    """
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

        # Heuristic: match rows often in tables; we'll scan each row for a date and a "result-looking" token
        for tr in soup.find_all("tr"):
            txt = _norm(tr.get_text(" ", strip=True))
            dt = parse_date_from_text(txt)
            if not dt:
                continue
            if dt < cutoff or dt > now + datetime.timedelta(hours=2):
                continue

            # "not published" often shows "-:-" or blank-ish patterns
            # Also sometimes "vs" without score.
            if "-:-" in txt or ":-" in txt:
                warnings.append(f"Team {t.name}: match within last 48h might have unpublished result: {txt}")
            else:
                # If there is no obvious score like "8:2" etc., treat as suspicious
                if not re.search(r"\b\d+\s*:\s*\d+\b", txt):
                    warnings.append(f"Team {t.name}: recent match may not have published score yet: {txt}")

    # de-dup
    return list(dict.fromkeys(warnings))[:10]


# -----------------------------
# Simplified eligibility logic
# -----------------------------
def decide_eligibility(
    target: TeamInfo,
    nominated_team_no: Optional[int],
    teams_by_no: Dict[int, TeamInfo],
    apps_by_team: Dict[int, int],
) -> Tuple[bool, List[str]]:
    """
    Simplified rules (fast):
    Case 1) Licensed but never played -> can play anywhere.
    Case 2) If player has played in a league, cannot play another team of same league or below,
            EXCEPT they may still play in their own (base) lower team if they only played 1–2 times above.
    Case 3) 3rd appearance in a higher team -> warning: becomes titulaire there and loses right to play base team.
    """

    msgs: List[str] = []

    total_apps = sum(apps_by_team.values())

    # ---------------- Case 1 ----------------
    if total_apps == 0:
        return True, [
            "ELIGIBLE (Case 1): player is licensed but has not played yet this season in this category.",
            "They may play in any team for their first match.",
        ]

    played_teams = sorted([tno for tno, n in apps_by_team.items() if n > 0 and tno in teams_by_no])
    if not played_teams:
        # Shouldn't happen, but be safe
        return True, ["ELIGIBLE (Case 1 fallback): no played teams detected."]

    # Determine base/own team
    base_team_no = None
    if nominated_team_no is not None and nominated_team_no in teams_by_no:
        base_team_no = nominated_team_no
    else:
        # fallback: team with most apps
        base_team_no = max(played_teams, key=lambda tno: apps_by_team.get(tno, 0))

    base_rank = teams_by_no[base_team_no].league_level_rank
    target_rank = target.league_level_rank

    # Highest league level the player has played in (among all played teams)
    max_played_rank = max(teams_by_no[tno].league_level_rank for tno in played_teams)

    # How many times has the player played ABOVE their base team?
    higher_than_base_apps = sum(
        apps_by_team[tno]
        for tno in played_teams
        if teams_by_no[tno].league_level_rank > base_rank
    )

    # ---------------- Case 2 with exception ----------------
    # If target is same league as, or lower than, the highest league they have played:
    # generally NOT allowed (can't play same/below)...
    if target_rank <= max_played_rank:
        # ...EXCEPT: target is base team and they only played 1–2 times above base
        if target.team_no == base_team_no and higher_than_base_apps <= 2:
            msgs.append(
                f"Case 2 exception: player has only {higher_than_base_apps} appearance(s) above base team {base_team_no}, "
                f"so they may still play in their base team."
            )
        else:
            # find an example team at max rank for message clarity
            max_rank_team = None
            for tno in played_teams:
                if teams_by_no[tno].league_level_rank == max_played_rank:
                    max_rank_team = tno
                    break
            return False, [
                "NOT ELIGIBLE (Case 2): player already played in the same league level or higher; cannot play another team of the same league or below.",
                f"Highest league played so far includes team {max_rank_team} ({teams_by_no[max_rank_team].league_label}); "
                f"target is team {target.team_no} ({target.league_label}).",
                f"Base team is {base_team_no} ({teams_by_no[base_team_no].league_label}).",
            ]
    else:
        msgs.append("Passed Case 2: target is in a higher league than any league the player has played so far.")

    # ---------------- Case 3 ----------------
    # If nominated/base is lower and target is higher than base:
    # 3rd appearance in target -> titulaire warning (loses right to play base team)
    if base_team_no is not None:
        if target_rank > base_rank:
            already_in_target = apps_by_team.get(target.team_no, 0)
            next_count = already_in_target + 1
            if next_count == 3:
                msgs.append(
                    f"WARNING (Case 3): this would be the 3rd appearance in higher team {target.team_no} this season. "
                    f"Player becomes titulaire of the higher team and loses the right to play base team {base_team_no}."
                )
            elif next_count > 3:
                msgs.append(
                    f"WARNING (Case 3): player already has {already_in_target} appearances in higher team {target.team_no}. "
                    f"They should no longer play in base team {base_team_no}."
                )
            else:
                msgs.append(
                    f"Case 3 info: appearance #{next_count} in higher team {target.team_no} "
                    f"(titulaire switch happens at #3)."
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

teams = build_teams_for_phase(team_entries, ui_phase)
teams_by_no = {t.team_no: t for t in teams}

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

run_pending_check = st.checkbox("Also check last 48h pending results (warnings)", value=True)

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

    # FAST: appearances scan across Meyrin teams (current phase list). Cached.
    apps_by_team = fetch_player_apps_across_meyrin_teams(player_key, teams)

    ok, messages = decide_eligibility(
        target=target,
        nominated_team_no=nominated_team_no,
        teams_by_no=teams_by_no,
        apps_by_team=apps_by_team,
    )

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
            "nominated_team_no": nominated_team_no,
            "apps_by_team": apps_by_team,
            "target_team": f"{target.name} ({target.league_label})",
        }
    )

    # Case 4: recent pending results for other teams same league level
    if ok and run_pending_check:
        same_level_teams = [t for t in teams if t.league_level_rank == target.league_level_rank]
        warnings = pending_results_last_48h(same_level_teams, exclude_team_no=target.team_no)
        if warnings:
            st.warning("Recent matches (last 48h) in other teams of the same league may have unpublished results. Verify with the other team:")
            for w in warnings:
                st.write(f"- {w}")
