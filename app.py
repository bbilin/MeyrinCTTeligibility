import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import streamlit as st

BASE = "https://www.click-tt.ch/cgi-bin/WebObjects/nuLigaTTCH.woa/wa"
MEYRIN_CLUB_ID = 33165

# Substitute markers seen on bilans pages (varies by language/competition)
SUB_MARKERS = {"S", "E", "V"}

session = requests.Session()
session.headers.update(
    {"User-Agent": "Meyrin-Eligibility-Checker/1.1 (+local tool; no automation; contact club admin)"}
)

# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class TeamInfo:
    team_no: int
    name: str
    league_label: str


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
    """
    Higher number == higher league.
    Tune mapping if needed.
    """
    s = (label or "").lower()

    # National leagues
    if "nla" in s:
        return 100
    if "nlb" in s:
        return 90
    if "nlc" in s:
        return 80

    # French labels / German labels
    # "Ligue 1" .. "Ligue 6"  / "1. Liga" .. "6. Liga"
    m = re.search(r"\b(?:ligue|liga)\s*(\d+)\b", s)
    if m:
        n = int(m.group(1))
        # Ligue 1 is higher than Ligue 5 -> invert
        return 70 - n  # Ligue 1 -> 69, Ligue 5 -> 65

    # fallback
    return 0


# -----------------------------
# Click-tt scraping
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_meyrin_teams() -> List[TeamInfo]:
    """
    Parses Meyrin clubTeams page where teams are listed as:
      Men, Hommes II, Hommes III, ... (plus O40, Jeunesse, Cup, etc.)
    We keep only men's league teams (Men/Hommes/Herren) and ignore Cup rows.
    """
    url = f"{BASE}/clubTeams?club={MEYRIN_CLUB_ID}"
    r = session.get(url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    def roman_to_int(token: str) -> Optional[int]:
        token = token.upper()
        roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
        return roman.get(token)

    found: Dict[int, TeamInfo] = {}

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        team_label = _norm(tds[0].get_text(" ", strip=True))
        if not team_label:
            continue

        tl = team_label.lower()
        # Keep "Men", "Hommes II", "Herren ..." (but we'll exclude Cup below)
        if not (tl.startswith("men") or tl.startswith("hommes") or tl.startswith("herren")):
            continue

        # Exclude Swiss Cup rows ("Hauptrunde", etc.)
        league_text = _norm(tds[1].get_text(" ", strip=True)).lower()
        if "hauptrunde" in league_text or "cup" in league_text:
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

        # League label is usually a link in column 2
        a = tds[1].find("a")
        league_label = _norm(a.get_text(" ", strip=True)) if a else _norm(tds[1].get_text(" ", strip=True))

        # dedupe: keep the "best" league label (by rank) if duplicated Phase A/B rows
        candidate = TeamInfo(team_no=team_no, name=f"Meyrin {team_no}", league_label=league_label)
        if team_no not in found or league_rank(candidate.league_label) > league_rank(found[team_no].league_label):
            found[team_no] = candidate

    return [found[k] for k in sorted(found.keys())]

def search_player_in_meyrin_club(last: str, first: str) -> List[PlayerPick]:
    """
    Uses Meyrin 'Licenced players' page which contains direct links to player portraits.
    Much more reliable than eloFilter on hosted environments.
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
        # "Bilin, Bugra"
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

def search_player_portraits(last: str, first: str) -> List[PlayerPick]:
    """
    Uses the public Elo-Filter page and extracts playerPortrait links.
    Param names may change; if click-tt changes it, this function is the only one to adjust.
    """
    url = f"{BASE}/eloFilter?federation=STT&preferredLanguage=German"
    params = {"Nachname": last, "Vorname": first}
    r = session.get(url, params=params, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    picks: List[PlayerPick] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "playerPortrait" in href and "person=" in href:
            display = _norm(a.get_text(" ", strip=True))
            picks.append(PlayerPick(display_name=display, portrait_url=_abs_url(href)))

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
    We use this to match names against registration/bilans lists.
    """
    r = session.get(portrait_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Usually contains "Lastname, Firstname" somewhere prominent
    text = soup.get_text("\n", strip=True)
    # heuristic: find first line with comma and two words
    for line in text.split("\n")[:40]:
        if "," in line and len(line) < 60:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                return PlayerKey(last=parts[0], first=parts[1])
    # fallback: unknown
    return PlayerKey(last="", first="")


#def fetch_regular_registration_nominated_team(
#    season_name: str,
#    contest_type_token: str,
#    player: PlayerKey,
#) -> Optional[int]:
#    """
#    Returns team_no if player is nominated (inscrit nominativement) in Regular Player Registration.
#    We scrape clubPools (Regular Player Registrationen und Bilanzen) and look for "X.Y Last, First".
#    """
#    url = f"{BASE}/clubPools"
#    params = {
#        "club": str(MEYRIN_CLUB_ID),
#        "contestType": contest_type_token,   # like ${herren}
#        "displayTyp": "vorrunde",
#        "preferredLanguage": "German",
#        "seasonName": season_name,
#    }
#    r = session.get(url, params=params, timeout=25)
#    r.raise_for_status()
#
#    soup = BeautifulSoup(r.text, "html.parser")
#    text = soup.get_text("\n", strip=True)
#
#    # match "7.1 Bilin, Bugra" etc
#    # allow either "Last, First" or "First Last" depending on the page
#    target1 = f"{player.last}, {player.first}".lower()
#    target2 = f"{player.first} {player.last}".lower()
#
#    for line in text.split("\n"):
#        s = _norm(line)
#        m = re.match(r"^(\d+)\.(\d+)\s+(.*)$", s)
#        if not m:
#            continue
#        team_no = int(m.group(1))
#        rest = m.group(3).lower()
#        if target1 in rest or target2 in rest:
#            return team_no
#
#    return None

def fetch_regular_registration_nominated_team(
    season_name: str,
    contest_type_token: str,
    player: PlayerKey,
) -> Optional[int]:
    """
    Robustly parses clubPools page tables.
    Looks for rows with first cell like '7.1' and a name cell containing 'Last, First'.
    """
    url = f"{BASE}/clubPools"
    params = {
        "club": str(MEYRIN_CLUB_ID),
        "contestType": contest_type_token,   # e.g. ${herren}
        "displayTyp": "vorrunde",
        "preferredLanguage": "German",
        "seasonName": season_name,
    }
    r = session.get(url, params=params, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    target_last = player.last.strip().lower()
    target_first = player.first.strip().lower()

    # Scan all tables for rank+name patterns
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            rank_txt = _norm(tds[0].get_text(" ", strip=True))
            m = re.match(r"^(\d+)\.\d+$", rank_txt)  # '7.1' -> team 7
            if not m:
                continue
            team_no = int(m.group(1))

            name_txt = _norm(tds[1].get_text(" ", strip=True)).lower()
            # handle "Bilin, Bugra" and "Bugra Bilin"
            if (target_last in name_txt and target_first in name_txt):
                return team_no

    # Optional: try Rückrunde too (some seasons only filled there)
    params["displayTyp"] = "rueckrunde"
    r = session.get(url, params=params, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            rank_txt = _norm(tds[0].get_text(" ", strip=True))
            m = re.match(r"^(\d+)\.\d+$", rank_txt)
            if not m:
                continue
            team_no = int(m.group(1))
            name_txt = _norm(tds[1].get_text(" ", strip=True)).lower()
            if (target_last in name_txt and target_first in name_txt):
                return team_no

    return None

def fetch_bilans_apps(
    season_name: str,
    contest_type_token: str,
    player: PlayerKey,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Returns:
      apps[team_no] = total Einsätze for player in that team
      sub_apps[team_no] = Einsätze counted as substitute in that team (row starts with S/E/V)
    """
    url = f"{BASE}/clubPortraitTT"
    params = {
        "club": str(MEYRIN_CLUB_ID),
        "contestType": contest_type_token,
        "displayTyp": "gesamt",
        "preferredLanguage": "German",
        "seasonName": season_name,
    }
    r = session.get(url, params=params, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Strategy: walk headings and tables; identify which team section we're in by "Meyrin VI" heading text.
    apps: Dict[int, int] = {}
    sub_apps: Dict[int, int] = {}

    # Build a flat list of elements in order, tracking current team_no
    current_team_no: Optional[int] = None

    # target name patterns
    target_last_first = f"{player.last}, {player.first}".lower()
    target_first_last = f"{player.first} {player.last}".lower()

    for el in soup.find_all(["h1", "h2", "h3", "table"]):
        if el.name in {"h1", "h2", "h3"}:
            htxt = _norm(el.get_text(" ", strip=True))
            if htxt.lower().startswith("meyrin"):
                tno = _parse_team_no_from_name(htxt)
                if tno:
                    current_team_no = tno

        if el.name == "table" and current_team_no is not None:
            # parse rows
            for tr in el.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue
                row_text = _norm(tr.get_text(" ", strip=True)).lower()
                if target_last_first not in row_text and target_first_last not in row_text:
                    continue

                # Try find "Einsätze" number anywhere in row
                # Commonly the last numeric field; take max integer found
                ints = [int(x) for x in re.findall(r"\b(\d+)\b", tr.get_text(" ", strip=True))]
                if not ints:
                    continue
                count = max(ints)

                # Determine marker: first cell might be "7.1" (regular) or "S" / "E" / "V"
                first_cell = _norm(tds[0].get_text(" ", strip=True))
                marker = first_cell.upper()

                apps[current_team_no] = count
                if marker in SUB_MARKERS:
                    sub_apps[current_team_no] = count

    return apps, sub_apps


# -----------------------------
# Rules engine: 50.4.5 / 50.4.6 / 50.4.7
# -----------------------------
def check_eligibility(
    target_team: TeamInfo,
    teams_by_no: Dict[int, TeamInfo],
    nominated_team_no: Optional[int],
    apps: Dict[int, int],
    sub_apps: Dict[int, int],
) -> Tuple[bool, List[str]]:

    reasons: List[str] = []

    # Determine which teams player has already played for (in this series/contestType)
    played_teams = sorted([t for t, n in apps.items() if n > 0])

    # Map league ranks for Meyrin teams
    lr: Dict[int, int] = {}
    for tno, info in teams_by_no.items():
        lr[tno] = league_rank(info.league_label)

    # lowest team in this contestType among Meyrin teams
    # (lowest rank -> smallest value)
    lowest_team_no = min(lr.keys(), key=lambda k: lr[k]) if lr else target_team.team_no

    def higher(a: int, b: int) -> bool:
        return lr.get(a, 0) > lr.get(b, 0)

    # ---- 50.4.5: max two teams AND in different leagues ----
    # prospective teams if we add target team
    prospective = sorted(set(played_teams + [target_team.team_no]))
    if len(prospective) > 2:
        return False, [
            "NOT ELIGIBLE (50.4.5): a player may be aligned in at most two teams.",
            f"Already played for teams {played_teams}; adding team {target_team.team_no} would make {prospective}."
        ]

    if len(prospective) == 2:
        t1, t2 = prospective
        l1 = teams_by_no.get(t1).league_label if t1 in teams_by_no else "?"
        l2 = teams_by_no.get(t2).league_label if t2 in teams_by_no else "?"
        if league_rank(l1) == league_rank(l2):
            return False, [
                "NOT ELIGIBLE (50.4.5): the two teams must be in different leagues.",
                f"Teams would be {t1} ({l1}) and {t2} ({l2}) which are the same league level."
            ]

    reasons.append("Passed 50.4.5 (max two teams, different leagues).")

    # Helper: if this appearance is a substitute appearance in target team (generally yes unless nominated there)
    def next_sub_count(team_no: int) -> int:
        return sub_apps.get(team_no, 0) + 1

    # ---- 50.4.7 (special): non-nominated, first appearance in lowest team -> becomes titulaire immediately ----
    if nominated_team_no is None:
        if apps.get(lowest_team_no, 0) >= 1:
            # already became titulaire in lowest team from first alignment (50.4.7)
            reasons.append(f"50.4.7 applies: player has appeared in lowest team {lowest_team_no}, thus is titulaire there.")
        # If target is lowest and would be first appearance: becoming titulaire is allowed (no prohibition).
        if target_team.team_no == lowest_team_no and apps.get(lowest_team_no, 0) == 0:
            reasons.append(f"50.4.7: first alignment in lowest team {lowest_team_no} makes player titulaire there.")
            return True, ["ELIGIBLE"] + reasons

    # ---- 50.4.6 (nominated case): substitute in higher league max 2; 3rd makes titulaire there ----
    if nominated_team_no is not None:
        base = nominated_team_no
        if target_team.team_no == base:
            reasons.append(f"Player is nominated in team {base}; playing there is always OK under these articles.")
            return True, ["ELIGIBLE"] + reasons

        if higher(target_team.team_no, base):
            n = next_sub_count(target_team.team_no)
            if n <= 2:
                reasons.append(
                    f"50.4.6: nominated in lower team {base}; this is substitute appearance #{n} in higher team {target_team.team_no} (allowed up to 2)."
                )
                return True, ["ELIGIBLE"] + reasons
            else:
                reasons.append(
                    f"50.4.6: this is substitute appearance #{n} in higher team {target_team.team_no}; "
                    f"player becomes titulaire of the higher team and must then play only for that team in this series."
                )
                return True, ["ELIGIBLE (but triggers titular lock to higher team)"] + reasons

        # Same or lower league than base: 50.4.6 does not add a restriction here.
        reasons.append("50.4.6: target team is not a higher league than the nominated team; no extra restriction here.")
        return True, ["ELIGIBLE"] + reasons

    # ---- 50.4.6 (non-nominated): can be aligned in two teams of different leagues; 3rd appearance in one -> titulaire there ----
    # (50.4.5 already enforced max 2 teams and different leagues)
    current_apps = apps.get(target_team.team_no, 0)
    n_after = current_apps + 1
    if n_after >= 3:
        reasons.append(
            f"50.4.6: this would be appearance #{n_after} in team {target_team.team_no}; player becomes titulaire of that team."
        )
        return True, ["ELIGIBLE (but becomes titulaire of the team)"] + reasons

    reasons.append(f"50.4.6: not nominated; this would be appearance #{n_after} in team {target_team.team_no} (still within substitute limits).")
    return True, ["ELIGIBLE"] + reasons


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Meyrin CTT – Eligibility Checker", layout="wide")
st.title("Meyrin CTT – Eligibility Checker (50.4.5 / 50.4.6 / 50.4.7)")

teams = fetch_meyrin_teams()
teams_by_no = {t.team_no: t for t in teams}

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
    target = st.selectbox("Meyrin team", options=teams, format_func=lambda t: f"{t.name} — {t.league_label}")

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
        st.error("No player portraits found via Elo-Filter search. Try different spelling.")
        st.stop()

    pick = picks[0]
    if len(picks) > 1:
        pick = st.radio(
            "Multiple matches found — pick one:",
            options=picks,
            format_func=lambda p: f"{p.display_name} — {p.portrait_url}",
        )

    player_key = infer_player_name_from_portrait(pick.portrait_url)
    if not player_key.last or not player_key.first:
        # fallback to entered values
        player_key = PlayerKey(last=last.strip(), first=first.strip())

    # Scrape nominated team + bilans
    nominated_team_no = fetch_regular_registration_nominated_team(season_name, contest_type, player_key)
    apps, sub_apps = fetch_bilans_apps(season_name, contest_type, player_key)

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
    st.write({
        "player": f"{player_key.last}, {player_key.first}",
        "portrait": pick.portrait_url,
        "nominated_team_no": nominated_team_no,
        "apps": apps,
        "sub_apps": sub_apps,
        "target_team": f"{target.name} ({target.league_label})",
    })

    st.markdown("### Rule reasoning")
    for r in reasons[1:]:
        st.write(f"- {r}")

st.sidebar.markdown("### Debug click-tt access")

if st.sidebar.button("Test click-tt"):
    try:
        test_url = f"{BASE}/clubTeams?club={MEYRIN_CLUB_ID}"
        r = session.get(test_url, timeout=20)
        st.sidebar.write("Status:", r.status_code)
        st.sidebar.write("Length:", len(r.text))
        st.sidebar.text(r.text[:300])
    except Exception as e:
        st.sidebar.write("Error:", repr(e))

