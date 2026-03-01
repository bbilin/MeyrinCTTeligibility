
# app.py — Meyrin CTT Eligibility Checker (match-day counting + season-aware paginated player search + 50.4.4 + 48h debug)
#
# Fixes in this version:
# ✅ Player search uses clubLicenceMembersPage WITH seasonName and follows pagination.
# ✅ Searches across all licence categories (Herren/Damen/Jeunesse/O40/Uxx) and auto-detects which categories matched.
# ✅ Category dropdown changes team list (Men vs Women teams).
# ✅ Appearances counted as TEAM MATCH-DAYS (meeting pages).
# ✅ 50.4.4 shown at end for replacements (target != base team) with roster+ranking list.
# ✅ Last-48h warnings + full debug (teams scanned, rows found, has_score).
# ✅ Optional "Search debug" shows how many players scanned per category and how many pages.

import re
import datetime
import unicodedata
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
session.headers.update({"User-Agent": "Meyrin-Eligibility-Checker/fast-3.2 (+club tool; public pages only)"})


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
    licence_categories: Tuple[str, ...]


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


def _fold(s: str) -> str:
    """Lowercase + remove accents + keep only letters/numbers/spaces."""
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def roman_to_int(token: str) -> Optional[int]:
    token = token.upper()
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    return mapping.get(token)


def infer_default_phase() -> str:
    m = datetime.date.today().month
    return "A" if 8 <= m <= 12 else "B"


def now_ch() -> datetime.datetime:
    if ZoneInfo:
        return datetime.datetime.now(ZoneInfo("Europe/Zurich"))
    return datetime.datetime.now()


def league_rank(label: str) -> int:
    """
    Higher number == higher league level.
    """
    s = (label or "").lower()
    s = re.sub(r"\bphase\s*[ab]\b", " ", s)
    s = re.sub(r"\bgr\.?\s*\d+\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if "nationalliga a" in s or "nla" in s:
        return 100
    if "nationalliga b" in s or "nlb" in s:
        return 90
    if "nationalliga c" in s or "nlc" in s:
        return 80

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


# -----------------------------
# Contest types config
# -----------------------------
CONTEST_TYPES: List[Tuple[str, str]] = [
    ("Men (Herren)", "${herren}"),
    ("Women (Damen)", "${damen}"),
    ("O40", "${o40}"),
    ("Junior (Jeunesse)", "${jugend}"),
    ("U19", "${u19}"),
    ("U15", "${u15}"),
    ("U13", "${u13}"),
]

CT_LABEL_BY_TOKEN = {token: label for (label, token) in CONTEST_TYPES}
TOKEN_BY_LABEL = {label: token for (label, token) in CONTEST_TYPES}


def team_kind_from_contest_type(contest_type_token: str) -> str:
    if contest_type_token == "${damen}":
        return "women"
    return "men"


# -----------------------------
# Teams (contestType-aware)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_meyrin_team_entries(team_kind: str) -> Dict[int, List[Tuple[str, str]]]:
    url = f"{BASE}/clubTeams?club={MEYRIN_CLUB_ID}"
    r = session.get(url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    def is_men_label(lbl: str) -> bool:
        tl = lbl.lower()
        return tl.startswith("men") or tl.startswith("hommes") or tl.startswith("herren")

    def is_women_label(lbl: str) -> bool:
        tl = lbl.lower()
        return tl.startswith("women") or tl.startswith("damen") or tl.startswith("femmes") or tl.startswith("ladies")

    entries: Dict[int, List[Tuple[str, str]]] = {}

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        team_label = _norm(tds[0].get_text(" ", strip=True))
        if not team_label:
            continue

        if team_kind == "men" and not is_men_label(team_label):
            continue
        if team_kind == "women" and not is_women_label(team_label):
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


def build_teams_for_phase(team_entries: Dict[int, List[Tuple[str, str]]], phase: str, prefix: str) -> List[TeamInfo]:
    teams: List[TeamInfo] = []
    for team_no, opts in sorted(team_entries.items()):
        label, url = pick_phase_entry(opts, phase)
        teams.append(TeamInfo(team_no, f"{prefix} {team_no}", label, url, league_rank(label)))
    return teams


def build_teams_for_both_phases(team_entries: Dict[int, List[Tuple[str, str]]], prefix: str) -> List[TeamInfo]:
    teams_all: List[TeamInfo] = []
    for ph in ["A", "B"]:
        teams_all.extend(build_teams_for_phase(team_entries, ph, prefix))
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
# Season-aware paginated player search
# -----------------------------
def _collect_players_from_licence_page(html: str) -> List[Tuple[str, str]]:
    """
    Returns list of (display_name, portrait_url) from a single page of clubLicenceMembersPage.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: List[Tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "playerPortrait" not in href or "person=" not in href:
            continue
        name_raw = _norm(a.get_text(" ", strip=True))
        if "," not in name_raw:
            continue
        out.append((name_raw, _abs_url(href)))
    return out


def _find_next_page_url(html: str) -> Optional[str]:
    """
    Best-effort: find a pagination 'next' link on click-tt pages.
    """
    soup = BeautifulSoup(html, "html.parser")
    # look for common next labels
    for a in soup.find_all("a", href=True):
        txt = _norm(a.get_text(" ", strip=True)).lower()
        if txt in ("next", "weiter", "suivant", ">"):
            return _abs_url(a["href"])
        if "next" in txt or "weiter" in txt or "suivant" in txt:
            return _abs_url(a["href"])
    # sometimes arrow images/buttons have no text; look for "page=" / "offset=" style in href
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(k in href.lower() for k in ["page=", "offset=", "start="]) and "clubLicenceMembersPage" in href:
            # not necessarily next; still helps for multi-page by exploring 2 pages max
            return _abs_url(href)
    return None


def fetch_all_licence_members_for_category(season_name: str, contest_type: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Downloads clubLicenceMembersPage for a given contestType and seasonName, following pagination.
    Returns (players, debug_info).
    """
    url = f"{BASE}/clubLicenceMembersPage"
    params = {
        "club": str(MEYRIN_CLUB_ID),
        "contestType": contest_type,
        "seasonName": season_name,
        "preferredLanguage": "German",
    }

    debug = {"contestType": contest_type, "pages": 0, "players": 0, "urls": []}
    all_players: List[Tuple[str, str]] = []
    seen_urls = set()

    try:
        r = session.get(url, params=params, timeout=25)
        r.raise_for_status()
    except Exception as e:
        debug["error"] = str(e)
        return [], debug

    html = r.text
    current_url = r.url
    # follow up to 5 pages max (safety)
    for _ in range(5):
        debug["pages"] += 1
        debug["urls"].append(current_url)

        page_players = _collect_players_from_licence_page(html)
        for name, purl in page_players:
            if purl in seen_urls:
                continue
            seen_urls.add(purl)
            all_players.append((name, purl))

        nxt = _find_next_page_url(html)
        if not nxt or nxt == current_url:
            break

        try:
            rr = session.get(nxt, timeout=25)
            rr.raise_for_status()
        except Exception:
            break

        current_url = rr.url
        html = rr.text

    debug["players"] = len(all_players)
    return all_players, debug


def search_player_in_meyrin_club(season_name: str, last: str, first: str) -> Tuple[List[PlayerPick], Dict[str, Any]]:
    """
    Search player across all licence categories for the given season, with pagination.
    Returns (picks, debug).
    """
    q_last = _fold(last)
    q_first = _fold(first)

    found: Dict[str, Dict[str, Any]] = {}  # portrait_url -> {"display": name, "cats": set()}
    dbg: Dict[str, Any] = {"seasonName": season_name, "by_category": []}

    for ct_label, ct_token in CONTEST_TYPES:
        players, d = fetch_all_licence_members_for_category(season_name, ct_token)
        d["categoryLabel"] = ct_label
        dbg["by_category"].append(d)

        for name_raw, purl in players:
            name_fold = _fold(name_raw)
            if q_last and q_last not in name_fold:
                continue
            if q_first and q_first not in name_fold:
                continue

            if purl not in found:
                found[purl] = {"display": name_raw, "cats": set()}
            found[purl]["cats"].add(ct_token)

    picks: List[PlayerPick] = []
    for purl, info in found.items():
        picks.append(PlayerPick(info["display"], purl, tuple(sorted(info["cats"]))))

    # rank results: more exact match first
    q = (q_last + " " + q_first).strip()

    def score(p: PlayerPick) -> int:
        n = _fold(p.display_name)
        if q and q in n:
            return 0
        if q_last and q_last in n:
            return 1
        return 2

    picks.sort(key=score)
    return picks, dbg


def infer_player_name_from_portrait(portrait_url: str) -> PlayerKey:
    r = session.get(portrait_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)

    for line in text.split("\n")[:120]:
        if "," in line and 3 < len(line) < 120:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                return PlayerKey(last=parts[0], first=parts[1])

    return PlayerKey(last="", first="")


# -----------------------------
# Ranking parsing (best-effort)
# -----------------------------
def _parse_ranking_token(s: str) -> Optional[str]:
    s = _norm(s).upper()
    m = re.search(r"\b([A-E])\s*([0-9]{1,2})\b", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return None


@st.cache_data(ttl=3600)
def fetch_player_ranking_from_portrait(portrait_url: str) -> Optional[str]:
    r = session.get(portrait_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    return _parse_ranking_token(text)


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
        try:
            r = session.get(url, params=params, timeout=25)
            r.raise_for_status()
        except Exception:
            continue

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
            try:
                rr = session.get(link, timeout=25)
                rr.raise_for_status()
            except Exception:
                continue
            team_no = scan_html(rr.text)
            if team_no is not None:
                return team_no

    return None


# -----------------------------
# Resolve team page + match list
# -----------------------------
@st.cache_data(ttl=3600)
def find_team_page_from_league(league_url: str, team_no: int, club_prefix: str) -> Optional[str]:
    if not league_url:
        return None
    r = session.get(league_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    roman = ROMAN.get(team_no, str(team_no))
    labels = [f"{club_prefix} {roman}", f"{club_prefix} {team_no}"]

    for a in soup.find_all("a", href=True):
        txt = _norm(a.get_text(" ", strip=True))
        if any(lab.lower() in txt.lower() for lab in labels):
            return _abs_url(a["href"])
    return None


def find_team_match_list_url(team_page_url: str) -> str:
    r = session.get(team_page_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for href in _find_links(soup, include=["teamMeetings", "teamMatches", "groupMeetings", "groupMatches"]):
        return href
    return team_page_url


# -----------------------------
# Match-day counting
# -----------------------------
@st.cache_data(ttl=900)
def count_player_matchdays_for_team(team_page_url: str, player: PlayerKey, max_meetings: int = 40) -> int:
    target1 = f"{player.last}, {player.first}".lower()
    target2 = f"{player.first} {player.last}".lower()

    list_url = find_team_match_list_url(team_page_url)
    r = session.get(list_url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

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
def fetch_player_apps_across_club_teams_matchdays(
    player: PlayerKey,
    teams: List[TeamInfo],
    club_prefix: str,
    max_meetings_per_team: int = 40,
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for t in teams:
        team_page = find_team_page_from_league(t.league_url, t.team_no, club_prefix=club_prefix)
        if not team_page:
            continue
        n = count_player_matchdays_for_team(team_page, player, max_meetings=max_meetings_per_team)
        if n > 0:
            out[t.team_no] = out.get(t.team_no, 0) + n
    return out


# -----------------------------
# 50.4.4 roster listing
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_team_roster_with_rankings(team_page_url: str) -> List[Dict[str, str]]:
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

        for tr in ss.find_all("tr"):
            row_txt = _norm(tr.get_text(" ", strip=True))
            if "," not in row_txt:
                continue
            m = re.search(r"([A-Za-zÀ-ÿ'\- ]+,\s*[A-Za-zÀ-ÿ'\- ]+)", row_txt)
            if not m:
                continue
            name = _norm(m.group(1))
            if name in seen_names:
                continue
            seen_names.add(name)
            rank = _parse_ranking_token(row_txt) or ""
            roster.append({"name": name, "rank": rank})
    return roster


# -----------------------------
# Last 48h scan + debug
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
    club_prefix: str,
) -> Tuple[List[str], Dict[str, Any]]:
    now = now_ch()
    cutoff = now - datetime.timedelta(hours=48)

    warnings: List[str] = []
    debug: Dict[str, Any] = {}

    for t in teams_same_league:
        if t.team_no == exclude_team_no:
            continue

        debug[t.name] = {"list_url": None, "found": []}

        team_page = find_team_page_from_league(t.league_url, t.team_no, club_prefix=club_prefix)
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
# Eligibility logic (Case 3 before Case 2)
# -----------------------------
def decide_eligibility(
    target: TeamInfo,
    nominated_team_no: Optional[int],
    teams_by_no: Dict[int, TeamInfo],
    apps_by_team: Dict[int, int],
) -> Tuple[bool, List[str], int]:
    msgs: List[str] = []
    total_apps = sum(apps_by_team.values())

    if total_apps == 0:
        return True, [
            "ELIGIBLE (Case 1): player has not played yet this season.",
            "First appearance may be in any team.",
        ], -1

    played_teams = [tno for tno, n in apps_by_team.items() if n > 0 and tno in teams_by_no]
    if not played_teams:
        return True, ["ELIGIBLE (fallback): no valid played teams detected."], -1

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

    # Case 3 first: playing UP
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

    # Case 2: same league or down
    if target_rank <= max_played_rank:
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
# UI
# -----------------------------
st.set_page_config(page_title="Meyrin CTT – Eligibility Checker", layout="wide")
st.title("Meyrin CTT – Eligibility Checker")

with st.sidebar:
    st.header("Season / Category")
    season_name = st.text_input("Season (format 2025/26)", value="2025/26")

    contest_choice = st.selectbox("Category (contestType)", options=[c[0] for c in CONTEST_TYPES])
    contest_type = TOKEN_BY_LABEL[contest_choice]

    st.header("Phase (for team dropdown)")
    ui_phase = st.selectbox("Phase", options=["A", "B"], index=["A", "B"].index(infer_default_phase()))

team_kind = team_kind_from_contest_type(contest_type)
team_entries = fetch_meyrin_team_entries(team_kind=team_kind)

club_prefix = "Meyrin"
teams_ui = build_teams_for_phase(team_entries, ui_phase, prefix=club_prefix)
teams_by_no = {t.team_no: t for t in teams_ui}

with st.sidebar:
    st.header("Target team")
    if not teams_ui:
        st.error("No teams found for this category on click-tt clubTeams page.")
        st.stop()
    target = st.selectbox("Team", options=teams_ui, format_func=lambda t: f"{t.name} — {t.league_label}")

st.divider()
st.subheader("Player")

c1, c2 = st.columns(2)
with c1:
    last = st.text_input("Last name", value="")
with c2:
    first = st.text_input("First name", value="")

run_pending_check = st.checkbox("Check last 48h unpublished results (warnings)", value=True)
show_last48_debug = st.checkbox("Show last 48h debug (teams + matches found)", value=False)
show_search_debug = st.checkbox("Show player search debug (pages scanned per category)", value=False)

max_meetings = st.slider("Max match-days to scan per team+phase", min_value=10, max_value=60, value=35, step=5)
show_5044 = st.checkbox("Show 50.4.4 ranking/roster info at end (replacement only)", value=True)

if st.button("Check eligibility"):
    if not last.strip() or not first.strip():
        st.error("Please enter both last name and first name.")
        st.stop()

    picks, search_dbg = search_player_in_meyrin_club(season_name=season_name, last=last.strip(), first=first.strip())

    if show_search_debug:
        st.markdown("### Debug — player search")
        st.json(search_dbg)

    if not picks:
        st.error("Player not found in Meyrin licence lists (season-aware, paginated, searched across Herren/Damen/Jeunesse/O40/Uxx).")
        st.stop()

    pick = picks[0]
    if len(picks) > 1:
        pick = st.radio(
            "Multiple matches found — pick one:",
            options=picks,
            format_func=lambda p: f"{p.display_name}  —  licences: {', '.join(CT_LABEL_BY_TOKEN.get(x, x) for x in p.licence_categories)}",
        )

    player_key = infer_player_name_from_portrait(pick.portrait_url)
    if not player_key.last or not player_key.first:
        player_key = PlayerKey(last=last.strip(), first=first.strip())

    licence_labels = [CT_LABEL_BY_TOKEN.get(x, x) for x in pick.licence_categories]

    # 50.4.2 informational note
    if contest_type == "${herren}" and "${damen}" in pick.licence_categories and "${herren}" not in pick.licence_categories:
        st.info("Detected **Damen** licence. In men category, **dames can also play** (commonly ref. 50.4.2).")

    nominated_team_no = fetch_regular_registration_nominated_team(season_name, contest_type, player_key)

    # match-day appearances across BOTH phases
    teams_both = build_teams_for_both_phases(team_entries, prefix=club_prefix)
    apps_by_team = fetch_player_apps_across_club_teams_matchdays(
        player_key,
        teams_both,
        club_prefix=club_prefix,
        max_meetings_per_team=max_meetings,
    )

    ok, messages, base_team_no = decide_eligibility(target, nominated_team_no, teams_by_no, apps_by_team)
    player_rank = fetch_player_ranking_from_portrait(pick.portrait_url) or ""

    st.markdown("### Result")
    st.success(messages[0]) if ok else st.error(messages[0])

    st.markdown("### Reasoning / Warnings")
    for m in messages[1:]:
        st.write(f"- {m}")

    st.markdown("### Details")
    st.write(
        {
            "player": f"{player_key.last}, {player_key.first}",
            "portrait": pick.portrait_url,
            "player_ranking": player_rank or "unknown",
            "licence_categories_detected": licence_labels,
            "selected_category": contest_choice,
            "nominated_team_no": nominated_team_no,
            "base_team_no": (base_team_no if base_team_no != -1 else None),
            "apps_by_team_matchdays": apps_by_team,
            "target_team": f"{target.name} ({target.league_label})",
        }
    )

    # Case 4 warning + debug
    last48_debug_data = None
    if ok and run_pending_check:
        same_level_teams = [t for t in teams_ui if t.league_level_rank == target.league_level_rank]
        warnings, dbg = pending_results_last_48h_with_debug(
            same_level_teams,
            exclude_team_no=target.team_no,
            club_prefix=club_prefix,
        )
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
        st.json(last48_debug_data or {})

    # 50.4.4 at the end (replacement only)
    if show_5044 and ok:
        is_replacement = (base_team_no != -1 and target.team_no != base_team_no)
        if is_replacement:
            st.markdown("### 50.4.4 — Remplacement / Ranking check (manual verification)")
            st.write("Rule 50.4.4: **Le classement du joueur remplaçant ne peut être supérieur à celui du joueur titulaire qu'il remplace.**")
            st.write(f"- Replacement player: **{player_key.last}, {player_key.first}** — ranking: **{player_rank or 'unknown'}**")

            team_page = find_team_page_from_league(target.league_url, target.team_no, club_prefix=club_prefix)
            roster = fetch_team_roster_with_rankings(team_page) if team_page else []

            if roster:
                st.write(f"Target team roster (best-effort from click-tt): **{target.name}**")
                st.dataframe(roster, use_container_width=True)
                st.info("⚠️ Verify the **specific titulaire being replaced** has ranking ≥ replacement player's ranking.")
            else:
                st.warning("Could not extract target team roster/rankings automatically; open the team page and verify manually.")
