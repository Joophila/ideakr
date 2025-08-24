# -*- coding: utf-8 -*-
"""
ideakr daily updater (KST midnight via GitHub Actions)

Outputs:
- data/rawitems.json     : 표준화 원천 아이템
- data/signals.json      : 동적 키워드 그룹 + 상위 용어
- data/trends.json       : Naver DataLab 시계열(선택된 그룹)
- data/ideas.json        : 오늘의 아이디어 카드(전체 섹션 + 점수 + 커뮤니티)

Notes:
- 근거 부족/불확실은 '알 수 없습니다' 또는 '확실하지 않음'으로 표기
- 추정치가 포함될 때는 '추측한 내용입니다'를 명시
"""

import json, os, re, time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import requests, feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dtp

# ------------ Config ------------
KST = timezone(timedelta(hours=9))
NOW_UTC = datetime.now(timezone.utc)
NOW_KST = NOW_UTC.astimezone(KST)

WINDOW_HOURS = 72
DATALAB_DAYS = 14     # 그래프가 보기 좋아지도록 14일
TOPICS = ["BUSINESS", "SCIENCE", "TECHNOLOGY"]
REDDIT_SUBS = ["Entrepreneur","startups","smallbusiness","technology","Futurology"]

MAX_GNEWS = 60        # per topic
MAX_REDDIT = 12       # per subreddit
MAX_GROUPS = 8
MIN_TERM_COUNT = 3

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "rawitems.json")
SIGNALS_PATH = os.path.join(DATA_DIR, "signals.json")
TRENDS_PATH = os.path.join(DATA_DIR, "trends.json")
IDEAS_PATH = os.path.join(DATA_DIR, "ideas.json")

POLITICS_BLOCK = ["대통령","총선","국회","정치","여당","야당","민주당","국민의힘","선거","의회","외교","북한"]
STOPWORDS = set(["속보","뉴스","단독","영상","사진","기자","오늘","이번","관련"])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# ------------ Utils ------------
def strip_html(s): 
    return "" if not s else BeautifulSoup(s, "html.parser").get_text(" ", strip=True)

def as_iso(dt):
    if isinstance(dt, str):
        try: dt = dtp.parse(dt)
        except Exception: return None
    if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def is_recent(iso_str, hours=WINDOW_HOURS):
    try:
        d = dtp.parse(iso_str)
        if not d.tzinfo: d = d.replace(tzinfo=timezone.utc)
        return d >= datetime.now(timezone.utc) - timedelta(hours=hours)
    except Exception:
        return False

def blocked(text):
    t = (text or "").lower()
    return any(k.lower() in t for k in POLITICS_BLOCK)

def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        pass
    if old == obj:
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return True

# ------------ Collectors ------------
def fetch_gnews_topic(topic):
    url = f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:MAX_GNEWS]:
        title = e.get("title","") or ""
        link = e.get("link","") or ""
        desc = strip_html(e.get("summary") or e.get("description") or "")
        pub = e.get("published") or e.get("updated")
        pub_iso = as_iso(pub) if pub else None
        if not pub_iso or not is_recent(pub_iso): 
            continue
        if blocked(title) or blocked(desc): 
            continue
        out.append({
            "title": title,
            "url": link,
            "published_at": pub_iso,
            "description": desc,
            "source_name": f"GoogleNews:{topic}",
            "source_url": "",
            "topic": topic
        })
    return out

def fetch_reddit(sub):
    url = f"https://www.reddit.com/r/{sub}/top.json"
    qs = {"t":"week","limit":MAX_REDDIT}
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        r = requests.get(url, params=qs, headers=headers, timeout=20)
        r.raise_for_status()
    except Exception:
        return []
    js = r.json()
    out = []
    for ch in (js.get("data",{}) or {}).get("children",[]):
        d = ch.get("data",{}) or {}
        title = d.get("title","") or ""
        desc  = d.get("selftext","") or ""
        created = d.get("created_utc")
        if not created: 
            continue
        pub_iso = as_iso(datetime.fromtimestamp(created, tz=timezone.utc))
        if not is_recent(pub_iso): 
            continue
        if blocked(title) or blocked(desc):
            continue
        permalink = d.get("permalink")
        url = f"https://www.reddit.com{permalink}" if permalink else (d.get("url","") or "")
        out.append({
            "title": title,
            "url": url,
            "published_at": pub_iso,
            "description": desc,
            "source_name": f"reddit:{sub}",
            "source_url": f"https://www.reddit.com/r/{sub}",
            "topic": "REDDIT",
            # 커뮤니티 시그널
            "ups": int(d.get("ups") or 0),
            "num_comments": int(d.get("num_comments") or 0)
        })
    return out

def collect_all():
    items=[]
    for t in TOPICS:
        items.extend(fetch_gnews_topic(t)); time.sleep(0.3)
    for sub in REDDIT_SUBS:
        items.extend(fetch_reddit(sub)); time.sleep(0.3)
    # dedupe by URL
    seen=set(); uniq=[]
    for it in items:
        key=(it.get("url") or "").strip().lower().rstrip("/")
        if not key or key in seen: 
            continue
        seen.add(key); uniq.append(it)
    return uniq

# ------------ Keyword groups ------------
def tokenize_ko_en(s):
    s=(s or "").lower()
    s=re.sub(r"&[a-z0-9#]+;"," ", s)
    s=re.sub(r"[^가-힣a-z0-9\- ]"," ", s)
    s=re.sub(r"\s+"," ", s).strip()
    return s.split(" ") if s else []

def build_keyword_groups(items, max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT):
    cnt={}
    for it in items:
        text=f"{it.get('title','')} {it.get('description','')}"
        toks=[w for w in tokenize_ko_en(strip_html(text)) if len(w)>=2 and w not in STOPWORDS]
        for w in toks:
            cnt[w]=cnt.get(w,0)+1
        for i in range(len(toks)-1):
            bg=f"{toks[i]} {toks[i+1]}"
            cnt[bg]=cnt.get(bg,0)+1
    terms=sorted([(k,v) for k,v in cnt.items() if v>=min_count], key=lambda x:x[1], reverse=True)[:80]
    groups=[]; used=set()
    for term,_ in terms:
        if len(groups)>=max_groups: break
        if term in used: continue
        groups.append({"groupName":term, "keywords":[term]})
        used.add(term)
    return {"keywordGroups":groups, "topTerms":terms, "itemCount":len(items)}

# ------------ Naver DataLab ------------
def naver_datalab(keyword_groups, days=DATALAB_DAYS):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"error":"missing_keys","results":[]}
    end_kst = NOW_KST.date()
    start_kst = (NOW_KST - timedelta(days=days)).date()
    body = {
        "startDate": str(start_kst),
        "endDate": str(end_kst),
        "timeUnit": "date",
        "keywordGroups": [{"groupName": g["groupName"], "keywords": g["keywords"][:3]} for g in keyword_groups[:MAX_GROUPS]],
        "device": "",
        "ages": [],
        "gender": ""
    }
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "Content-Type": "application/json; charset=utf-8",
    }
    try:
        r = requests.post("https://openapi.naver.com/v1/datalab/search", headers=headers, data=json.dumps(body), timeout=20)
        if r.status_code != 200:
            return {"error": f"http_{r.status_code}", "results":[]}
        return r.json()
    except Exception:
        return {"error":"exception","results":[]}

def extract_trend_metrics(datalab):
    """DataLab results → 각 그룹별 volume(마지막), growth(처음~마지막 %)"""
    metrics=[]
    for g in datalab.get("results",[]):
        data=g.get("data",[])
        if not data: 
            metrics.append({"groupName":g.get("title",""),"volume":0,"growth":"알 수 없습니다"})
            continue
        first=data[0]["ratio"]; last=data[-1]["ratio"]
        growth = "알 수 없습니다"
        try:
            growth = round(((last - first) / (first if first else 1)) * 100, 1)
        except Exception:
            pass
        metrics.append({"groupName":g.get("title",""), "volume": int(last), "growth": growth})
    return metrics

# ------------ Scoring (데이터 기반, 보수적) ------------
def score_from_metrics(metrics, items):
    # Trend: volume+growth 평균 → 0~100 스케일
    if metrics:
        vols=[m["volume"] for m in metrics if isinstance(m.get("volume"), (int,float))]
        grs=[m["growth"] for m in metrics if isinstance(m.get("growth"), (int,float))]
        v = sum(vols)/len(vols) if vols else 0
        g = sum(grs)/len(grs) if grs else 0
        trend = max(0, min(100, int(v*0.5 + g*1.0)))  # 단순 휴리스틱 (추측한 내용입니다)
    else:
        trend = 0

    # Market: 최근 72h 기사수로 근사 (추측)
    market = max(0, min(100, len(items)*2))  # 50개 기사≈100 (추측한 내용입니다)

    # Competition invert: 중복 도메인 적을수록 높게 (추측)
    domains={}
    for it in items:
        u=(it.get("url") or "").lower()
        host=u.split("/")[2] if u.startswith("http") and len(u.split("/"))>2 else ""
        if host: domains[host]=domains.get(host,0)+1
    avg = sum(domains.values())/len(domains) if domains else 0
    competition_invert = max(0, min(100, int(100 - min(90, avg*10))))  # (추측한 내용입니다)

    # Monetization/Feasibility/Regulatory는 확실한 근거 부족 → 50 기준
    monetization = 50
    feasibility = 50
    regulatory_invert = 50

    overall = int(0.35*trend + 0.25*market + 0.15*competition_invert + 0.1*monetization + 0.15*feasibility)  # (추측)

    notes = {
        "trend":"Naver DataLab volume/growth 기반",
        "market":"최근 72시간 기사 수를 이용한 근사치(추측한 내용입니다)",
        "competition_invert":"출처 도메인 다양성으로 근사(추측한 내용입니다)",
        "feasibility":"근거 부족: 50으로 고정",
        "monetization":"근거 부족: 50으로 고정",
        "regulatory_invert":"근거 부족: 50으로 고정"
    }
    return {
        "trend":trend, "market":market, "competition_invert":competition_invert,
        "feasibility":feasibility, "monetization":monetization, "regulatory_invert":regulatory_invert,
        "overall":overall, "notes":notes
    }

# ------------ Community Signals ------------
def community_from_items(items, datalab_metrics):
    red = [it for it in items if str(it.get("source_name","")).startswith("reddit")]
    reddit = {
        "posts": len(red),
        "upvotes": sum(int(it.get("ups",0)) for it in red),
        "comments": sum(int(it.get("num_comments",0)) for it in red)
    }
    # 네이버는 포스트수가 아니라 검색량이므로 마지막 volume/growth 레벨만 표시
    naver = {}
    if datalab_metrics:
        last = datalab_metrics[0]
        naver = {
            "posts": 1,   # 지표 슬롯 채우기용 (실제 의미 없음 → '확실하지 않음')
            "vol_last": last.get("volume", 0),
            "growth": last.get("growth","알 수 없습니다")
        }
    return {"reddit": reddit, "naver": naver}

# ------------ GPT Writer (모든 섹션) ------------
def write_full_card_with_gpt(items, trends, metrics, groups):
    if not os.getenv("OPENAI_API_KEY"):
        # GPT 미사용 모드
        return {
            "title":"GPT 미사용 모드",
            "tagline":"알 수 없습니다",
            "sections":{
                "problem":"근거가 부족합니다",
                "solution":"근거가 부족합니다",
                "target_user":"근거가 부족합니다",
                "gtm":"근거가 부족합니다",
                "why_now":"근거가 부족합니다",
                "proof_signals":"근거가 부족합니다",
                "market_gap":"근거가 부족합니다",
                "execution_plan":{
                    "core":"근거가 부족합니다",
                    "growth":"근거가 부족합니다",
                    "lead_gen":"근거가 부족합니다",
                    "steps":[]
                }
            },
            "evidence":[]
        }

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    payload = {
        "date_kr": NOW_KST.strftime("%Y-%m-%d"),
        "keywordGroups": groups.get("keywordGroups", []),
        "topTerms": groups.get("topTerms", [])[:20],
        "trends": trends.get("results", [])[:MAX_GROUPS],
        "trend_metrics": metrics,
        "articles": [{"title": it.get("title",""), "url": it.get("url","")} for it in items[:20]]
    }
    sys_prompt = (
        "너는 한국 시장 트렌드 리서처다. 아래 데이터만 근거로 삼아 사업화 아이디어 카드를 한국어로 작성하라.\n"
        "원칙: 과장 금지, 정치/연예 배제, 숫자는 근거 있을 때만 제시. 근거 부족 시 '알 수 없습니다' 또는 '확실하지 않음' 사용.\n"
        "반드시 JSON 하나로만 출력: {title,tagline,sections{problem,solution,target_user,gtm,why_now,proof_signals,market_gap,execution_plan{core,growth,lead_gen,steps[]}},evidence[]}.\n"
        "evidence는 기사 URL 3~5개와 간단한 메모."
    )
    user_msg = "DATA:\n" + json.dumps(payload, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
            temperature=0.4,
            response_format={"type":"json_object"},
        )
        card = json.loads(resp.choices[0].message.content)
    except Exception:
        card = {
            "title":"아이디어 생성 실패",
            "tagline":"확실하지 않음",
            "sections":{"problem":"알 수 없습니다","solution":"알 수 없습니다","target_user":"알 수 없습니다",
                        "gtm":"알 수 없습니다","why_now":"알 수 없습니다","proof_signals":"알 수 없습니다",
                        "market_gap":"알 수 없습니다","execution_plan":{"core":"알 수 없습니다","growth":"알 수 없습니다","lead_gen":"알 수 없습니다","steps":[]}},
            "evidence":[]
        }
    return card

# ------------ Main ------------
def main():
    items = collect_all()
    signals = build_keyword_groups(items, max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT)
    datalab = naver_datalab(signals.get("keywordGroups", []), days=DATALAB_DAYS)
    metrics = extract_trend_metrics(datalab)
    scores = score_from_metrics(metrics, items)
    community = community_from_items(items, metrics)
    card = write_full_card_with_gpt(items, datalab, metrics, signals)

    # trends.json은 DataLab 원문 + metrics를 같이 저장 (프론트 차트 용이)
    trends_out = {
        "period": {"days": DATALAB_DAYS, "end_kr": NOW_KST.strftime("%Y-%m-%d")},
        "results": datalab.get("results", []),
        "metrics": metrics
    }

    # ideas.json — 프론트가 읽기 쉬운 형태
    idea_payload = {
        "date_kr": NOW_KST.strftime("%Y-%m-%d"),
        "idea": {
            "title": card.get("title",""),
            "tagline": card.get("tagline",""),
            "sections": card.get("sections", {}),
            "scores": scores,                # Trend/Market/Feasibility 등
            "community": community,          # reddit/naver
            "evidence": card.get("evidence", []),
            "keywordGroups": signals.get("keywordGroups", []),
            "topTerms": signals.get("topTerms", [])[:20]
        }
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    ch1 = dump_json(RAW_PATH, items)
    ch2 = dump_json(SIGNALS_PATH, signals)
    ch3 = dump_json(TRENDS_PATH, trends_out)
    ch4 = dump_json(IDEAS_PATH, idea_payload)

    print(f"[{NOW_KST.isoformat()}] items={len(items)} saved raw={ch1} signals={ch2} trends={ch3} ideas={ch4}")

if __name__ == "__main__":
    main()
