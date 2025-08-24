#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, math, hashlib, random
from datetime import datetime, timedelta, timezone

import requests, feedparser
from bs4 import BeautifulSoup
from dateutil import tz

# ====== 환경 ======
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
NAVER_CLIENT_ID    = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET= os.getenv("NAVER_CLIENT_SECRET", "")
YOUTUBE_API_KEY    = os.getenv("YOUTUBE_API_KEY", "")

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

FN_IDEAS   = os.path.join(DATA, "ideas.json")
FN_TRENDS  = os.path.join(DATA, "trends.json")
FN_RAW     = os.path.join(DATA, "rawitems.json")
FN_SIGNALS = os.path.join(DATA, "signals.json")

KST = timezone(timedelta(hours=9))
NOW = datetime.now(KST)

HEADERS = {"User-Agent": "ideakr-bot/1.0 (+github actions)"}

# ====== 필터/토크나이저 ======
DOMAIN_TOKEN_RE = re.compile(r"(https?://\S+)|([a-z0-9_-]+\.(co|com|kr|net|org)\b)", re.I)
PUBLISHER_BLACKLIST = {
    "매일경제","한국경제","한경","조선일보","중앙일보","동아일보","한겨레","경향신문","연합뉴스",
    "서울경제","머니투데이","이데일리","전자신문","블로터","네이버","다음","네이트","구글",
    "kbs","mbc","sbs","ytn","jtbc","hankookilbo","chosun","hankyung","zdnet","zumnet"
}
STOPWORDS = {
    "the","and","for","with","from","into","your","our","new","how","what","why","when","where","who",
    "this","that","are","is","was","to","of","in","on","by","at","as","it",
    "기사","속보","단독","사진","영상","인터뷰","전문","종합","오늘","어제","최근","관련",
    "https","http","com","kr","net","news","뉴스","기자","포토","라이브","업데이트",
}

def strip_html(s: str) -> str:
    try:
        return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)
    except Exception:
        return s or ""

def clean_text(s: str) -> str:
    s = strip_html(s)
    s = DOMAIN_TOKEN_RE.sub(" ", s)
    s = re.sub(r"[\[\]()<>{}“”\"'`•·…~^_|=:;#@※]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_ko_en(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣\s\-]", " ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    out = []
    for t in toks:
        if t.isdigit(): 
            continue
        if t in STOPWORDS or t in PUBLISHER_BLACKLIST:
            continue
        out.append(t)
    return out

def gen_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()[:10]

# ====== 수집 ======
def fetch_gnews_topics() -> list[dict]:
    """Google News RSS (KR, 주제 제한) 최근 3일"""
    # 기술/과학/비즈니스 위주
    feeds = [
        # topic=tc(technology), b( business ), snc(science) — 한국판은 topic 파라미터가 제한적이라 검색 RSS도 섞음
        "https://news.google.com/rss/search?q=site:co.kr%20테크%20OR%20기술&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=비즈니스%20OR%20스타트업%20site:co.kr&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=AI%20OR%20인공지능%20site:co.kr&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=%EB%8F%99%ED%96%A5%20OR%20%ED%8A%B8%EB%A0%8C%EB%93%9C%20site:co.kr&hl=ko&gl=KR&ceid=KR:ko",
    ]
    out = []
    since = NOW - timedelta(days=3)
    for u in feeds:
        try:
            f = feedparser.parse(u)
            for e in f.entries:
                title = e.get("title","")
                link  = e.get("link","")
                desc  = e.get("summary","")
                pub   = e.get("published_parsed") or e.get("updated_parsed")
                dt    = datetime.fromtimestamp(time.mktime(pub), tz=timezone.utc).astimezone(KST) if pub else NOW
                if dt < since: 
                    continue
                out.append({
                    "raw_id": f"gn_{sha(link or title)}",
                    "source_platform": "GoogleNews",
                    "source_name": e.get("source",{}).get("title",""),
                    "title": title,
                    "description": desc,
                    "url": link,
                    "published_at": dt.isoformat(),
                    "metrics_upvotes": 0,
                    "metrics_comments": 0
                })
        except Exception:
            continue
    return out

def fetch_reddit_by_keywords(keywords: list[str]) -> list[dict]:
    out = []
    since = NOW - timedelta(days=7)
    for kw in keywords[:6]:
        q = requests.utils.quote(kw)
        url = f"https://www.reddit.com/search.json?q={q}&sort=top&t=week&limit=15"
        try:
            r = requests.get(url, headers={"User-Agent":"ideakr/0.1"}, timeout=15)
            if r.status_code != 200:
                continue
            js = r.json()
            for c in js.get("data",{}).get("children",[]):
                d = c.get("data",{})
                dt = datetime.fromtimestamp(float(d.get("created_utc", time.time())), tz=timezone.utc).astimezone(KST)
                if dt < since: 
                    continue
                out.append({
                    "raw_id": f"rd_{sha(d.get('permalink',''))}",
                    "source_platform": "Reddit",
                    "source_name": d.get("subreddit",""),
                    "title": d.get("title",""),
                    "description": d.get("selftext","")[:300],
                    "url": "https://www.reddit.com"+d.get("permalink",""),
                    "published_at": dt.isoformat(),
                    "metrics_upvotes": int(d.get("ups",0)),
                    "metrics_comments": int(d.get("num_comments",0))
                })
        except Exception:
            continue
    return out

def fetch_youtube_for_keywords(keywords: list[str]) -> list[dict]:
    if not YOUTUBE_API_KEY:
        return []
    out = []
    since = (NOW - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for kw in keywords[:6]:
        try:
            # search
            params = {
                "part": "snippet",
                "q": kw,
                "type": "video",
                "maxResults": 10,
                "order": "viewCount",
                "publishedAfter": since,
                "regionCode": "KR",
                "relevanceLanguage": "ko",
                "key": YOUTUBE_API_KEY
            }
            sr = requests.get("https://www.googleapis.com/youtube/v3/search",
                              params=params, timeout=20)
            ids = [i["id"]["videoId"] for i in sr.json().get("items",[])]
            if not ids:
                continue
            # stats
            vr = requests.get("https://www.googleapis.com/youtube/v3/videos",
                              params={"part":"statistics,snippet","id":",".join(ids),"key":YOUTUBE_API_KEY},
                              timeout=20)
            for it in vr.json().get("items",[]):
                sn = it.get("snippet",{})
                st = it.get("statistics",{})
                published = sn.get("publishedAt")
                out.append({
                    "raw_id": f"yt_{it.get('id')}",
                    "source_platform": "YouTube",
                    "source_name": sn.get("channelTitle",""),
                    "title": sn.get("title",""),
                    "description": sn.get("description","")[:400],
                    "url": "https://www.youtube.com/watch?v="+it.get("id",""),
                    "published_at": published,
                    "metrics_upvotes": int(st.get("likeCount","0")) if st.get("likeCount") else 0,
                    "metrics_comments": int(st.get("commentCount","0")) if st.get("commentCount") else 0
                })
        except Exception:
            continue
    return out

# ====== 키워드 그룹/트렌드 ======
MAX_GROUPS = 8
def build_groups(items: list[dict]) -> list[dict]:
    score = {}

    def bump(term: str, w: float):
        if not term:
            return
        if term in STOPWORDS or term in PUBLISHER_BLACKLIST:
            return
        if re.fullmatch(r"[a-z]{1,3}", term):  # 너무 짧은 약어
            return
        if re.search(r"\b(http|https|www|com|co|kr|net)\b", term):
            return
        score[term] = score.get(term, 0.0) + w

    for it in items:
        txt = clean_text((it.get("title","") or "") + " " + (it.get("description","") or ""))
        toks = tokenize_ko_en(txt)
        bigr = gen_ngrams(toks, 2)
        trgr = gen_ngrams(toks, 3)

        src = (it.get("source_platform","") or "").lower()
        w_base = 1.0
        if src.startswith("reddit"):  w_base = 1.5
        if src.startswith("youtube"): w_base = 1.4

        w = w_base * (1.0 + 0.00002*it.get("metrics_upvotes",0) + 0.00005*it.get("metrics_comments",0))
        for t in toks: bump(t, w)
        for t in bigr: bump(t, w*1.2)
        for t in trgr: bump(t, w*1.4)

    cand = [(k,v) for k,v in score.items()]
    cand.sort(key=lambda x:x[1], reverse=True)

    groups = []
    for k,_ in cand:
        if any(k in " ".join(g["keywords"]) or g["groupName"] in k for g in groups):
            continue
        groups.append({"groupName": k, "keywords":[k]})
        if len(groups) >= MAX_GROUPS:
            break
    if not groups:
        groups = [{"groupName":"시장 동향","keywords":["시장 동향"]}]
    return groups

def query_naver_datalab(groups: list[dict]) -> dict:
    """그룹별 검색량/증감/시계열 (14일)"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {}
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "Content-Type": "application/json; charset=utf-8",
    }
    end = NOW.date()
    start = (NOW - timedelta(days=13)).date()
    body = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate":   end.strftime("%Y-%m-%d"),
        "timeUnit":  "date",
        "keywordGroups": [{"groupName": g["groupName"], "keywords": g["keywords"][:3]} for g in groups],
        "device": "", "gender": "", "ages": []
    }
    try:
        r = requests.post("https://openapi.naver.com/v1/datalab/search",
                          headers=headers, data=json.dumps(body), timeout=20)
        if r.status_code != 200:
            return {}
        js = r.json()
        out = {}
        for series in js.get("results", []):
            name = series.get("title")
            data = series.get("data", [])
            out[name] = data
        return out
    except Exception:
        return {}

def build_trends(groups: list[dict], datalab: dict) -> list[dict]:
    trends = []
    for i,g in enumerate(groups, start=1):
        name = g["groupName"]
        series = datalab.get(name, [])
        vol = sum([d.get("ratio",0) for d in series])
        growth = 0.0
        if len(series) >= 2:
            growth = (series[-1]["ratio"] - series[0]["ratio"]) / max(1.0, series[0]["ratio"])
        trend_score = min(100, round(0.7*vol + 30*growth))
        trends.append({
            "trend_id": f"tr_{i:04d}",
            "keyword": name,
            "trend_score": trend_score,
            "volume": round(vol),
            "growth_percent": float(growth),  # 0.23 → 23%
            "region": "KR",
            "timespan": "14d",
            "series": [{"date": d["period"], "value": d["ratio"]} for d in series],
            "evidence_rawitems": [],
            "updated_at": NOW.isoformat()
        })
    return trends

# ====== 카드 생성 ======
def write_card(items: list[dict], metrics: list[dict], groups: list[dict]) -> dict:
    top_kw = metrics[0]["groupName"] if metrics else (groups[0]["groupName"] if groups else "신규")
    gr = metrics[0].get("growth")
    growth_txt = (f"{gr}%" if isinstance(gr,(int,float)) else "증감률 불명")

    cites = [{"title": it.get("title",""), "url": it.get("url","")} for it in
             sorted(items, key=lambda it:(it.get("metrics_upvotes",0)*2 + it.get("metrics_comments",0)), reverse=True)[:6]]

    if not OPENAI_API_KEY:
        return {
            "title": f"{top_kw} 인사이트",
            "tagline": f"{top_kw}: 최근 {growth_txt}",
            "sections": {
                "problem": f"‘{top_kw}’ 관련 최신 동향 파악과 실천 가이드 부재.",
                "solution": "뉴스/커뮤니티/유튜브 신호를 한 장의 카드로 요약해 의사결정에 즉시 활용.",
                "target_user": "초기 창업자, PM/마케터",
                "gtm": "매일 1장 자동 발행 → 구독 유도 → 주간 브리핑"
            },
            "why_cards": [f"{top_kw} 관심도 {growth_txt}"],
            "gap_notes": ["근거가 부족합니다"],
            "exec_steps": ["근거가 부족합니다"],
            "evidence": cites
        }

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    schema = ("{"
      "\"title\":str, \"tagline\":str,"
      "\"sections\":{"
        "\"problem\":str, \"solution\":str, \"target_user\":str, \"gtm\":str"
      "},"
      "\"why_cards\":[str], \"gap_notes\":[str], \"exec_steps\":[str],"
      "\"evidence\":[{\"title\":str,\"url\":str}]"
    "}")

    sys = (
      "너는 한국 스타트업 리서처다. 아래 데이터만 근거로 한국어 카드 한 장을 만들어라. "
      "응답은 JSON 한 개로만 반환하라. 스키마는 " + schema +
      " 이다. 과장 금지, 추정은 '근거가 부족합니다'라고 써라. "
      "Market gap과 Execution Plan은 각각 3~5개 불릿으로 구체적으로 작성하라."
    )
    payload = {
        "keyword": top_kw, "growth": growth_txt,
        "metrics": metrics[:5],
        "citations": cites
    }
    msg = "DATA:\n" + json.dumps(payload, ensure_ascii=False)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":msg}],
            temperature=0.2,
            response_format={"type":"json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "title": f"{top_kw} 인사이트",
            "tagline": f"{top_kw}: 최근 {growth_txt}",
            "sections": {
                "problem": f"‘{top_kw}’ 관련 최신 동향 파악과 실천 가이드 부재.",
                "solution": "뉴스/커뮤니티/유튜브 신호를 요약.",
                "target_user": "초기 창업자, PM/마케터",
                "gtm": "매일 1장 자동 발행 → 구독 유도"
            },
            "why_cards": [f"{top_kw} 관심도 {growth_txt}"],
            "gap_notes": ["근거가 부족합니다"],
            "exec_steps": ["근거가 부족합니다"],
            "evidence": cites
        }

def build_metrics_for_groups(groups, datalab):
    metrics = []
    for g in groups:
        name = g["groupName"]
        data = datalab.get(name, [])
        vol = sum([d.get("ratio",0) for d in data])
        growth = 0.0
        if len(data) >= 2:
            growth = round( (data[-1]["ratio"]-data[0]["ratio"]) / max(1.0,data[0]["ratio"]) * 100, 1 )
        metrics.append({"groupName": name, "volume": round(vol), "growth": growth})
    metrics.sort(key=lambda m:(m["volume"], m["growth"]), reverse=True)
    return metrics

# ====== 메인 ======
def main():
    # 1) 뉴스 1차 수집
    news = fetch_gnews_topics()

    # 2) 1차 키워드 후보 생성
    groups1 = build_groups(news)

    # 3) 후보로 Reddit/YouTube 보강
    reddit = fetch_reddit_by_keywords([g["groupName"] for g in groups1])
    youtube = fetch_youtube_for_keywords([g["groupName"] for g in groups1])

    raw = news + reddit + youtube

    # 4) 최종 키워드 그룹(보강 텍스트로 재생성)
    groups = build_groups(raw)

    # 5) DataLab 검색량/시계열
    datalab = query_naver_datalab(groups)

    # 6) 메트릭 테이블
    metrics = build_metrics_for_groups(groups, datalab)

    # 7) 트렌드 JSON
    trends = build_trends(groups, datalab)

    # 8) 커뮤니티 시그널 요약
    sig = {
        "reddit":  {"posts": len(reddit), "upvotes": sum(i["metrics_upvotes"] for i in reddit), "comments": sum(i["metrics_comments"] for i in reddit)},
        "youtube": {"videos": len(youtube)},
        "naver":   {"groups": len(datalab)}
    }

    # 9) 아이디어 카드
    card = write_card(raw, metrics, groups)

    idea_obj = {
        "idea_id": f"idea_{NOW.strftime('%Y_%m_%d')}_001",
        "title_ko": card.get("title") or "아이디어 생성 실패",
        "one_liner": card.get("tagline") or "확실하지 않음",
        "problem": card.get("sections",{}).get("problem","알 수 없습니다"),
        "solution": card.get("sections",{}).get("solution","알 수 없습니다"),
        "target_user": card.get("sections",{}).get("target_user","알 수 없습니다"),
        "why_now": card.get("sections",{}).get("why_now",""),
        "biz_model": "알 수 없습니다",
        "gtm_tactics": card.get("sections",{}).get("gtm","알 수 없습니다"),
        "validation_steps": "확실하지 않음",
        "tags": [t.get("keyword") for t in trends[:4]],
        "score_breakdown": {},
        "score_total": 0,
        "trend_link": [t["trend_id"] for t in trends[:3]],
        "is_today": True,

        # 확장 필드
        "why_cards": card.get("why_cards",[]),
        "gap_notes": card.get("gap_notes",[]),
        "exec_steps": card.get("exec_steps",[]),
        "evidence": card.get("evidence",[])
    }

    # 10) 저장 (원자적 덮어쓰기)
    def dump(fp, obj): 
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    dump(FN_IDEAS,   [idea_obj])
    dump(FN_TRENDS,  trends)
    dump(FN_RAW,     raw)
    dump(FN_SIGNALS, sig)

    changed = True
    print(f"[{NOW.isoformat()}] items={len(raw)} changed={changed} "
          f"(secrets: NAVER={'ok' if NAVER_CLIENT_ID else 'no'}, "
          f"OPENAI={'ok' if OPENAI_API_KEY else 'no'}, YT={'ok' if YOUTUBE_API_KEY else 'no'})")

if __name__ == "__main__":
    main()
