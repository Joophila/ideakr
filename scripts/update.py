#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ideakr daily updater (KR, signal-driven)
- Google News RSS 수집 → rawitems.json
- 한글 명사 키워드 추출(블랙리스트 제거)
- Naver DataLab(12주 주간) OR rawitems 집계로 trends.json(series/volume/growth)
- Reddit/YouTube 신호 집계 → signals.json
- 신호 최소 기준 미달 키워드 제거 → 최고 점수 1개 선택
- GPT로 "구체 제품 1개" 카드 생성(실패 시 규칙 템플릿) → ideas.json
"""

import os, re, json, time, math, hashlib, requests, sys
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
from typing import List, Dict, Any
from bs4 import BeautifulSoup

# ---------- ENV ----------
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY","")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID","")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET","")

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY","")     # YouTube Data API v3
REDDIT_CLIENT_ID    = os.getenv("REDDIT_CLIENT_ID","")
REDDIT_CLIENT_SECRET= os.getenv("REDDIT_CLIENT_SECRET","")
REDDIT_USER_AGENT   = os.getenv("REDDIT_USER_AGENT","ideakr/0.1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

KST = timezone(timedelta(hours=9))
def now_kst(): return datetime.now(KST)

def save_json(path, obj):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path, default):
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except: return default

def strip_html(s: str) -> str:
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

# ---------- 튜닝 파라미터 ----------
WEEKS_WINDOW = 12
MIN_VOL      = 30     # DataLab 누적 ratio 합계 최소
MIN_YT       = 5      # 최근 14일 유튜브 영상 최소
MIN_REDDIT   = 1      # 최근 7일 레딧 게시물 최소

W_DATALAB = 0.60
W_YT      = 0.25
W_REDDIT  = 0.15

# ---------- 뉴스 소스 ----------
RSS_SOURCES = [
    # 국내 IT/비즈 위주 구글뉴스 질의
    "https://news.google.com/rss/search?q=site:zdnet.co.kr+OR+site:etnews.com+OR+site:aitimes.com+OR+site:hankyung.com+OR+site:mk.co.kr+OR+%EA%B8%B0%EC%88%A0+OR+%ED%95%9C%EA%B5%AD+%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85&hl=ko&gl=KR&ceid=KR:ko",
    "https://news.google.com/rss/search?q=%EC%82%AC%EC%97%85%ED%99%94+OR+AI+OR+SaaS+OR+%EC%9E%90%EB%8F%99%ED%99%94+OR+%EB%A1%9C%EC%A7%81%EC%8A%A4&hl=ko&gl=KR&ceid=KR:ko",
]

def fetch_rss_items(url: str):
    out=[]
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        for it in soup.find_all("item"):
            title = strip_html(it.title.text if it.title else "")
            link  = (it.link.text if it.link else "").strip()
            pub   = (it.pubDate.text if it.pubDate else "").strip()
            try:
                pub_dt = datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S").replace(tzinfo=timezone.utc)
                pub_kst= pub_dt.astimezone(KST).isoformat()
            except:
                pub_kst = now_kst().isoformat()
            rec = {
                "raw_id":"gn_"+hashlib.md5((title+link).encode("utf-8")).hexdigest()[:12],
                "source_platform":"GoogleNews",
                "source_name":"news.google.com",
                "title":title,
                "description": strip_html(it.description.text if it.description else ""),
                "url":link,
                "published_at":pub_kst,
                "metrics_upvotes":0,
                "metrics_comments":0
            }
            if title and link:
                out.append(rec)
    except Exception as e:
        print("RSS fail:", e)
    return out

def collect_rawitems():
    all_items=[]
    for u in RSS_SOURCES:
        all_items += fetch_rss_items(u)
        time.sleep(0.5)
    # 중복 제거(제목 기준)
    uniq={}
    for x in all_items:
        k=x["title"][:180]
        if k not in uniq: uniq[k]=x
    arr=list(uniq.values())[:400]
    print(f"[rawitems] {len(arr)}")
    return arr

# ---------- 키워드 추출 ----------
BLACK = set("""
매일경제 한국경제 조선일보 중앙일보 한겨레 경향신문 연합뉴스 머니투데이 이데일리 전자신문
디지털데일리 지디넷코리아 코리아타임스 코리아헤럴드 뉴스 속보 단독 인터뷰 칼럼 사설 포토 오피니언
네이버 카카오 구글 유튜브 트위터 페이스북 다음 블로그 카페 커뮤니티
서비스 글로벌 기술 플랫폼 솔루션 시스템 프로그램 프로젝트 산업 기업 정부 한국 중국 일본 미국 유럽 세계
오늘 어제 내일 발표 정책 지원 증가 감축 강화 확대 예정 계획
""".split())
HANGUL_NOUN = re.compile(r"[가-힣]{2,}")

def extract_keywords(raw, topn=12):
    c=Counter()
    for it in raw:
        t=(it.get("title","")+" "+it.get("description",""))
        for w in HANGUL_NOUN.findall(t):
            if w in BLACK: continue
            if w.endswith(("신문","일보","방송","뉴스")): continue
            c[w]+=1
    cand=[(w,n) for w,n in c.most_common(60) if n>=2]
    out=[w for w,_ in cand][:topn]
    if not out:
        out=["살롱 예약","리필 스테이션","재활용 포장","동네 물류","노코드 자동화","반려동물 헬스"]
    print("[keywords]", out)
    return out

# ---------- DataLab ----------
def datalab_series(keyword: str, weeks=WEEKS_WINDOW):
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
        return {"series":[], "volume":0.0, "growth":0.0, "ok":False}
    try:
        start = (now_kst().date() - timedelta(weeks=weeks)).isoformat()
        end   = now_kst().date().isoformat()
        body = {
            "startDate": start,
            "endDate": end,
            "timeUnit": "week",
            "keywordGroups":[{"groupName":keyword, "keywords":[keyword]}]
        }
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            "Content-Type":"application/json"
        }
        r = requests.post("https://openapi.naver.com/v1/datalab/search", headers=headers, json=body, timeout=20)
        if r.status_code!=200: raise RuntimeError(r.text)
        data = r.json()["results"][0]["data"]  # [{period, ratio}]
        series=[{"date":d["period"], "value": float(d["ratio"])} for d in data]
        vol=sum(x["value"] for x in series)
        prev=sum(x["value"] for x in series[:-4]) or 0.0
        last=sum(x["value"] for x in series[-4:]) or 0.0
        growth = (last-prev)/prev if prev>0 else (1.0 if last>0 else 0.0)
        return {"series":series, "volume":vol, "growth":growth, "ok":True}
    except Exception as e:
        print("datalab fail:", keyword, e)
        return {"series":[], "volume":0.0, "growth":0.0, "ok":False}

# rawitems 보강 시계열
def raw_series(keyword: str, raw):
    end = now_kst().date()
    days=[(end - timedelta(days=i)).isoformat() for i in range(27,-1,-1)]  # 4주
    d=defaultdict(int)
    for it in raw:
        txt = (it.get("title","")+" "+it.get("description",""))
        if keyword in txt:
            try:
                dt = datetime.fromisoformat(it["published_at"]).astimezone(KST).date().isoformat()
            except:
                dt = now_kst().date().isoformat()
            d[dt]+=1
    series=[{"date":x,"value":d.get(x,0)} for x in days]
    vol=sum(v["value"] for v in series)
    prev=sum(v["value"] for v in series[:14])
    last=sum(v["value"] for v in series[14:])
    growth=(last-prev)/prev if prev>0 else (1.0 if last>0 else 0.0)
    # 주차로 변환(프론트는 일/주 모두 단일값만 있어도 표시됨)
    wk=[]
    for i in range(0,len(series),7):
        seg=series[i:i+7]
        wk.append({"date":seg[-1]["date"], "value":sum(x["value"] for x in seg)})
    return {"series":wk, "volume":vol, "growth":growth}

# ---------- 커뮤니티 신호 ----------
def reddit_counts(query: str):
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        return {"posts":0,"upvotes":0,"comments":0}
    try:
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        data={"grant_type":"client_credentials"}
        headers={"User-Agent":REDDIT_USER_AGENT}
        tok = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers, timeout=15).json()
        hdr={"Authorization":f"bearer {tok['access_token']}", "User-Agent":REDDIT_USER_AGENT}
        r=requests.get("https://oauth.reddit.com/search",
                       params={"q":query, "sort":"hot","t":"week","limit":50},
                       headers=hdr, timeout=20)
        j=r.json()
        posts=j.get("data",{}).get("dist",0)
        ups=sum(max(0,c.get("data",{}).get("ups",0)) for c in j.get("data",{}).get("children",[]))
        com=sum(max(0,c.get("data",{}).get("num_comments",0)) for c in j.get("data",{}).get("children",[]))
        return {"posts":posts,"upvotes":ups,"comments":com}
    except Exception as e:
        print("reddit fail", e)
        return {"posts":0,"upvotes":0,"comments":0}

def youtube_counts(query: str):
    if not GOOGLE_API_KEY:
        return {"videos":0}
    try:
        url="https://www.googleapis.com/youtube/v3/search"
        params={"key":GOOGLE_API_KEY,"q":query,"part":"id","type":"video","maxResults":50,
                "publishedAfter":(now_kst()-timedelta(days=14)).astimezone(timezone.utc).isoformat()}
        j=requests.get(url, params=params, timeout=20).json()
        return {"videos": len(j.get("items",[]))}
    except Exception as e:
        print("youtube fail", e)
        return {"videos":0}

# ---------- 트렌드 구성 ----------
def build_trends(raw, kws):
    trends=[]
    for i,kw in enumerate(kws, start=1):
        dl = datalab_series(kw)
        if not dl["ok"]:
            dl = raw_series(kw, raw)
        trends.append({
            "trend_id": f"tr_{i:04d}",
            "keyword": kw,
            "trend_score": max(0,min(100, int(dl["volume"] + dl["growth"]*100))),
            "volume": dl["volume"],
            "growth_percent": round(dl["growth"],3),
            "region":"KR",
            "timespan": "12w" if dl.get("ok") else "4w(raw)",
            "series": dl["series"],
            "evidence_rawitems": [],
            "updated_at": now_kst().isoformat()
        })
    return trends

# ---------- 주제 선정 ----------
def pick_topic(trends):
    best=None; best_score=-1; reason=""
    for t in trends:
        kw=t["keyword"]
        sig_yt = youtube_counts(kw)
        sig_rd = reddit_counts(kw)
        vol = t["volume"]; gr = t["growth_percent"]
        # 최소 기준
        if not (vol>=MIN_VOL or sig_yt["videos"]>=MIN_YT or sig_rd["posts"]>=MIN_REDDIT):
            continue
        # 점수(가중 합)
        s = W_DATALAB*min(1.0, vol/120.0) + W_YT*min(1.0, sig_yt["videos"]/40.0) + W_REDDIT*min(1.0, sig_rd["posts"]/20.0)
        if s>best_score:
            best_score=s
            best={"trend":t, "yt":sig_yt, "rd":sig_rd}
    if not best: return None, {"reddit":{"posts":0,"upvotes":0,"comments":0},"youtube":{"videos":0},"naver":{"groups":0}}
    signals={"reddit":best["rd"], "youtube":best["yt"], "naver":{"groups":0}}
    return best, signals

# ---------- 근거(뉴스) ----------
def choose_evidence(raw, kw, limit=6):
    rows=[]
    for it in raw:
        text=(it.get("title","")+" "+it.get("description",""))
        if kw in text:
            rows.append({"title": it["title"][:120], "url": it["url"]})
    seen=set(); ev=[]
    for r in rows:
        if r["title"] in seen: continue
        ev.append(r); seen.add(r["title"])
        if len(ev)>=limit: break
    return ev

# ---------- GPT 카드 (구체 제품 강제) ----------
def make_card_specific(top_kw, metrics, evidence):
    base = {
        "title": f"{top_kw} 특화 자동화 도구",
        "tagline": f"{top_kw} 실무자의 반복작업을 줄이는 경량 SaaS",
        "sections": {
            "problem": f"{top_kw} 관련 현장에 반복·수작업이 많고, 팀 규모 대비 도구가 과합니다.",
            "solution": f"{top_kw} 핵심 워크플로우 1~2개만 자동화하는 초경량 웹앱(MVP)으로 가설검증을 목표로 합니다.",
            "target_user": "초기 채택자(파워유저/작은팀)",
            "gtm": "커뮤니티/유튜브 데모→대기열→유료 베타"
        },
        "why_cards": [f"최근 12주 검색지수 합계 {int(metrics.get('volume',0))}, 최근 4주/이전 4주 성장률 {round(metrics.get('growth',0.0)*100,1)}%"],
        "gap_notes": ["근거가 부족합니다"], "exec_steps": ["근거가 부족합니다"],
        "offer_ladder":[{"name":"Lead Magnet","price":"Free","unit":"체크리스트"},
                        {"name":"Core MVP","price":"₩9,900~₩29,000/월","unit":"SaaS"}],
        "pricing":["월 구독"], "channels":["YouTube","SEO"], "competitors":[], "personas":[]
    }
    if not OPENAI_API_KEY: return base
    try:
        from openai import OpenAI
        client=OpenAI(api_key=OPENAI_API_KEY)
        sysmsg=("너는 한국 스타트업 리서처다. 아래 정량근거만 사용해 **구체적인 제품 1개**로 작성."
                " 응답은 JSON 하나. 과장/추정 금지, 모호하면 '근거가 부족합니다'.")
        user = {
            "keyword": top_kw,
            "metrics": metrics,
            "evidence": evidence,
            "format": {
                "title":"제품명(예: 'AI 기반 살롱 재예약 어시스턴트')",
                "tagline":"한 줄 설명",
                "sections":{"problem":"문제","solution":"해결","target_user":"타겟","gtm":"초기획득"},
                "why_cards":["정량 근거 2~3개(숫자 포함)"],
                "gap_notes":["경쟁의 빈틈(숫자/근거)"],
                "exec_steps":["2주/4주/8주 실행 스텝(측정지표 포함)"],
                "offer_ladder":[{"name":"Lead Magnet","price":"Free"},{"name":"Core","price":"₩"}],
                "pricing":["간단 가격"],
                "channels":["주요 채널 2~3개"]
            }
        }
        resp=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sysmsg},
                      {"role":"user","content":"DATA:\n"+json.dumps(user,ensure_ascii=False)}],
            temperature=0.2,
            response_format={"type":"json_object"}
        )
        out=json.loads(resp.choices[0].message.content)
        # 안전 보강
        for k,v in base.items():
            if k not in out or (isinstance(out[k],(str,list,dict)) and not out[k]):
                out[k]=v
        for sk,sv in base["sections"].items():
            if "sections" not in out: out["sections"]={}
            if not out["sections"].get(sk): out["sections"][sk]=sv
        return out
    except Exception as e:
        print("GPT fail -> fallback:", e)
        return base

# ---------- 메인 ----------
def main():
    raw = collect_rawitems()
    save_json(os.path.join(DATA,"rawitems.json"), raw)

    kws     = extract_keywords(raw, topn=10)
    trends  = build_trends(raw, kws)

    pick, signals = pick_topic(trends)
    if not pick:
        # 모든 키워드가 기준 미달 → 보류 상태 저장
        trends_sorted = sorted(trends, key=lambda t: (t["volume"], t["growth_percent"]), reverse=True)
        save_json(os.path.join(DATA,"trends.json"), trends_sorted)
        save_json(os.path.join(DATA,"signals.json"), signals)
        idea = {
            "idea_id": f"idea_{now_kst().strftime('%Y_%m_%d')}_001",
            "title_ko": "아이디어 생성 보류",
            "one_liner": "근거 부족으로 오늘은 제안 보류",
            "problem": "근거가 부족합니다",
            "solution": "근거가 부족합니다",
            "target_user": "근거가 부족합니다",
            "why_now": "모든 후보가 최소 신호 기준 미달(DataLab/YouTube/Reddit)",
            "biz_model": "알 수 없습니다",
            "gtm_tactics": "알 수 없습니다",
            "validation_steps": "추가 데이터 수집 필요",
            "tags": [t["keyword"] for t in trends_sorted[:4]],
            "score_breakdown": {"trend":0,"market":0,"competition_invert":50,"feasibility":50,"monetization":50,"regulatory_invert":50},
            "score_total": 27,
            "trend_link": [t["trend_id"] for t in trends_sorted[:3]],
            "is_today": True,
            "why_cards": ["최소 기준: DataLab≥30 or YouTube≥5 or Reddit≥1"],
            "gap_notes": ["근거가 부족합니다"],
            "exec_steps": ["근거가 부족합니다"],
            "evidence": [],
            "offer_ladder": [], "pricing": [], "channels": [], "competitors": [], "personas":[]
        }
        save_json(os.path.join(DATA,"ideas.json"), [idea])
        print("[hold] no topic passed thresholds")
        return

    t  = pick["trend"]
    kw = t["keyword"]
    evidence = choose_evidence(raw, kw, limit=6)

    metrics = {"volume": t["volume"], "growth": t["growth_percent"]}
    card = make_card_specific(kw, metrics, evidence)

    # 결과 저장
    trends_sorted = sorted(trends, key=lambda x: (x["volume"], x["growth_percent"]), reverse=True)
    save_json(os.path.join(DATA,"trends.json"), trends_sorted)
    save_json(os.path.join(DATA,"signals.json"), signals)

    idea = {
        "idea_id": f"idea_{now_kst().strftime('%Y_%m_%d')}_001",
        "title_ko": card.get("title") or f"{kw} 기반 솔루션",
        "one_liner": card.get("tagline",""),
        "problem": card["sections"]["problem"],
        "solution": card["sections"]["solution"],
        "target_user": card["sections"]["target_user"],
        "why_now": " · ".join(card.get("why_cards",[])[:2]),
        "biz_model": " · ".join(card.get("pricing",[])[:1]),
        "gtm_tactics": card["sections"]["gtm"],
        "validation_steps": " / ".join(card.get("exec_steps",[])[:2]) or "빠른 인터뷰-대기열-유료베타",
        "tags": [t["keyword"] for t in trends_sorted[:4]],
        "score_breakdown": {
            "trend": max(0,min(100,int(t["volume"] + t["growth_percent"]*100))),
            "market": min(100, len(raw)*2),
            "competition_invert": 100 - min(90, int(len(raw)/5)*10),
            "feasibility": 50, "monetization": 50, "regulatory_invert":50
        },
        "score_total": 0,
        "trend_link": [t["trend_id"]],
        "is_today": True,
        "why_cards": card.get("why_cards",[]),
        "gap_notes": card.get("gap_notes",[]) or ["근거가 부족합니다"],
        "exec_steps": card.get("exec_steps",[]) or ["근거가 부족합니다"],
        "evidence": evidence,
        "offer_ladder": card.get("offer_ladder",[]),
        "pricing": card.get("pricing",[]),
        "channels": card.get("channels",[]),
        "competitors": card.get("competitors",[]),
        "personas": card.get("personas",[])
    }
    idea["score_total"] = int(0.45*idea["score_breakdown"]["trend"] + 0.30*idea["score_breakdown"]["market"]
                              + 0.25*idea["score_breakdown"]["competition_invert"])

    save_json(os.path.join(DATA,"ideas.json"), [idea])
    print(f"[done] topic='{kw}' vol={t['volume']} yt={signals['youtube']['videos']} rd={signals['reddit']['posts']}")
    
if __name__ == "__main__":
    main()

