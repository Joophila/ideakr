# -*- coding: utf-8 -*-
# ideakr — daily updater (GNews + Reddit + YouTube + Naver DataLab + optional GPT)

from __future__ import annotations
import json, os, re, time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import feedparser, requests
from bs4 import BeautifulSoup
from dateutil import parser as dtp

# --------- time / env ---------
KST = timezone(timedelta(hours=9))
NOW_UTC = datetime.now(timezone.utc)
NOW_KST = NOW_UTC.astimezone(KST)

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID      = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET  = os.getenv("NAVER_CLIENT_SECRET")
YOUTUBE_API_KEY      = os.getenv("YOUTUBE_API_KEY")

ENABLE_GNEWS  = (os.getenv("ENABLE_GNEWS","1") != "0")
ENABLE_REDDIT = (os.getenv("ENABLE_REDDIT","1") != "0")
ENABLE_YT     = (os.getenv("ENABLE_YT","1") != "0")

WINDOW_HOURS  = int(os.getenv("WINDOW_HOURS","72"))
DATALAB_DAYS  = int(os.getenv("DATALAB_DAYS","14"))
MAX_GNEWS     = int(os.getenv("MAX_GNEWS","60"))
MAX_REDDIT    = int(os.getenv("MAX_REDDIT","20"))
MAX_YT_TREND  = int(os.getenv("MAX_YT_TREND","20"))
MAX_YT_SEARCH = int(os.getenv("MAX_YT_SEARCH","10"))
MAX_GROUPS          = int(os.getenv("MAX_GROUPS","8"))
MAX_GROUPS_DATALAB  = int(os.getenv("MAX_GROUPS_DATALAB","5"))
MIN_TERM_COUNT      = int(os.getenv("MIN_TERM_COUNT","3"))

TOPICS = [t.strip() for t in os.getenv("GNEWS_TOPICS","BUSINESS,SCIENCE,TECHNOLOGY").split(",") if t.strip()]
REDDIT_SUBS = [s.strip() for s in os.getenv("REDDIT_SUBS","Entrepreneur,startups,technology,Futurology,smallbusiness").split(",") if s.strip()]
YOUTUBE_REGION = os.getenv("YOUTUBE_REGION","KR")

DATA_DIR    = "data"
RAW_PATH    = os.path.join(DATA_DIR,"rawitems.json")
TRENDS_PATH = os.path.join(DATA_DIR,"trends.json")
IDEAS_PATH  = os.path.join(DATA_DIR,"ideas.json")
SIGNALS_PATH= os.path.join(DATA_DIR,"signals.json")

# --------- filters / utils ---------
POLITICS_BLOCK = {"대통령","총선","국회","정치","여당","야당","민주당","국민의힘","선거","의회","외교","북한","연예","아이돌"}
STOPWORDS = {"속보","뉴스","단독","영상","사진","기자","오늘","이번","관련"}
MEDIA_DOMAINS = {"naver","daum","nate","chosun","joongang","hankyung","mk","kbs","mbn","sbs","jtbc","ytn","zdnet","zdnet","hankookilbo","hankyoreh"}
TERM_BLOCK=set((os.getenv("BLOCK_TERMS") or "").split(",")) - {""}
DEFAULT_SEEDS=[s for s in (os.getenv("DEFAULT_SEEDS") or "인공지능,전기차,친환경,리필 스테이션,구독 서비스").split(",") if s]

def strip_html(s:str)->str:
    if not s: return ""
    # 경고 회피: 파일경로로 오인될 때가 있어 직접 문자열로 파싱
    try: return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    except Exception: return str(s)

def as_iso(dt_any:Any)->str|None:
    if isinstance(dt_any,str):
        try: dt_any = dtp.parse(dt_any)
        except Exception: return None
    if not getattr(dt_any,"tzinfo",None): dt_any = dt_any.replace(tzinfo=timezone.utc)
    return dt_any.isoformat()

def is_recent_iso(iso:str|None, hours:int=WINDOW_HOURS)->bool:
    if not iso: return False
    try:
        d=dtp.parse(iso); d=d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        return d >= datetime.now(timezone.utc)-timedelta(hours=hours)
    except Exception: return False

def dump_json(path:str, obj:Any)->bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old=None
    try:
        with open(path,"r",encoding="utf-8") as f: old=json.load(f)
    except Exception: pass
    if old==obj: return False
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.write("\n")
    return True

def is_blocked_text(text:str|None)->bool:
    t=(text or "").lower()
    return any(k.lower() in t for k in POLITICS_BLOCK)

def host_parts(url:str)->set[str]:
    if not url or not url.startswith("http"): return set()
    try:
        host=url.split("/")[2].lower()
        parts=re.split(r"[.\-]",host)
        return {p for p in parts if p and p not in {"www","m","news","co","kr","com","net"}}
    except Exception: return set()

def bad_token(tok:str,url:str)->bool:
    t=(tok or "").lower().strip()
    if not t: return True
    if t in TERM_BLOCK: return True
    if t.isdigit(): return True
    if len(t.replace(" ",""))<2: return True
    # 매체명/도메인 파편 배제
    if t in host_parts(url): return True
    return False

def tokenize_ko_en(s:str)->list[str]:
    s=(s or "").lower()
    s=re.sub(r"&[a-z0-9#]+;"," ",s)
    s=re.sub(r"[^가-힣a-z0-9\- ]"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s.split(" ") if s else []

def clamp(x, lo, hi): 
    try: 
        x=float(x)
        return max(lo, min(hi, x))
    except Exception:
        return lo

# --------- collectors ---------
def fetch_gnews_topic(topic:str)->list[dict]:
    url=f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=ko&gl=KR&ceid=KR:ko"
    out=[]
    try: feed=feedparser.parse(url)
    except Exception: return out
    for e in feed.entries[:MAX_GNEWS]:
        title=e.get("title") or ""; link=e.get("link") or ""
        desc=strip_html(e.get("summary") or e.get("description") or "")
        pub=e.get("published") or e.get("updated"); pub_iso=as_iso(pub) if pub else None
        if not pub_iso or not is_recent_iso(pub_iso): continue
        if is_blocked_text(title) or is_blocked_text(desc): continue
        out.append({
            "title":title,"url":link,"published_at":pub_iso,
            "description":desc,
            "source_name":f"GoogleNews:{topic}","source_url":url,"topic":topic,
            "ups":0,"num_comments":0
        })
    return out

def fetch_reddit_sub(sub:str)->list[dict]:
    url=f"https://www.reddit.com/r/{sub}/top.json"
    qs={"t":"week","limit":MAX_REDDIT}; headers={"User-Agent":"Mozilla/5.0"}
    try:
        r=requests.get(url,params=qs,headers=headers,timeout=20); r.raise_for_status(); js=r.json()
    except Exception: return []
    out=[]
    for ch in (js.get("data",{}) or {}).get("children",[]):
        d=ch.get("data",{}) or {}
        title=d.get("title") or ""; desc=d.get("selftext") or ""; created=d.get("created_utc")
        if not created: continue
        pub_iso=as_iso(datetime.fromtimestamp(created, tz=timezone.utc))
        if not is_recent_iso(pub_iso): continue
        if is_blocked_text(title) or is_blocked_text(desc): continue
        permalink=d.get("permalink"); urlp=f"https://www.reddit.com{permalink}" if permalink else (d.get("url") or "")
        out.append({
            "title":title,"url":urlp,"published_at":pub_iso,"description":desc,
            "source_name":f"reddit:{sub}","source_url":f"https://www.reddit.com/r/{sub}","topic":"REDDIT",
            "ups":int(d.get("ups") or 0),"num_comments":int(d.get("num_comments") or 0),
        })
    return out

# ---- YouTube (Data API v3)
def yt_get(url:str, params:dict)->dict|None:
    try:
        params=dict(params or {})
        params["key"]=YOUTUBE_API_KEY
        r=requests.get(url, params=params, timeout=20)
        if r.status_code!=200: return None
        return r.json()
    except Exception:
        return None

def fetch_yt_trending(region:str=YOUTUBE_REGION, max_n:int=MAX_YT_TREND)->list[dict]:
    if not (ENABLE_YT and YOUTUBE_API_KEY): return []
    js=yt_get("https://www.googleapis.com/youtube/v3/videos", {
        "part":"snippet,statistics",
        "chart":"mostPopular",
        "regionCode":region,
        "maxResults": min(50, max_n)
    })
    out=[]
    for it in (js or {}).get("items",[]):
        sn=it.get("snippet",{}) or {}; st=it.get("statistics",{}) or {}
        title=sn.get("title") or ""; desc=sn.get("description") or ""
        pub=sn.get("publishedAt"); pub_iso=as_iso(pub) if pub else None
        if not is_recent_iso(pub_iso): continue
        if is_blocked_text(title) or is_blocked_text(desc): continue
        vid=it.get("id")
        url=f"https://www.youtube.com/watch?v={vid}" if vid else ""
        out.append({
            "title":title, "url":url, "published_at":pub_iso,
            "description":desc,
            "source_name":f"YouTube:trending","source_url":"https://www.youtube.com/feed/trending","topic":"YOUTUBE",
            "ups":int(st.get("viewCount") or 0), "num_comments":int(st.get("commentCount") or 0),
        })
    return out

def fetch_yt_search(query:str, max_n:int=MAX_YT_SEARCH)->list[dict]:
    if not (ENABLE_YT and YOUTUBE_API_KEY): return []
    # 최근 WINDOW_HOURS 내 업로드만
    published_after=(datetime.now(timezone.utc)-timedelta(hours=WINDOW_HOURS)).isoformat()
    s=yt_get("https://www.googleapis.com/youtube/v3/search", {
        "part":"snippet","q":query,"type":"video",
        "maxResults": min(50, max_n),
        "order":"viewCount","relevanceLanguage":"ko",
        "regionCode":YOUTUBE_REGION,"publishedAfter":published_after
    })
    items = (s or {}).get("items",[])
    if not items: return []
    ids=[it["id"]["videoId"] for it in items if "id" in it and "videoId" in it["id"]]
    stats={}
    if ids:
        js=yt_get("https://www.googleapis.com/youtube/v3/videos", {
            "part":"statistics","id":",".join(ids)
        }) or {}
        for it in js.get("items",[]):
            stats[it.get("id")] = it.get("statistics",{}) or {}
    out=[]
    for it in items:
        sn=it.get("snippet",{}) or {}
        title=sn.get("title") or ""; desc=sn.get("description") or ""
        pub=sn.get("publishedAt"); pub_iso=as_iso(pub) if pub else None
        if not is_recent_iso(pub_iso): continue
        if is_blocked_text(title) or is_blocked_text(desc): continue
        vid=it.get("id",{}).get("videoId")
        st=stats.get(vid, {})
        url=f"https://www.youtube.com/watch?v={vid}" if vid else ""
        out.append({
            "title":title,"url":url,"published_at":pub_iso,
            "description":desc,
            "source_name":f"YouTube:search","source_url":"https://www.youtube.com","topic":query,
            "ups":int(st.get("viewCount") or 0),"num_comments":int(st.get("commentCount") or 0),
        })
    return out

# -------- keywording --------
def collect_all()->list[dict]:
    items=[]
    if ENABLE_GNEWS:
        for t in TOPICS:
            items.extend(fetch_gnews_topic(t)); time.sleep(0.25)
    if ENABLE_REDDIT:
        for sub in REDDIT_SUBS:
            items.extend(fetch_reddit_sub(sub)); time.sleep(0.25)
    if ENABLE_YT and YOUTUBE_API_KEY:
        items.extend(fetch_yt_trending())
    # dedupe by URL
    seen=set(); uniq=[]
    for it in items:
        key=(it.get("url") or "").strip().lower().rstrip("/")
        if not key or key in seen: continue
        seen.add(key); uniq.append(it)
    return uniq

def build_keyword_groups(items:list[dict], max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT)->dict:
    cnt={}
    for it in items:
        text=f"{it.get('title','')} {it.get('description','')}"; url=it.get("url","")
        toks=[w for w in tokenize_ko_en(strip_html(text)) if not bad_token(w,url) and w not in STOPWORDS]
        w=2.0 if str(it.get("source_name","")).startswith(("reddit","YouTube")) else 1.0
        for tok in toks: cnt[tok]=cnt.get(tok,0)+w
        # bigram
        for i in range(len(toks)-1):
            bg=f"{toks[i]} {toks[i+1]}"
            if not bad_token(bg,url): cnt[bg]=cnt.get(bg,0)+w
    terms=sorted([(k,v) for k,v in cnt.items() if v>=min_count], key=lambda x:x[1], reverse=True)[:120]
    groups=[]; used=set()
    for term,_ in terms:
        if len(groups)>=max_groups: break
        if term in used: continue
        groups.append({"groupName":term,"keywords":[term]}); used.add(term)
    return {"keywordGroups":groups,"topTerms":terms,"itemCount":len(items)}

def is_korean(s:str)->bool: return bool(re.search(r"[가-힣]", s or ""))

def valid_for_datalab(term:str, allow_space=True)->bool:
    t=(term or "").strip()
    if not t: return False
    if not allow_space and " " in t: return False
    if is_korean(t): return len(t.replace(" ",""))>=2
    return len(t.replace(" ",""))>=3

def select_groups_for_datalab(groups:list[dict], allow_space=True, include_english=True, max_groups=MAX_GROUPS_DATALAB)->list[dict]:
    picked=[]
    for g in groups:
        rep=g.get("groupName") or (g.get("keywords") or [""])[0]
        if not rep: continue
        if not include_english and not is_korean(rep): continue
        if not valid_for_datalab(rep, allow_space=allow_space): continue
        picked.append({"groupName":rep,"keywords":[rep]})
        if len(picked)>=max_groups: break
    if not picked:
        for s in DEFAULT_SEEDS:
            if valid_for_datalab(s): picked.append({"groupName":s,"keywords":[s]})
            if len(picked)>=max_groups: break
    return picked

# -------- Naver DataLab --------
def naver_datalab(keyword_groups:list[dict], days:int=DATALAB_DAYS)->dict:
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"error":"missing_keys","results":[]}
    end_kst=NOW_KST.date(); start_kst=(NOW_KST - timedelta(days=days)).date()
    body={"startDate":str(start_kst),"endDate":str(end_kst),"timeUnit":"date",
          "keywordGroups":[{"groupName":g["groupName"],"keywords":g["keywords"][:3]} for g in keyword_groups[:MAX_GROUPS_DATALAB]],
          "device":"","ages":[],"gender":""}
    headers={"X-Naver-Client-Id":NAVER_CLIENT_ID,"X-Naver-Client-Secret":NAVER_CLIENT_SECRET,"Content-Type":"application/json; charset=utf-8"}
    try:
        r=requests.post("https://openapi.naver.com/v1/datalab/search", headers=headers, data=json.dumps(body), timeout=20)
        if r.status_code!=200: return {"error":f"http_{r.status_code}","results":[]}
        return r.json()
    except Exception: return {"error":"exception","results":[]}

def extract_trend_metrics(datalab:dict)->list[dict]:
    metrics=[]
    for g in datalab.get("results",[]):
        data=g.get("data",[])
        if not data:
            metrics.append({"groupName":g.get("title",""),"volume":0,"growth":"알 수 없습니다"})
            continue
        first=data[0]["ratio"]; last=data[-1]["ratio"]
        try:
            growth = round(((last - first)/(first or 1))*100, 1)    # % 값
        except Exception:
            growth = "알 수 없습니다"
        metrics.append({"groupName":g.get("title",""),"volume":int(last),"growth":growth})
    return metrics

# -------- scoring / community --------
def score_from_metrics(metrics:list[dict], items:list[dict])->dict:
    if metrics:
        vols=[m["volume"] for m in metrics if isinstance(m.get("volume"),(int,float))]
        grs=[m["growth"] for m in metrics if isinstance(m.get("growth"),(int,float))]
        v=sum(vols)/len(vols) if vols else 0
        g=sum(grs)/len(grs) if grs else 0
        trend=int(clamp(v*0.5 + g*1.0, 0, 100))  # (추측한 내용입니다)
    else: trend=0
    market=int(clamp(len(items)*2, 0, 100))
    # 도메인 다양성 → 경쟁역전 점수
    domains={}
    for it in items:
        u=(it.get("url") or "").lower()
        host=u.split("/")[2] if u.startswith("http") and len(u.split("/"))>2 else ""
        if host: domains[host]=domains.get(host,0)+1
    avg=sum(domains.values())/len(domains) if domains else 0
    competition_invert=int(clamp(100 - min(90, avg*10), 0, 100))
    monetization=50; feasibility=50; regulatory_invert=50
    overall=int(0.35*trend + 0.25*market + 0.15*competition_invert + 0.10*monetization + 0.15*feasibility)
    return {"trend":trend,"market":market,"competition_invert":competition_invert,"feasibility":feasibility,
            "monetization":monetization,"regulatory_invert":regulatory_invert,"overall":overall}

def summarize_community(items:list[dict], metrics:list[dict])->dict:
    red=[it for it in items if str(it.get("source_name","")).startswith("reddit")]
    yts=[it for it in items if str(it.get("source_name","")).startswith("YouTube")]
    return {
        "reddit":{"posts":len(red),"upvotes":sum(int(it.get("ups") or 0) for it in red),"comments":sum(int(it.get("num_comments") or 0) for it in red)},
        "youtube":{"videos":len(yts),"views":sum(int(it.get("ups") or 0) for it in yts)},
        "naver":{"vol_last":(metrics[0].get("volume") if metrics else 0),
                 "growth":(metrics[0].get("growth") if metrics else "알 수 없습니다")}
    }

# -------- evidence / why --------
def top_evidence_for_keyword(items:list[dict], kw:str, k:int=3)->list[dict]:
    kw=(kw or "").lower()
    def rel(it:dict)->float:
        t=(it.get("title") or "").lower(); d=(it.get("description") or "").lower()
        score=0.0
        if kw in t: score+=3
        if kw in d: score+=1
        score += (it.get("ups") or 0)*0.01 + (it.get("num_comments") or 0)*0.01
        return score
    cands=[it for it in items if kw in (it.get("title","")+" "+it.get("description","")).lower()]
    return sorted(cands, key=rel, reverse=True)[:k]

def derive_why_factors(items:list[dict], metrics:list[dict])->dict:
    vols=[m.get("volume",0) for m in metrics if isinstance(m.get("volume"),(int,float))]
    growths=[m.get("growth") for m in metrics if isinstance(m.get("growth"),(int,float))]
    avg_v=int(sum(vols)/len(vols)) if vols else 0
    avg_g=round(sum(growths)/len(growths),1) if growths else None

    mt=[]
    if avg_v: mt.append(f"최근 검색량 평균 {avg_v:,} 수준")
    if isinstance(avg_g,(int,float)): mt.append(f"{len(metrics)}개 키워드 평균 성장 {avg_g}%")
    mt.append(f"최근 {WINDOW_HOURS}시간 내 유효 근거 {len(items)}건")

    TECH_TOKENS=["api","sdk","오픈소스","open source","모델","llm","inference","gpu","클라우드","saas","앱","automation","agent"]
    te_hits=[it for it in items if any(tok in (it.get("title","")+it.get("description","")).lower() for tok in TECH_TOKENS)]
    te=[]
    if te_hits: te.append(f"기술 신호 포함 원문 {len(te_hits)}건 (API/오픈소스/LLM 등)")
    domains={}
    for it in items:
        u=(it.get("url") or "").lower()
        host=u.split("/")[2] if u.startswith("http") and len(u.split("/"))>2 else ""
        if host: domains[host]=domains.get(host,0)+1
    if len(domains)>=3: te.append(f"출처 다양성 {len(domains)}개 도메인")

    rr=[]
    if len(items)>=20: rr.append("언론·커뮤니티 다수 언급 → 수요 검증 신호")
    if avg_v and avg_v>=50: rr.append("검색량 하방경직(최소 볼륨 확보)")
    if not rr: rr.append("규모는 작지만 초기 신호 관찰됨")

    def rel(it:dict)->float:
        return (it.get("ups",0)*2 + it.get("num_comments",0)) if str(it.get("source_name","")).startswith(("reddit","YouTube")) else 1
    top=sorted(items, key=rel, reverse=True)[:3]
    sdp=[f"{(it.get('source_name') or 'src')}: {it.get('title','')[:40]}…" for it in top if it.get("title")]

    return {
        "market_timing":mt[:4] or ["근거가 부족합니다"],
        "tech_enablers":te[:4] or ["근거가 부족합니다"],
        "risk_reduction":rr[:4] or ["근거가 부족합니다"],
        "supporting_data":sdp[:4] or ["근거가 부족합니다"]
    }

# -------- card writers --------
def build_rule_based_card(items:list[dict], metrics:list[dict], groups:dict)->dict:
    top_kw=(metrics[0]["groupName"] if metrics else (groups.get("keywordGroups",[{}])[0].get("groupName","신규 트렌드")))
    gr=metrics[0].get("growth") if metrics else None
    gr_txt=f"최근 {DATALAB_DAYS}일 {gr}%" if isinstance(gr,(int,float)) else "증감률 불명"
    def sc(it:dict)->float: return (it.get("ups",0)*2) + it.get("num_comments",0)
    top_items=sorted(items, key=sc, reverse=True)[:3]
    evidence=[{"title":it.get("title",""),"url":it.get("url","")} for it in top_items if it.get("url")]
    return {
        "title":f"{top_kw} 트렌드 리서치 카드",
        "tagline":f"{top_kw}: {gr_txt} · 근거 기반 요약",
        "sections":{
            "problem":f"‘{top_kw}’ 관심은 늘지만 근거 중심 한국어 리서치가 부족합니다.",
            "solution":f"{gr_txt} 트렌드를 바탕으로 뉴스/커뮤니티 근거 링크와 요약 제공.",
            "target_user":"초기 창업가, 리서치가 필요한 마케터/PM",
            "gtm":"매일 카드 1건 무료 → 이메일 구독 → 주간 리포트 유료",
            "why_now":f"관심 급증({gr_txt}) + 생성형 도구 확산으로 ‘근거 요약’ 수요 증가.",
            "proof_signals":"상위 기사·레딧/유튜브 포스트 링크, 검색량 추세.",
            "market_gap":"한국시장 맞춤의 근거 중심 요약 도구 부재.",
            "execution_plan":{
                "core":"상위 키워드 자동 카드 → 검수",
                "growth":"SNS 미리보기 카드, 구독 리드 축적",
                "lead_gen":"키워드 알림·맞춤 리포트 폼",
                "steps":["수집→키워드/검색량 갱신","카드 자동 생성","검수 후 Pages 배포"]
            }
        },
        "evidence":evidence
    }

def write_full_card_with_gpt(items:list[dict], trends:dict, metrics:list[dict], groups:dict)->dict:
    if not OPENAI_API_KEY:
        print("CARD_SOURCE: rule-based (no OPENAI)")
        return build_rule_based_card(items,metrics,groups)
    try:
        from openai import OpenAI
        client=OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"OPENAI_INIT_ERROR: {e} -> fallback")
        return build_rule_based_card(items,metrics,groups)

    payload={"date_kr":NOW_KST.strftime("%Y-%m-%d"),
             "keywordGroups":groups.get("keywordGroups",[]),
             "topTerms":groups.get("topTerms",[])[:20],
             "trends":trends.get("results",[])[:MAX_GROUPS],
             "trend_metrics":metrics,
             "articles":[{"title":it.get("title",""),"url":it.get("url","")} for it in items[:20]]}

    sys_prompt=("너는 한국 시장 트렌드 리서처다. 아래 데이터(기사/커뮤니티/유튜브/검색트렌드)만 근거로 "
                "사업화 아이디어 카드를 한국어로 작성하라. 과장 금지, 정치/연예 배제. "
                "숫자/시장규모는 근거 없으면 '알 수 없습니다' 또는 '확실하지 않음'. "
                "JSON 하나로 출력: {title,tagline,sections{problem,solution,target_user,gtm,why_now,"
                "proof_signals,market_gap,execution_plan{core,growth,lead_gen,steps[]}},evidence[]}.")
    user_msg="DATA:\n"+json.dumps(payload, ensure_ascii=False)

    for m in ["gpt-4o-mini","gpt-4o"]:
        try:
            resp=client.chat.completions.create(
                model=m,
                messages=[{"role":"system","content":sys_prompt},
                          {"role":"user","content":user_msg}],
                temperature=0.4,
                response_format={"type":"json_object"}
            )
            card=json.loads(resp.choices[0].message.content)
            print("CARD_SOURCE: gpt", m)
            return card
        except Exception as e:
            print(f"OPENAI_ERROR model={m} -> {repr(e)}")
            continue
    print("CARD_SOURCE: rule-based (gpt failed)")
    return build_rule_based_card(items,metrics,groups)

# -------- frontend json builders --------
def make_id(prefix:str, n:int)->str: return f"{prefix}_{n:04d}"

def to_frontend_json(items:list[dict], datalab:dict, metrics:list[dict], signals:dict, card:dict, scores:dict):
    # rawitems.json
    raw_list=[]
    for idx,it in enumerate(items,1):
        raw_list.append({
            "raw_id":make_id("raw",idx),
            "source_platform":it.get("source_name") or "",
            "query_or_topic":it.get("topic") or "",
            "title":it.get("title") or "",
            "content_snippet":(it.get("description") or "")[:300],
            "url":it.get("url") or "",
            "metrics_upvotes":it.get("ups") or 0,
            "metrics_comments":it.get("num_comments") or 0,
            "search_volume":"",
            "language":"ko",
            "published_at":it.get("published_at") or "",
            "fetched_at":NOW_KST.isoformat()
        })
    url2rawid={r["url"]:r["raw_id"] for r in raw_list if r.get("url")}

    # trends.json (series => [{date, volume}] 로 내보냄)
    series_map={}
    for g in (datalab.get("results") or []):
        title=g.get("title") or ""; data=g.get("data") or []
        series_map[title]=[
            {"date":p.get("period"), "volume":p.get("ratio")}
            for p in data if "period" in p and "ratio" in p
        ]

    trend_arr=[]
    for i,m in enumerate(metrics,1):
        kw=m.get("groupName") or ""
        last_vol=int(m.get("volume") or 0)
        # growth_percent은 UI가 *100 해서 표시하므로 0~1 스케일로 저장
        gr_num=m.get("growth") if isinstance(m.get("growth"),(int,float)) else 0.0
        growth_ratio=round((gr_num/100.0), 4)
        evid=[url2rawid[e["url"]] for e in top_evidence_for_keyword(items, kw, k=3) if e.get("url") in url2rawid]
        # 간단 점수
        trend_score=int(clamp(last_vol*1.0 + gr_num*0.8 + len(evid)*3, 0, 100))
        trend_arr.append({
            "trend_id":make_id("tr", i),
            "keyword":kw,
            "trend_score":trend_score,
            "volume":last_vol,
            "growth_percent":growth_ratio,   # 0~1
            "region":"KR",
            "timespan":f"{DATALAB_DAYS}d",
            "evidence_rawitems":evid,
            "updated_at":NOW_KST.isoformat()
        })

    # signals.json (커뮤니티 집계 + 샘플)
    signals_out={"summary":signals, "updated_at":NOW_KST.isoformat()}

    # ideas.json (오늘 카드 1건)
    idea={
        "idea_id": f"idea_{NOW_KST.strftime('%Y_%m_%d')}_001",
        "title_ko": card.get("title") or "아이디어 생성 실패",
        "one_liner": card.get("tagline") or "확실하지 않음",
        "problem": card.get("sections",{}).get("problem","알 수 없습니다"),
        "solution": card.get("sections",{}).get("solution","알 수 없습니다"),
        "target_user": card.get("sections",{}).get("target_user","알 수 없습니다"),
        "why_now": card.get("sections",{}).get("why_now","알 수 없습니다"),
        "biz_model": card.get("sections",{}).get("market_gap","알 수 없습니다"),
        "gtm_tactics": card.get("sections",{}).get("gtm","알 수 없습니다"),
        "validation_steps": "확실하지 않음",
        "tags": [m.get("groupName") for m in metrics[:4] if m.get("groupName")],
        "score_breakdown": scores,
        "score_total": scores.get("overall",0),
        "trend_link": [t["trend_id"] for t in trend_arr[:3]],
        "is_today": True,
        "why_factors": derive_why_factors(items, metrics),
        "community": signals
    }
    ideas_arr=[idea]

    return raw_list, trend_arr, ideas_arr, signals_out

# -------- main --------
def main():
    items = collect_all()

    # 동적 키워드 → DataLab
    groups = build_keyword_groups(items)
    datalab_groups = select_groups_for_datalab(groups["keywordGroups"], allow_space=True, include_english=True)
    datalab = naver_datalab(datalab_groups)
    metrics = extract_trend_metrics(datalab)

    # YouTube 검색은 선별된 대표 키워드 위주로
    if ENABLE_YT and YOUTUBE_API_KEY and datalab_groups:
        for g in datalab_groups[:3]:
            q=g["groupName"]
            items.extend(fetch_yt_search(q, max_n=MAX_YT_SEARCH))
            time.sleep(0.2)
        # 중복 제거
        seen=set(); uniq=[]
        for it in items:
            key=(it.get("url") or "").strip().lower().rstrip("/")
            if not key or key in seen: continue
            seen.add(key); uniq.append(it)
        items=uniq

    scores  = score_from_metrics(metrics, items)
    signals = summarize_community(items, metrics)

    # 카드 작성 (GPT → 실패시 규칙기반)
    card   = write_full_card_with_gpt(items, datalab, metrics, groups)

    raw_list, trend_arr, ideas_arr, signals_out = to_frontend_json(items, datalab, metrics, signals, card, scores)

    c1 = dump_json(RAW_PATH,     raw_list)
    c2 = dump_json(TRENDS_PATH,  trend_arr)
    c3 = dump_json(IDEAS_PATH,   ideas_arr)
    c4 = dump_json(SIGNALS_PATH, signals_out)

    print(f"[{NOW_KST.isoformat()}] items={len(items)} changed={any([c1,c2,c3,c4])} "
          f"(secrets: NAVER={'ok' if NAVER_CLIENT_ID else 'no'}, OPENAI={'ok' if OPENAI_API_KEY else 'no'}, YT={'ok' if YOUTUBE_API_KEY else 'no'})")

if __name__=="__main__":
    main()
