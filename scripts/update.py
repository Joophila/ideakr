# (기존 import 아래 아무 데나 추가)
DOMAIN_TOKEN_RE = re.compile(r"(https?://\S+)|([a-z0-9_-]+\.(co|com|kr|net|org)\b)", re.I)

# 뉴스 매체/포털 블랙리스트 (키워드에서 제외)
PUBLISHER_BLACKLIST = {
    "매일경제","한국경제","한경","조선일보","중앙일보","동아일보","한겨레","경향신문","연합뉴스",
    "서울경제","머니투데이","이데일리","전자신문","블로터","네이버","다음","네이트","구글",
    "kbs","mbc","sbs","ytn","jtbc","hankookilbo","chosun","hankyung","zdnet","zumnet"
}

# 한국어/영어 공통 불용어 (필요시 자유롭게 추가)
STOPWORDS = {
    "the","and","for","with","from","into","your","our","new","how","what","why","when","where","who",
    "this","that","are","is","was","to","of","in","on","by","at","as","it",
    "기사","속보","단독","사진","영상","인터뷰","전문","종합","오늘","어제","이번","최근","관련",
    "https","http","com","kr","net","news","뉴스","기자","표지","포토","영상","라이브",
}

def clean_text(s: str) -> str:
    """URL/도메인/특수문자 제거 + 공백 정리"""
    s = s or ""
    s = strip_html(s)
    s = DOMAIN_TOKEN_RE.sub(" ", s)
    s = re.sub(r"[\[\]()<>{}“”\"'`•·…~^_|=:;#@※※]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_ko_en(s: str) -> list[str]:
    """한글/영문/숫자 단어 토큰화 (2자 이상) + 불용어/숫자/언론사 제거"""
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣\s\-]", " ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    out = []
    for t in toks:
        if t.isdigit(): 
            continue
        if t in STOPWORDS: 
            continue
        if t in PUBLISHER_BLACKLIST: 
            continue
        out.append(t)
    return out

def gen_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
