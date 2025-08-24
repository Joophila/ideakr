/* globals fetch */
const state = {
  ideas: [],
  trends: [],
  raw: [],
  signals: {}
};

function buster() { return '?cb=' + Date.now(); }
async function getJSON(url) {
  const r = await fetch(url + buster());
  if (!r.ok) throw new Error('HTTP ' + r.status + ' @ ' + url);
  return r.json();
}

async function loadData() {
  try {
    const [ideas, trends, raw, signals] = await Promise.all([
      getJSON('data/ideas.json'),
      getJSON('data/trends.json'),
      getJSON('data/rawitems.json'),
      getJSON('data/signals.json').catch(() => ({}))
    ]);
    // 그대로 사용 (예전 스키마 전제)
    state.ideas   = Array.isArray(ideas) ? ideas : [];
    state.trends  = Array.isArray(trends) ? trends : [];
    state.raw     = Array.isArray(raw) ? raw : [];
    state.signals = signals || {};
  } catch (e) {
    console.error('loadData failed', e);
  }
  renderAll();
}

function renderAll() {
  renderIdea();
  renderScores();
  renderKeywords();
  renderCommunity();
  bindTabs();
}

/* ---------- Idea card ---------- */
function textOr(v, alt='—') { return (v == null || v === '') ? alt : v; }

function renderIdea() {
  const idea = state.ideas[0] || {};
  document.getElementById('nav-date').textContent = new Date().toISOString().slice(0,10);
  document.getElementById('idea-title').textContent = textOr(idea.title_ko || idea.title, 'Idea of the Day');

  const tagsEl = document.getElementById('idea-tags');
  tagsEl.innerHTML = (idea.tags || []).map(t => `<span class="badge">${t}</span>`).join('');

  document.getElementById('field-problem').textContent = textOr(idea.problem, '알 수 없습니다');
  document.getElementById('field-solution').textContent = textOr(idea.solution, '알 수 없습니다');
  document.getElementById('field-target').textContent = textOr(idea.target_user, '알 수 없습니다');
  document.getElementById('field-gtm').textContent = textOr(idea.gtm_tactics || idea.gtm, '알 수 없습니다');

  // Why Now 본문
  document.getElementById('whyBody').textContent = textOr(idea.why_now, '근거가 부족합니다');

  // Proof cards: raw 중 상위 6개
  const topRaw = [...state.raw].sort((a,b) =>
    (b.metrics_upvotes||0)+(b.metrics_comments||0) - ((a.metrics_upvotes||0)+(a.metrics_comments||0))
  ).slice(0,6);
  document.getElementById('proofCards').innerHTML = topRaw.map(r => cardLink(r.title, r.url, r.source_platform)).join('');

  // Gap/Exec: 데이터가 별도 없으면 기본 문구
  document.getElementById('gapCards').innerHTML  = emptyCard('근거가 부족합니다');
  document.getElementById('execCards').innerHTML = emptyCard('근거가 부족합니다');
}

function cardLink(title, url, sub) {
  const src = sub ? `<div class="mini muted">${sub}</div>` : '';
  return `<div class="card"><div><a class="link" href="${url||'#'}" target="_blank" rel="noopener">${title||'제목 없음'}</a>${src}</div></div>`;
}
function emptyCard(msg) {
  return `<div class="card muted">${msg}</div>`;
}

/* ---------- Scores ---------- */
function renderScores() {
  const idea = state.ideas[0] || {};
  // 아이디어가 점수를 내놨으면 사용, 아니면 대충 계산
  const s = idea.score_breakdown || {};
  let trend = s.trend, market = s.market, comp = s.competition_invert, feas = s.feasibility, mon = s.monetization, reg = s.regulatory_invert, overall = idea.score_total;

  if ([trend,market,comp,feas,mon,reg,overall].some(v => v == null)) {
    // 간단 산식
    const vol = avg(state.trends.map(t => +t.volume || 0));
    const gr  = avg(state.trends.map(t => +t.growth_percent || 0)) * 100;
    trend = Math.min(100, Math.round(vol*0.5 + gr*0.8));
    market= Math.min(100, state.raw.length*2);
    comp  = Math.max(0, 100 - Math.min(90, (state.raw.length/5)*10));
    feas  = 50; mon=50; reg=50;
    overall = Math.round(0.35*trend + 0.25*market + 0.15*comp + 0.25*50);
  }

  setText('scoreOverall', overall);
  setText('sTrend', trend);
  setText('sMarket', market);
  setText('sComp', comp);
  setText('sFeas', feas);
  setText('sMon', mon);
  setText('sReg', reg);
}

function setText(id, v){ const el=document.getElementById(id); if(el) el.textContent = (v==null?'—':v); }
function avg(a){ if(!a.length) return 0; return a.reduce((x,y)=>x+y,0)/a.length; }

/* ---------- Keyword & Chart ---------- */
function renderKeywords() {
  const sel = document.getElementById('kwSelect');
  sel.innerHTML = state.trends.map((t,i)=>`<option value="${i}">${t.keyword}</option>`).join('');
  if (state.trends.length) {
    sel.value = "0";
    sel.onchange = () => drawTrend(+sel.value);
    drawTrend(0);
  } else {
    // 차트 비움
    document.getElementById('trendSvg').innerHTML = '';
    document.getElementById('chartEmpty').style.display = 'block';
    setText('kwVol','—'); setText('kwGrowth','—');
  }
}

function drawTrend(idx) {
  const t = state.trends[idx];
  if (!t) return;
  setText('kwVol', t.volume ?? '—');
  const gp = (typeof t.growth_percent === 'number') ? Math.round(t.growth_percent*1000)/10 : null;
  setText('kwGrowth', gp==null ? '—' : (gp+'%'));

  const svg = document.getElementById('trendSvg');
  const empty = document.getElementById('chartEmpty');

  const series = Array.isArray(t.series) ? t.series : [];
  if (!series.length) {
    svg.innerHTML = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';

  // scale
  const W=600, H=260, P=10;
  const xs = series.map((p,i)=>i);
  const ys = series.map(p => +p.value || 0);
  const xmin=0, xmax=Math.max(1, xs[xs.length-1] || 1);
  const ymin=Math.min(...ys), ymax=Math.max(...ys,1);
  function X(x){ return P + (x-xmin)/(xmax-xmin) * (W-2*P); }
  function Y(y){ return H-P - (y-ymin)/(ymax-ymin || 1) * (H-2*P); }

  let d = '';
  series.forEach((p,i)=>{
    const x=X(xs[i]), y=Y(ys[i]);
    d += (i===0?`M ${x} ${y}`:` L ${x} ${y}`);
  });

  svg.innerHTML = `
    <rect x="0" y="0" width="${W}" height="${H}" fill="transparent"/>
    <path d="${d}" fill="none" stroke="#7cc6ff" stroke-width="2"/>
  `;
}

/* ---------- Community ---------- */
function renderCommunity() {
  const s = state.signals || {};
  const el = document.getElementById('communityPanel');
  const rows = [];
  if (s.reddit)  rows.push(line('reddit',  `posts ${s.reddit.posts||0} · 👍 ${s.reddit.upvotes||0} · 💬 ${s.reddit.comments||0}`));
  if (s.youtube) rows.push(line('YouTube',`videos ${s.youtube.videos||0} · ▶ ${s.youtube.views||0}`));
  if (s.naver)   rows.push(line('naver',  `groups ${s.naver.groups||0} · vol ${s.naver.vol||0}`));
  el.innerHTML = rows.join('') || '<div class="muted">근거가 부족합니다</div>';
}
function line(a,b){ return `<div class="mini"><b>${a}</b> — ${b}</div>`; }

/* ---------- Tabs ---------- */
function bindTabs() {
  const btns = document.querySelectorAll('.tab-btn');
  const panels = {
    'why': document.getElementById('tab-why'),
    'proof': document.getElementById('tab-proof'),
    'gap': document.getElementById('tab-gap'),
    'exec': document.getElementById('tab-exec'),
  };
  btns.forEach(btn=>{
    btn.onclick = ()=>{
      btns.forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const key = btn.dataset.tab;
      Object.values(panels).forEach(p=>p.classList.remove('active'));
      (panels[key]||panels['why']).classList.add('active');
    };
  });
}

document.addEventListener('DOMContentLoaded', loadData);
