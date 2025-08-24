/* globals fetch */
const state = { ideas: [], trends: [], raw: [], signals: {} };

function buster() { return '?cb=' + Date.now(); }
async function getJSON(url) { const r = await fetch(url + buster()); if(!r.ok) throw new Error(url); return r.json(); }

async function loadData() {
  try {
    const [ideas,trends,raw,signals] = await Promise.all([
      getJSON('data/ideas.json'),
      getJSON('data/trends.json'),
      getJSON('data/rawitems.json'),
      getJSON('data/signals.json').catch(()=>({}))
    ]);
    state.ideas = Array.isArray(ideas)?ideas:[];
    state.trends= Array.isArray(trends)?trends:[];
    state.raw   = Array.isArray(raw)?raw:[];
    state.signals=signals||{};
  } catch(e){ console.error(e); }
  renderAll();
}

function renderAll(){ renderIdea(); renderScores(); renderKeywords(); renderCommunity(); bindTabs(); }
function textOr(v,a='—'){ return (v==null||v==='')?a:v; }
function setText(id,v){ const el=document.getElementById(id); if(el) el.textContent = (v==null?'—':v); }

function renderIdea(){
  const idea = state.ideas[0]||{};
  document.getElementById('nav-date').textContent = new Date().toISOString().slice(0,10);
  document.getElementById('idea-title').textContent = textOr(idea.title_ko||idea.title,'Idea of the Day');
  document.getElementById('idea-tags').innerHTML = (idea.tags||[]).map(t=>`<span class="badge">${t}</span>`).join('');

  setText('field-problem', textOr(idea.problem,'알 수 없습니다'));
  setText('field-solution', textOr(idea.solution,'알 수 없습니다'));
  setText('field-target', textOr(idea.target_user,'알 수 없습니다'));
  setText('field-gtm',     textOr(idea.gtm_tactics||idea.gtm,'알 수 없습니다'));

  // Why
  document.getElementById('whyBody').textContent = textOr(idea.why_now,'');
  const whyCards = Array.isArray(idea.why_cards)?idea.why_cards:[];
  document.getElementById('whyCards').innerHTML = whyCards.length? whyCards.map(t=>`<div class="card">${t}</div>`).join('') : '';

  // Proof
  let proofs=[];
  if(Array.isArray(idea.evidence)&&idea.evidence.length){
    proofs = idea.evidence.slice(0,6).map(e=>({title:e.title,url:e.url,source_platform:''}));
  } else {
    proofs = [...state.raw].sort((a,b)=>
      (b.metrics_upvotes||0)+(b.metrics_comments||0) - ((a.metrics_upvotes||0)+(a.metrics_comments||0))
    ).slice(0,6);
  }
  document.getElementById('proofCards').innerHTML = proofs.map(r=>cardLink(r.title,r.url,r.source_platform)).join('');

  // Gap / Exec
  const gaps = Array.isArray(idea.gap_notes)?idea.gap_notes:[];
  const exec = Array.isArray(idea.exec_steps)?idea.exec_steps:[];
  document.getElementById('gapCards').innerHTML  = gaps.length? gaps.map(t=>`<div class="card">${t}</div>`).join('') : `<div class="card muted">근거가 부족합니다</div>`;
  document.getElementById('execCards').innerHTML = exec.length? exec.map(t=>`<div class="card">${t}</div>`).join('') : `<div class="card muted">근거가 부족합니다</div>`;
}

function cardLink(title,url,src){ const s=src?`<div class="mini muted">${src}</div>`:''; return `<div class="card"><div><a class="link" href="${url||'#'}" target="_blank" rel="noopener">${title||'제목 없음'}</a>${s}</div></div>`; }

function renderScores(){
  const idea = state.ideas[0]||{};
  const s = idea.score_breakdown||{};
  let trend=s.trend, market=s.market, comp=s.competition_invert, feas=s.feasibility, mon=s.monetization, reg=s.regulatory_invert, overall=idea.score_total;

  if([trend,market,comp,feas,mon,reg,overall].some(v=>v==null)){
    const vol = avg(state.trends.map(t=>+t.volume||0));
    const gr  = avg(state.trends.map(t=>(+t.growth_percent||0)*100));
    trend = Math.min(100, Math.round(vol*0.5 + gr*0.8));
    market= Math.min(100, state.raw.length*2);
    comp  = Math.max(0, 100 - Math.min(90, (state.raw.length/5)*10));
    feas  = 50; mon=50; reg=50;
    overall = Math.round(0.35*trend + 0.25*market + 0.15*comp + 0.25*50);
  }
  setText('scoreOverall', overall);
  setText('sTrend', trend); setText('sMarket', market); setText('sComp', comp);
  setText('sFeas', feas); setText('sMon', mon); setText('sReg', reg);
}

function avg(a){ if(!a.length) return 0; return a.reduce((x,y)=>x+y,0)/a.length; }

// ---- Keywords & Chart ----
function isBadKw(s){
  if(!s) return true;
  const bad = /^(https?|www|com|co|kr|net|news)$/i;
  if (bad.test(s)) return true;
  const blacklist = new Set(["매일경제","한국경제","한겨레","조선일보","중앙일보","연합뉴스","네이버","다음","네이트"]);
  if (blacklist.has(s)) return true;
  return false;
}

function renderKeywords(){
  const list = state.trends.filter(t=>!isBadKw(t.keyword));
  const sel = document.getElementById('kwSelect');
  sel.innerHTML = list.map((t,i)=>`<option value="${i}">${t.keyword}</option>`).join('');
  if(list.length){
    sel.value="0";
    sel.onchange = ()=> drawTrend(+sel.value,list);
    drawTrend(0,list);
  } else {
    document.getElementById('trendSvg').innerHTML='';
    document.getElementById('chartEmpty').style.display='block';
    setText('kwVol','—'); setText('kwGrowth','—');
  }
}

function drawTrend(idx,arr){
  const t = arr[idx];
  setText('kwVol', t.volume??'—');
  const gp = (typeof t.growth_percent==='number')? Math.round(t.growth_percent*1000)/10 : null;
  setText('kwGrowth', gp==null?'—':(gp+'%'));

  const svg = document.getElementById('trendSvg');
  const empty = document.getElementById('chartEmpty');
  const series = Array.isArray(t.series)?t.series:[];
  if(!series.length){ svg.innerHTML=''; empty.style.display='block'; return; }
  empty.style.display='none';

  const W=600,H=260,P=10;
  const xs = series.map((p,i)=>i);
  const ys = series.map(p=>+p.value||0);
  const xmin=0,xmax=Math.max(1,xs[xs.length-1]||1);
  const ymin=Math.min(...ys), ymax=Math.max(...ys,1);
  const X=x=> P + (x-xmin)/(xmax-xmin) * (W-2*P);
  const Y=y=> H-P - (y-ymin)/(ymax-ymin||1) * (H-2*P);
  let d='';
  series.forEach((p,i)=>{ const x=X(xs[i]), y=Y(ys[i]); d+= (i===0?`M ${x} ${y}`:` L ${x} ${y}`); });
  svg.innerHTML = `<rect x="0" y="0" width="${W}" height="${H}" fill="transparent"/><path d="${d}" fill="none" stroke="#7cc6ff" stroke-width="2"/>`;
}

// ---- Community ----
function renderCommunity(){
  const s=state.signals||{};
  const el=document.getElementById('communityPanel');
  const rows=[];
  if(s.reddit)  rows.push(line('reddit',  `posts ${s.reddit.posts||0} · 👍 ${s.reddit.upvotes||0} · 💬 ${s.reddit.comments||0}`));
  if(s.youtube) rows.push(line('YouTube',`videos ${s.youtube.videos||0}`));
  if(s.naver)   rows.push(line('naver',  `groups ${s.naver.groups||0}`));
  el.innerHTML = rows.join('') || '<div class="muted">근거가 부족합니다</div>';
}
function line(a,b){ return `<div class="mini"><b>${a}</b> — ${b}</div>`; }

// ---- Tabs ----
function bindTabs(){
  const btns=document.querySelectorAll('.tab-btn');
  const panels={'why':id('tab-why'),'proof':id('tab-proof'),'gap':id('tab-gap'),'exec':id('tab-exec')};
  btns.forEach(btn=>{
    btn.onclick=()=>{
      btns.forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const k=btn.dataset.tab;
      Object.values(panels).forEach(p=>p.classList.remove('active'));
      (panels[k]||panels['why']).classList.add('active');
    };
  });
}
function id(s){ return document.getElementById(s); }

document.addEventListener('DOMContentLoaded', loadData);
