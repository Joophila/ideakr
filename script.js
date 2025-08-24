/* globals fetch */
const S = { ideas: [], trends: [], raw: [], signals: {} };

const qs = (s)=>document.querySelector(s);
const el = (id)=>document.getElementById(id);
const txt = (id,v)=>{ const n=el(id); if(n) n.textContent = (v==null||v==='')?'—':v; };
const cacheBust = ()=>'?cb=' + Date.now();
async function j(url){ const r = await fetch(url+cacheBust()); if(!r.ok) throw new Error(url); return r.json(); }

document.addEventListener('DOMContentLoaded', async ()=>{
  try{
    const [ideas,trends,raw,signals] = await Promise.all([
      j('data/ideas.json'), j('data/trends.json'), j('data/rawitems.json'), j('data/signals.json').catch(()=>({}))
    ]);
    S.ideas = Array.isArray(ideas)?ideas:[];
    S.trends= Array.isArray(trends)?trends:[];
    S.raw   = Array.isArray(raw)?raw:[];
    S.signals = signals||{};
  }catch(e){ console.error(e); }
  render();
});

function render(){
  renderIdea();
  renderScores();
  renderKeywords();
  renderCommunity();
  bindTabs();
}

function renderIdea(){
  const idea = S.ideas[0]||{};
  el('nav-date').textContent = new Date().toISOString().slice(0,10);
  el('idea-title').textContent = idea.title_ko || idea.title || 'Idea of the Day';
  el('idea-tags').innerHTML = (idea.tags||[]).map(t=>`<span class="badge">${t}</span>`).join('');

  txt('field-problem', idea.problem || '알 수 없습니다');
  txt('field-solution', idea.solution || '알 수 없습니다');
  txt('field-target', idea.target_user || '알 수 없습니다');
  txt('field-gtm',     idea.gtm_tactics || '알 수 없습니다');

  // Why Now
  const whyCards = Array.isArray(idea.why_cards)?idea.why_cards:[];
  el('whyBody').textContent = idea.why_now || '';
  el('whyCards').innerHTML  = whyCards.map(t=>`<div class="card">${t}</div>`).join('');

  // Proof & Signals (우선 순위: evidence → raw top)
  let proofs=[];
  if(Array.isArray(idea.evidence)&&idea.evidence.length){
    proofs = idea.evidence.slice(0,8).map(e=>({title:e.title,url:e.url,src:'ref'}));
  } else {
    proofs = [...S.raw].sort((a,b)=>
      (b.metrics_upvotes||0)+(b.metrics_comments||0) - ((a.metrics_upvotes||0)+(a.metrics_comments||0))
    ).slice(0,8).map(r=>({title:r.title,url:r.url,src:r.source_platform}));
  }
  el('proofCards').innerHTML = proofs.map(p=>cardLink(p.title,p.url,p.src)).join('');

  // Market Gap
  const gaps = Array.isArray(idea.gap_notes)?idea.gap_notes:[];
  const comp = Array.isArray(idea.competitors)?idea.competitors:[];
  el('gapCards').innerHTML =
    (gaps.map(t=>`<div class="card">${t}</div>`).join('') || `<div class="card muted">근거가 부족합니다</div>`) +
    (comp.length? `<div class="card"><b>Competitors</b><ul style="margin:8px 0;padding-left:18px">${comp.map(c=>`<li>${c}</li>`).join('')}</ul></div>` : '');

  // Execution Plan (+ Offer Ladder / Pricing / Channels / Personas)
  const exec = Array.isArray(idea.exec_steps)?idea.exec_steps:[];
  const offer = Array.isArray(idea.offer_ladder)?idea.offer_ladder:[];
  const pricing = Array.isArray(idea.pricing)?idea.pricing:[];
  const channels = Array.isArray(idea.channels)?idea.channels:[];
  const personas = Array.isArray(idea.personas)?idea.personas:[];
  el('execCards').innerHTML =
    (exec.map(t=>`<div class="card">${t}</div>`).join('') || `<div class="card muted">근거가 부족합니다</div>`) +
    (offer.length? `<div class="card"><b>Offer Ladder</b><ul style="margin:8px 0;padding-left:18px">${offer.map(o=>`<li>${o.name} — ${o.price} (${o.unit})</li>`).join('')}</ul></div>` : '') +
    (pricing.length? `<div class="card"><b>Pricing</b><ul style="margin:8px 0;padding-left:18px">${pricing.map(p=>`<li>${p}</li>`).join('')}</ul></div>` : '') +
    (channels.length? `<div class="card"><b>Channels</b><ul style="margin:8px 0;padding-left:18px">${channels.map(c=>`<li>${c}</li>`).join('')}</ul></div>` : '') +
    (personas.length? `<div class="card"><b>Personas</b><ul style="margin:8px 0;padding-left:18px">${personas.map(p=>`<li>${p.name} — pain: ${p.pain} / JTBD: ${p.jtbd}</li>`).join('')}</ul></div>` : '');
}

function cardLink(title,url,src){
  const ss = src?`<div class="mini muted" style="margin-top:6px">${src}</div>`:'';
  return `<div class="card"><div><a class="link" target="_blank" rel="noopener" href="${url||'#'}">${title||'제목 없음'}</a>${ss}</div></div>`;
}

// --- Scores ---
function renderScores(){
  const idea = S.ideas[0]||{};
  const s = idea.score_breakdown||{};
  let trend=s.trend, market=s.market, comp=s.competition_invert, feas=s.feasibility, mon=s.monetization, reg=s.regulatory_invert, overall=idea.score_total;

  if([trend,market,comp,feas,mon,reg,overall].some(v=>v==null)){
    const vol = avg(S.trends.map(t=>+t.volume||0));
    const gr  = avg(S.trends.map(t=>(+t.growth_percent||0)*100));
    trend = Math.min(100, Math.round(vol*0.5 + gr*0.8));
    market= Math.min(100, S.raw.length*2);
    comp  = Math.max(0, 100 - Math.min(90, (S.raw.length/5)*10));
    feas  = 50; mon=50; reg=50;
    overall = Math.round(0.35*trend + 0.25*market + 0.15*comp + 0.25*50);
  }
  txt('scoreOverall', overall);
  txt('sTrend', trend); txt('sMarket', market); txt('sComp', comp);
  txt('sFeas', feas); txt('sMon', mon); txt('sReg', reg);
}
function avg(a){ if(!a.length) return 0; return a.reduce((x,y)=>x+y,0)/a.length; }

// --- Trend chart ---
function isBadKw(s){
  if(!s) return true;
  const bad = /^(https?|www|com|co|kr|net|news)$/i;
  if (bad.test(s)) return true;
  const blacklist = new Set(["매일경제","한국경제","한겨레","조선일보","중앙일보","연합뉴스","네이버","다음","네이트"]);
  if (blacklist.has(s)) return true;
  return false;
}

function renderKeywords(){
  const list = S.trends.filter(t=>!isBadKw(t.keyword));
  const sel = el('kwSelect');
  sel.innerHTML = list.map((t,i)=>`<option value="${i}">${t.keyword}</option>`).join('');
  if(list.length){
    sel.value="0";
    sel.onchange = ()=> drawTrend(+sel.value,list);
    drawTrend(0,list);
  } else {
    el('trendSvg').innerHTML='';
    el('chartEmpty').style.display='block';
    txt('kwVol','—'); txt('kwGrowth','—');
  }
}

function drawTrend(idx,arr){
  const t = arr[idx];
  txt('kwVol', t.volume??'—');
  const gp = (typeof t.growth_percent==='number')? Math.round(t.growth_percent*1000)/10 : null;
  txt('kwGrowth', gp==null?'—':(gp+'%'));

  const svg = el('trendSvg');
  const empty = el('chartEmpty');
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

// --- Community ---
function renderCommunity(){
  const s=S.signals||{};
  const elp=el('communityPanel');
  const rows=[];
  if(s.reddit)  rows.push(line('reddit',`posts ${s.reddit.posts||0} · 👍 ${s.reddit.upvotes||0} · 💬 ${s.reddit.comments||0}`));
  if(s.youtube) rows.push(line('YouTube',`videos ${s.youtube.videos||0}`));
  if(s.naver)   rows.push(line('naver',`groups ${s.naver.groups||0}`));
  elp.innerHTML = rows.join('') || '<div class="muted">근거가 부족합니다</div>';
}
function line(a,b){ return `<div class="mini"><b>${a}</b> — ${b}</div>`; }

// --- Tabs ---
function bindTabs(){
  const btns=document.querySelectorAll('.tab-btn');
  const panels={'why':el('tab-why'),'proof':el('tab-proof'),'gap':el('tab-gap'),'exec':el('tab-exec')};
  btns.forEach(b=>{
    b.onclick=()=>{
      btns.forEach(x=>x.classList.remove('active'));
      b.classList.add('active');
      const k=b.dataset.tab;
      Object.values(panels).forEach(p=>p.classList.remove('active'));
      (panels[k]||panels['why']).classList.add('active');
    };
  });
}
