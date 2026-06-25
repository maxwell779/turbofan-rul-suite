const $ = (s) => document.querySelector(s);
const C = {cnn:'#3b82f6', lstm:'#a855f7', true:'#22c55e', pred:'#3b82f6'};
const j = (u) => fetch(u).then(r => r.json());
const NS = 'http://www.w3.org/2000/svg';
function el(t, a={}, kids=[]) { const e=document.createElementNS(NS,t); for(const k in a)e.setAttribute(k,a[k]); kids.forEach(c=>e.appendChild(c)); return e; }
function svg(w,h){ const s=el('svg',{viewBox:`0 0 ${w} ${h}`,width:'100%',height:h}); return s; }

function lineChart(host, series, {h=240, ymin=null, ymax=null, xlab='', fmt=(v)=>v}={}) {
  host.innerHTML=''; const w=720, pl=46, pr=14, pt=12, pb=28;
  const xs = series[0].pts.map(p=>p.x);
  const allY = series.flatMap(s=>s.pts.map(p=>p.y));
  const y0 = ymin!=null?ymin:Math.min(...allY), y1 = ymax!=null?ymax:Math.max(...allY);
  const x0=Math.min(...xs), x1=Math.max(...xs);
  const X=(x)=>pl+(x1===x0?0:(x-x0)/(x1-x0))*(w-pl-pr);
  const Y=(y)=>pt+(1-(y-y0)/((y1-y0)||1))*(h-pt-pb);
  const s=svg(w,h);
  for(let i=0;i<=4;i++){const yy=y0+(y1-y0)*i/4; const py=Y(yy);
    s.appendChild(el('line',{x1:pl,y1:py,x2:w-pr,y2:py,stroke:'#eef2f7'}));
    s.appendChild(el('text',{x:pl-6,y:py+4,'text-anchor':'end','font-size':10,fill:'#94a3b8'})).textContent=fmt(yy);}
  series.forEach(se=>{
    const d=se.pts.map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`).join(' ');
    s.appendChild(el('path',{d,fill:'none',stroke:se.color,'stroke-width':se.w||2.2,opacity:se.op||1, 'stroke-dasharray':se.dash||''}));
  });
  s.appendChild(el('text',{x:(w)/2,y:h-4,'text-anchor':'middle','font-size':10,fill:'#94a3b8'})).textContent=xlab;
  host.appendChild(s);
}

function barGroups(host, labels, series, {h=240, fmt=(v)=>v, ymax=null}={}) {
  host.innerHTML=''; const w=720, pl=46, pr=14, pt=14, pb=30;
  const all=series.flatMap(s=>s.vals); const y1=ymax!=null?ymax:Math.max(...all)*1.12;
  const s=svg(w,h); const gw=(w-pl-pr)/labels.length, bw=gw/(series.length+1);
  const Y=(v)=>pt+(1-v/y1)*(h-pt-pb);
  for(let i=0;i<=4;i++){const v=y1*i/4,py=Y(v); s.appendChild(el('line',{x1:pl,y1:py,x2:w-pr,y2:py,stroke:'#eef2f7'}));
    s.appendChild(el('text',{x:pl-6,y:py+4,'text-anchor':'end','font-size':10,fill:'#94a3b8'})).textContent=fmt(v);}
  labels.forEach((lab,gi)=>{ const gx=pl+gi*gw;
    series.forEach((se,si)=>{ const v=se.vals[gi]; if(v==null)return; const x=gx+bw*(si+0.5), py=Y(v);
      s.appendChild(el('rect',{x,y:py,width:bw*0.8,height:h-pb-py,rx:3,fill:se.color}));
      s.appendChild(el('text',{x:x+bw*0.4,y:py-4,'text-anchor':'middle','font-size':9,fill:'#475569'})).textContent=fmt(v);});
    s.appendChild(el('text',{x:gx+gw/2,y:h-10,'text-anchor':'middle','font-size':11,fill:'#475569'})).textContent=lab;});
  host.appendChild(s);
  const lg=document.createElement('div'); lg.className='legend';
  lg.innerHTML=series.map(se=>`<span><i style="background:${se.color}"></i>${se.name}</span>`).join('');
  host.appendChild(lg);
}

function scatter(host, pts, {h=240,max=130}={}) {
  host.innerHTML=''; const w=720,pl=44,pr=14,pt=12,pb=30; const s=svg(w,h);
  const X=(v)=>pl+v/max*(w-pl-pr), Y=(v)=>pt+(1-v/max)*(h-pt-pb);
  s.appendChild(el('line',{x1:X(0),y1:Y(0),x2:X(max),y2:Y(max),stroke:'#cbd5e1','stroke-dasharray':'4 4'}));
  pts.forEach(p=>s.appendChild(el('circle',{cx:X(p.true),cy:Y(p.pred),r:3,fill:'#3b82f6',opacity:.5})));
  s.appendChild(el('text',{x:w/2,y:h-6,'text-anchor':'middle','font-size':10,fill:'#94a3b8'})).textContent='실제 RUL →';
  host.appendChild(s);
}

function spark(host, name, vals) {
  const w=200,h=70,pl=4,pr=4,pt=8,pb=4; const y0=Math.min(...vals),y1=Math.max(...vals);
  const s=svg(w,h); const X=(i)=>pl+i/(vals.length-1)*(w-pl-pr), Y=(v)=>pt+(1-(v-y0)/((y1-y0)||1))*(h-pt-pb);
  s.appendChild(el('path',{d:vals.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(' '),fill:'none',stroke:'#f59e0b','stroke-width':1.6}));
  const box=document.createElement('div'); box.className='sc';
  box.innerHTML=`<div class="lab">${name}</div>`; box.appendChild(s); host.appendChild(box);
}

let RUNS=[];
async function init(){
  const lb=(await j('/api/leaderboard')).rows;
  RUNS=(await j('/api/runs')).runs;
  const done=RUNS.filter(r=>r.test);
  $('#status').textContent = done.length? `모델 ${done.length}개 학습완료` : '학습 진행 중…';

  // KPI: 각 FD의 best(test rmse)
  const byfd={}; lb.forEach(r=>{ if(!byfd[r.fd]||r.rmse<byfd[r.fd].rmse) byfd[r.fd]=r; });
  const tones={FD001:'#3b82f6',FD002:'#f59e0b',FD003:'#22c55e',FD004:'#ef4444'};
  $('#kpis').innerHTML = ['FD001','FD002','FD003','FD004'].map(fd=>{const r=byfd[fd];
    return `<div class="kpi"><div class="bar" style="background:${tones[fd]}"></div>
      <div class="k">${fd} best RMSE</div><div class="v">${r?r.rmse.toFixed(2):'—'}</div>
      <div class="s">${r?r.model.toUpperCase()+' · NASA '+Math.round(r.nasa_score):'학습 대기'}</div></div>`;}).join('');

  // 리더보드 바
  const fds=['FD001','FD002','FD003','FD004'];
  const val=(fd,m,k)=>{const r=lb.find(x=>x.fd===fd&&x.model===m);return r?r[k]:null;};
  barGroups($('#lb_rmse'),fds,[
    {name:'CNN',color:C.cnn,vals:fds.map(f=>val(f,'cnn','rmse'))},
    {name:'LSTM',color:C.lstm,vals:fds.map(f=>val(f,'lstm','rmse'))}],{fmt:v=>v.toFixed(1)});
  barGroups($('#lb_nasa'),fds,[
    {name:'CNN',color:C.cnn,vals:fds.map(f=>val(f,'cnn','nasa_score'))},
    {name:'LSTM',color:C.lstm,vals:fds.map(f=>val(f,'lstm','nasa_score'))}],{fmt:v=>v>=1000?(v/1000).toFixed(1)+'k':Math.round(v)});

  // 표
  let html='<table><tr><th>FD</th><th>모델</th><th>val RMSE(무누수)</th><th>test RMSE</th><th>NASA score</th></tr>';
  const best=Math.min(...lb.map(r=>r.rmse));
  RUNS.forEach(r=>{ html+=`<tr class="${r.test&&r.test.rmse===best?'best':''}"><td>${r.fd}</td><td>${r.model.toUpperCase()}</td>
    <td>${r.val?r.val.rmse.toFixed(2):'—'}</td><td>${r.test?r.test.rmse.toFixed(2):'…'}</td>
    <td>${r.test?Math.round(r.test.nasa_score):'…'}</td></tr>`;});
  $('#table').innerHTML=html+'</table>';

  // 셀렉터
  const fdsAvail=[...new Set(RUNS.map(r=>r.fd))];
  $('#sel_fd').innerHTML=fdsAvail.map(f=>`<option>${f}</option>`).join('');
  syncModels(); $('#sel_fd').onchange=()=>{syncModels();loadEngineList();};
  $('#sel_model').onchange=refresh; $('#sel_engine').onchange=drawEngine;
  await loadEngineList();
}
function syncModels(){ const fd=$('#sel_fd').value;
  const ms=RUNS.filter(r=>r.fd===fd&&r.test).map(r=>r.model);
  $('#sel_model').innerHTML=(ms.length?ms:['cnn']).map(m=>`<option>${m}</option>`).join(''); }
async function loadEngineList(){ const fd=$('#sel_fd').value;
  const es=(await j('/api/engines?fd='+fd)).engines.sort((a,b)=>a.true_rul-b.true_rul);
  $('#sel_engine').innerHTML=es.map(e=>`<option value="${e.unit}">엔진 ${e.unit} (실제RUL ${e.true_rul})</option>`).join('');
  refresh(); }
async function refresh(){ await drawEngine(); await drawHist(); await drawPred(); }

async function drawEngine(){ const fd=$('#sel_fd').value,m=$('#sel_model').value,u=$('#sel_engine').value;
  if(!u)return; const d=await j(`/api/engine?fd=${fd}&model=${m}&unit=${u}`);
  lineChart($('#engine_chart'),[
    {pts:d.cycle.map((x,i)=>({x,y:d.true_rul[i]})),color:C.true,w:2.4},
    {pts:d.cycle.map((x,i)=>({x,y:d.pred_rul[i]})),color:C.pred,w:2.4,dash:'5 4'}],
    {h:260,ymin:0,xlab:'사이클',fmt:v=>Math.round(v)});
  const lg=`<div class="legend"><span><i style="background:${C.true}"></i>실제 RUL</span><span><i style="background:${C.pred}"></i>예측 RUL(${m.toUpperCase()})</span></div>`;
  $('#engine_chart').insertAdjacentHTML('beforeend',lg);
  const sc=$('#sensor_charts'); sc.innerHTML='';
  Object.entries(d.sensors).forEach(([k,v])=>spark(sc,'센서 '+k,v));
}
async function drawHist(){ const fd=$('#sel_fd').value,m=$('#sel_model').value;
  const h=(await j(`/api/history?fd=${fd}&model=${m}`)).history; if(!h.length)return;
  lineChart($('#hist_chart'),[{pts:h.map(e=>({x:e.epoch,y:e.rmse})),color:C.cnn,w:2.2}],
    {h:240,xlab:'에폭',fmt:v=>v.toFixed(0)});
  $('#hist_sub').textContent=`${fd}/${m.toUpperCase()} · 검증 RMSE/에폭 (무누수 val)`; }
async function drawPred(){ const fd=$('#sel_fd').value,m=$('#sel_model').value;
  const rows=(await j(`/api/pred?fd=${fd}&model=${m}`)).rows; if(!rows.length)return;
  scatter($('#pred_chart'),rows); $('#pred_sub').textContent=`${fd}/${m.toUpperCase()} · test 엔진별 (점선=y=x)`; }

init();
