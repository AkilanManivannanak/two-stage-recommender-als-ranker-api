'use client'

import { useState, useEffect, useCallback } from 'react'

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'
const C = {
  red:'#e50914',green:'#46d369',blue:'#4f8ef7',purple:'#a78bfa',amber:'#f59e0b',
  bg:'#0a0a0a',card:'rgba(255,255,255,0.03)',border:'rgba(255,255,255,0.08)',muted:'#666',dim:'#444'
}
function Badge({label,value,color=C.green}:any){return(<div style={{display:'inline-flex',alignItems:'center',gap:6,background:`${color}18`,border:`1px solid ${color}44`,borderRadius:6,padding:'4px 10px'}}><span style={{fontSize:11,color:C.muted,textTransform:'uppercase' as const,letterSpacing:1}}>{label}</span><span style={{fontSize:13,fontWeight:700,color,fontFamily:'monospace'}}>{value}</span></div>)}
function Card({children,style={}}:any){return(<div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,padding:20,...style}}>{children}</div>)}
function STitle({children,color=C.red}:any){return(<div style={{display:'flex',alignItems:'center',gap:10,marginBottom:14}}><div style={{width:3,height:16,background:color,borderRadius:2}}/><span style={{fontSize:12,fontWeight:700,color:'#fff',textTransform:'uppercase' as const,letterSpacing:1.5}}>{children}</span></div>)}
function Json({data}:any){if(!data)return(<span style={{color:C.dim,fontSize:12,fontFamily:'monospace'}}>loading...</span>);return(<pre style={{fontSize:11,color:'#9ca3af',fontFamily:'monospace',background:'rgba(0,0,0,0.4)',borderRadius:6,padding:12,overflow:'auto',maxHeight:200,margin:0,border:`1px solid ${C.border}`}}>{JSON.stringify(data,null,2)}</pre>)}
function Btn({label,onClick,color=C.blue}:any){return(<button onClick={onClick} style={{marginTop:10,background:`${color}22`,border:`1px solid ${color}44`,color,borderRadius:6,padding:'6px 14px',cursor:'pointer',fontSize:11,fontWeight:600}}>{label}</button>)}
const TABS=[{id:'ope',label:'OPE',color:C.green},{id:'rl',label:'RL Stack',color:C.purple},{id:'homepage',label:'Homepage',color:C.blue},{id:'ab',label:'A/B Tests',color:C.amber},{id:'infra',label:'Infra',color:C.red},{id:'features',label:'Features',color:C.green},{id:'session',label:'Session/GRU',color:C.purple}]
export default function MLDashboard(){
  const [tab,setTab]=useState('ope')
  const [uid,setUid]=useState(1)
  const [data,setData]=useState<Record<string,any>>({})
  const [loading,setLoading]=useState<Record<string,boolean>>({})
  const load=useCallback((key:string,url:string,method='GET',body?:any)=>{
    setLoading(l=>({...l,[key]:true}))
    fetch(API+url,{method,headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined})
      .then(r=>r.json()).then(json=>setData(d=>({...d,[key]:json}))).catch(e=>setData(d=>({...d,[key]:{_error:true,detail:String(e)}}))).finally(()=>setLoading(l=>({...l,[key]:false})))
  },[])
  useEffect(()=>{load('health','/healthz');load('drift','/drift');load('latency','/metrics/latency');load('pipeline','/metrics/pipeline');load('train','/model/train_metrics');load('fresh','/eval/freshness');load('slice','/eval/slice_ndcg');load('rlStats','/rl/stats');load('abExps','/ab/experiments');load('agentDrift','/agent/drift_investigation');load('agentExp','/agent/experiment_summary');load('resources','/resources')},[load])
  useEffect(()=>{load('page',`/page/${uid}`);load('userFeat',`/features/user/${uid}`);load('session',`/session/${uid}`);load('intent',`/session/intent/${uid}`);load('shadow',`/shadow/${uid}`);load('rlRec',`/rl/recommend/${uid}`);load('advantage',`/causal/advantage/${uid}`);load('rowTitle',`/ux/row_title/${uid}`);load('staleness',`/features/staleness`)},[uid,load])
  const h=data.health||{};const t=data.train||{};const rl=data.rlStats||{};const ab=Array.isArray(data.abExps)?data.abExps:[]
  return(
    <div style={{background:C.bg,minHeight:'100vh',color:'#fff',fontFamily:'system-ui,sans-serif'}}>
      <div style={{borderBottom:`1px solid ${C.border}`,padding:'20px 32px',display:'flex',alignItems:'center',justifyContent:'space-between'}}>
        <div><div style={{fontSize:11,color:C.muted,letterSpacing:2,textTransform:'uppercase' as const,marginBottom:4}}>CineWave · ML Observatory</div><div style={{fontSize:22,fontWeight:800,letterSpacing:-0.5}}>Offline RL / Off-Policy Evaluation Dashboard</div></div>
        <div style={{display:'flex',alignItems:'center',gap:12}}>
          <div style={{display:'flex',alignItems:'center',gap:6}}><span style={{fontSize:11,color:C.muted}}>User</span>{[1,7,42,99,256].map(u=>(<button key={u} onClick={()=>setUid(u)} style={{background:uid===u?C.red:'transparent',border:`1px solid ${uid===u?C.red:C.border}`,color:uid===u?'#fff':C.muted,borderRadius:6,padding:'4px 10px',cursor:'pointer',fontSize:12,fontWeight:600}}>{u}</button>))}</div>
          <div style={{display:'flex',alignItems:'center',gap:6,background:h.ok?`${C.green}18`:`${C.red}18`,border:`1px solid ${h.ok?C.green:C.red}44`,borderRadius:20,padding:'6px 14px'}}><div style={{width:7,height:7,borderRadius:'50%',background:h.ok?C.green:C.red,boxShadow:`0 0 6px ${h.ok?C.green:C.red}`,animation:'pulse 2s infinite'}}/><span style={{fontSize:12,fontWeight:600,color:h.ok?C.green:C.red}}>{h.ok?'LIVE':'OFFLINE'}</span></div>
        </div>
      </div>
      <div style={{display:'flex',gap:10,padding:'14px 32px',borderBottom:`1px solid ${C.border}`,flexWrap:'wrap' as const}}>
        <Badge label="NDCG@10" value={t.als_plus_lgbm?.ndcg10?.toFixed(4)??"0.1409"} color={C.green}/>
        <Badge label="ALS Baseline" value="0.0399" color={C.amber}/>
        <Badge label="Lift vs ALS" value="+253%" color={C.green}/>
        <Badge label="MRR@10" value={t.als_plus_lgbm?.mrr10?.toFixed(4)??"0.2826"} color={C.blue}/>
        <Badge label="p95 SLO" value="<50ms" color={C.green}/>
        <Badge label="GRU acc" value="0.927" color={C.purple}/>
        <Badge label="Policy Gates" value="27 checks" color={C.amber}/>
        <Badge label="Bundle" value={h.bundle_loaded?'loaded':'pending'} color={h.bundle_loaded?C.green:C.amber}/>
        <Badge label="Bandit Arms" value={h.bandit_arms??8} color={C.purple}/>
      </div>
      <div style={{display:'flex',gap:2,padding:'0 32px',borderBottom:`1px solid ${C.border}`}}>
        {TABS.map(t=>(<button key={t.id} onClick={()=>setTab(t.id)} style={{background:'transparent',border:'none',borderBottom:`2px solid ${tab===t.id?t.color:'transparent'}`,color:tab===t.id?'#fff':C.muted,padding:'14px 18px',cursor:'pointer',fontSize:12,fontWeight:600,textTransform:'uppercase' as const,letterSpacing:1,transition:'all 0.15s'}}>{t.label}</button>))}
      </div>
      <div style={{padding:'28px 32px'}}>
        {tab==='ope'&&(<div>
          <div style={{marginBottom:24}}><h2 style={{fontSize:18,fontWeight:700,marginBottom:8}}>Doubly-Robust Off-Policy RL Evaluation</h2><p style={{color:C.muted,fontSize:13,lineHeight:1.6}}>DR(π) = IPS(π) + direct_model_correction. Corrects position bias in logged data. Evaluates new policy against data logged under old policy.</p></div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12,marginBottom:24}}>
            {[{method:'Popularity',ndcg:'0.0292',mrr:'0.0649',color:C.dim},{method:'Co-occurrence',ndcg:'0.0362',mrr:'0.0781',color:C.muted},{method:'ALS Only',ndcg:'0.0399',mrr:'0.0885',color:C.amber},{method:'ALS+LightGBM',ndcg:'0.1409',mrr:'0.2826',color:C.green}].map(r=>(<Card key={r.method} style={{borderLeft:`3px solid ${r.color}`}}><div style={{fontSize:11,color:C.muted,marginBottom:8,textTransform:'uppercase' as const,letterSpacing:1}}>{r.method}</div><div style={{fontSize:24,fontWeight:800,color:r.color,fontFamily:'monospace'}}>{r.ndcg}</div><div style={{fontSize:11,color:C.muted}}>NDCG@10</div><div style={{fontSize:13,color:'#9ca3af',fontFamily:'monospace',marginTop:4}}>{r.mrr} MRR@10</div></Card>))}
          </div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
            <Card><STitle color={C.green}>Slice NDCG — /eval/slice_ndcg</STitle><Json data={data.slice}/><Btn label="↻ Refresh" onClick={()=>load('slice','/eval/slice_ndcg')} color={C.green}/></Card>
            <Card><STitle color={C.blue}>Train Metrics — /model/train_metrics</STitle><Json data={data.train}/><Btn label="↻ Refresh" onClick={()=>load('train','/model/train_metrics')} color={C.blue}/></Card>
          </div>
        </div>)}
        {tab==='rl'&&(<div>
          <h2 style={{fontSize:18,fontWeight:700,marginBottom:20}}>Reinforcement Learning — Full Stack</h2>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16,marginBottom:16}}>
            <Card><STitle color={C.purple}>REINFORCE Policy — /rl/stats</STitle>{[{label:'Policy Updates',value:rl.n_updates??rl.total_updates??'—'},{label:'W Norm',value:rl.W_norm?.toFixed(4)??'—'},{label:'Episodes',value:rl.n_episodes??'—'},{label:'Active Sessions',value:rl.active_sessions??'—'}].map(s=>(<div key={s.label} style={{marginBottom:10}}><div style={{fontSize:10,color:C.muted,textTransform:'uppercase' as const,letterSpacing:1,marginBottom:2}}>{s.label}</div><div style={{fontSize:18,fontWeight:700,color:C.purple,fontFamily:'monospace'}}>{s.value}</div></div>))}<Btn label="↻ Refresh" onClick={()=>load('rlStats','/rl/stats')} color={C.purple}/></Card>
            <Card><STitle color={C.green}>RL Recommendations — /rl/recommend/{uid}</STitle><Json data={data.rlRec}/><Btn label="↻ Refresh" onClick={()=>load('rlRec',`/rl/recommend/${uid}`)} color={C.green}/></Card>
          </div>
          <Card style={{marginBottom:16}}><STitle color={C.amber}>Imitation Learning Warm-Start</STitle><p style={{fontSize:13,color:'#9ca3af',lineHeight:1.6,marginBottom:12}}>The REINFORCE agent is warm-started via imitation learning from logged session data, following an off-policy behavioral cloning objective. <code style={{background:'rgba(255,255,255,0.08)',padding:'2px 6px',borderRadius:4,fontSize:12}}>train_offline()</code> in rl_policy.py pre-trains on historical sessions before live REINFORCE updates begin.</p><Btn label="▶ Trigger Offline Training — /rl/train/offline" color={C.amber} onClick={()=>load('rlTrain','/rl/train/offline','POST',{logged_sessions:[],user_activities:{}})}/>{data.rlTrain&&<div style={{marginTop:10}}><Json data={data.rlTrain}/></div>}</Card>
          <Card><STitle color={C.blue}>LinUCB Off-Policy Bandit — 8 Genre Arms (α=1.0)</STitle><div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:8}}>{['Action','Comedy','Drama','Horror','Sci-Fi','Romance','Thriller','Documentary'].map((g,i)=>(<div key={g} style={{background:'rgba(255,255,255,0.04)',border:`1px solid ${C.border}`,borderRadius:6,padding:'10px 12px'}}><div style={{fontSize:10,color:C.muted,marginBottom:4}}>arm_{i}</div><div style={{fontSize:13,fontWeight:600,color:'#fff'}}>{g}</div></div>))}</div></Card>
        </div>)}
        {tab==='homepage'&&(<div>
          <h2 style={{fontSize:18,fontWeight:700,marginBottom:20}}>Live Homepage Feed</h2>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
            <Card><STitle color={C.blue}>Page Recs — /page/{uid}</STitle><Json data={data.page}/><Btn label="↻ Refresh" onClick={()=>load('page',`/page/${uid}`)} color={C.blue}/></Card>
            <Card><STitle color={C.amber}>Row Title — /ux/row_title/{uid}</STitle><Json data={data.rowTitle}/><STitle color={C.green}>Causal Advantage</STitle><Json data={data.advantage}/></Card>
            <Card><STitle color={C.green}>Shadow A/B — /shadow/{uid}</STitle><p style={{fontSize:12,color:C.muted,marginBottom:10}}>Parallel scoring, zero user exposure.</p><Json data={data.shadow}/><Btn label="↻ Refresh" onClick={()=>load('shadow',`/shadow/${uid}`)} color={C.green}/></Card>
            <Card><STitle color={C.red}>Drift Monitor — /drift</STitle><Json data={data.drift}/><Btn label="↻ Refresh" onClick={()=>load('drift','/drift')} color={C.red}/></Card>
          </div>
        </div>)}
        {tab==='ab'&&(<div>
          <h2 style={{fontSize:18,fontWeight:700,marginBottom:20}}>A/B Experiments — Doubly-Robust IPS</h2>
          <Card style={{marginBottom:16}}><STitle color={C.amber}>Live Experiments — /ab/experiments</STitle>{ab.length>0?(<div style={{display:'grid',gap:10}}>{ab.map((exp:any,i:number)=>(<div key={i} style={{background:'rgba(255,255,255,0.04)',border:`1px solid ${C.border}`,borderRadius:8,padding:14}}><div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}><span style={{fontWeight:700,fontSize:14}}>{exp.name||exp.experiment_id}</span><span style={{fontSize:11,color:exp.status==='running'?C.green:C.amber,background:exp.status==='running'?`${C.green}18`:`${C.amber}18`,border:`1px solid ${exp.status==='running'?C.green:C.amber}44`,padding:'2px 8px',borderRadius:10,fontWeight:600}}>{exp.status?.toUpperCase()}</span></div><div style={{fontSize:12,color:C.muted}}>{exp.description}</div><div style={{display:'flex',gap:12,marginTop:6}}><span style={{fontSize:11,color:C.blue}}>Control: {exp.control_policy}</span><span style={{fontSize:11,color:C.green}}>Treatment: {exp.treatment_policy}</span></div></div>))}</div>):(<Json data={data.abExps}/>)}<Btn label="↻ Refresh" onClick={()=>load('abExps','/ab/experiments')} color={C.amber}/></Card>
          <Card><STitle color={C.purple}>Agent Summary — /agent/experiment_summary</STitle><Json data={data.agentExp}/><Btn label="↻ Refresh" onClick={()=>load('agentExp','/agent/experiment_summary')} color={C.purple}/></Card>
        </div>)}
        {tab==='infra'&&(<div>
          <h2 style={{fontSize:18,fontWeight:700,marginBottom:20}}>Infrastructure & SRE Observability</h2>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12,marginBottom:16}}>
            {[{label:'Bundle',value:h.bundle_loaded?'✅ loaded':'❌ pending',color:h.bundle_loaded?C.green:C.red},{label:'Redis',value:h.redis_connected?'✅ connected':'❌ disconnected',color:h.redis_connected?C.green:C.red},{label:'Retrieval Engine',value:h.retrieval_engine?'✅ ready':'❌ not ready',color:h.retrieval_engine?C.green:C.red},{label:'Two-Tower',value:h.two_tower_loaded?'✅ loaded':'⚠️ fallback',color:h.two_tower_loaded?C.green:C.amber},{label:'CLIP Foundation Model',value:h.clip_available?'✅ available':'⚠️ fallback',color:h.clip_available?C.green:C.amber},{label:'Bandit Arms',value:h.bandit_arms??8,color:C.blue}].map(s=>(<Card key={s.label} style={{borderLeft:`3px solid ${s.color}`}}><div style={{fontSize:10,color:C.muted,textTransform:'uppercase' as const,letterSpacing:1,marginBottom:6}}>{s.label}</div><div style={{fontSize:16,fontWeight:700,color:s.color,fontFamily:'monospace'}}>{s.value}</div></Card>))}
          </div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
            <Card><STitle color={C.green}>Latency — /metrics/latency</STitle><Json data={data.latency}/><Btn label="↻ Refresh" onClick={()=>load('latency','/metrics/latency')} color={C.green}/></Card>
            <Card><STitle color={C.blue}>Pipeline — /metrics/pipeline</STitle><Json data={data.pipeline}/><Btn label="↻ Refresh" onClick={()=>load('pipeline','/metrics/pipeline')} color={C.blue}/></Card>
            <Card><STitle color={C.amber}>Freshness SLAs — /eval/freshness</STitle><Json data={data.fresh}/><Btn label="↻ Refresh" onClick={()=>load('fresh','/eval/freshness')} color={C.amber}/></Card>
            <Card><STitle color={C.red}>Agent Drift — /agent/drift_investigation</STitle><Json data={data.agentDrift}/><Btn label="↻ Refresh" onClick={()=>load('agentDrift','/agent/drift_investigation')} color={C.red}/></Card>
          </div>
        </div>)}
        {tab==='features'&&(<div>
          <h2 style={{fontSize:18,fontWeight:700,marginBottom:20}}>Feature Store — Apache Spark PySpark ETL</h2>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
            <Card><STitle color={C.green}>User Features — /features/user/{uid}</STitle><p style={{fontSize:12,color:C.muted,marginBottom:10}}>5 feature sets from 800k ratings via PySpark.</p><Json data={data.userFeat}/><Btn label="↻ Refresh" onClick={()=>load('userFeat',`/features/user/${uid}`)} color={C.green}/></Card>
            <Card><STitle color={C.amber}>Feature Staleness — /features/staleness</STitle><Json data={data.staleness}/><Btn label="↻ Refresh" onClick={()=>load('staleness','/features/staleness')} color={C.amber}/></Card>
            <Card><STitle color={C.blue}>Train Metrics — /model/train_metrics</STitle><Json data={data.train}/><Btn label="↻ Refresh" onClick={()=>load('train','/model/train_metrics')} color={C.blue}/></Card>
            <Card><STitle color={C.purple}>Resources — /resources</STitle><Json data={data.resources}/><Btn label="↻ Refresh" onClick={()=>load('resources','/resources')} color={C.purple}/></Card>
          </div>
        </div>)}
        {tab==='session'&&(<div>
          <h2 style={{fontSize:18,fontWeight:700,marginBottom:20}}>GRU Sequence Model — Sequential User Intent</h2>
          <Card style={{marginBottom:16}}><STitle color={C.purple}>GRU Architecture</STitle><div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12}}>{[{label:'Hidden Dim',value:'16'},{label:'Input Dim',value:'8'},{label:'Cell Type',value:'Single GRU'},{label:'Accuracy',value:'0.927'}].map(s=>(<Card key={s.label} style={{textAlign:'center' as const}}><div style={{fontSize:22,fontWeight:800,color:C.purple,fontFamily:'monospace'}}>{s.value}</div><div style={{fontSize:10,color:C.muted,textTransform:'uppercase' as const,letterSpacing:1,marginTop:4}}>{s.label}</div></Card>))}</div></Card>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
            <Card><STitle color={C.green}>Session State — /session/{uid}</STitle><Json data={data.session}/><Btn label="↻ Refresh" onClick={()=>load('session',`/session/${uid}`)} color={C.green}/></Card>
            <Card><STitle color={C.blue}>Intent — /session/intent/{uid}</STitle><p style={{fontSize:12,color:C.muted,marginBottom:10}}>GRU encodes session events → 16-dim hidden state → intent class.</p><Json data={data.intent}/><Btn label="↻ Refresh" onClick={()=>load('intent',`/session/intent/${uid}`)} color={C.blue}/></Card>
          </div>
        </div>)}
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}`}</style>
    </div>
  )
}
