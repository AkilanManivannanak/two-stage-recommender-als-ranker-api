'use client'
import { useState, useEffect, useCallback } from 'react'

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'
const C = {
  red:'#e50914', green:'#46d369', blue:'#4f8ef7', purple:'#a78bfa',
  amber:'#f59e0b', teal:'#2dd4bf', bg:'#0a0a0a',
  card:'rgba(255,255,255,0.03)', border:'rgba(255,255,255,0.08)', muted:'#555', dim:'#333'
}

function Tag({children, color=C.green}:any) {
  return <span style={{background:`${color}18`,border:`1px solid ${color}44`,color,borderRadius:4,padding:'2px 8px',fontSize:11,fontWeight:600,fontFamily:'monospace'}}>{children}</span>
}

function Card({children, style={}, glow=''}:any) {
  return <div style={{background:C.card,border:`1px solid ${glow||C.border}`,borderRadius:10,padding:20,boxShadow:glow?`0 0 20px ${glow}22`:undefined,...style}}>{children}</div>
}

function STitle({children, color=C.red}:any) {
  return <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:12}}>
    <div style={{width:3,height:14,background:color,borderRadius:2}}/>
    <span style={{fontSize:11,fontWeight:700,color:'#fff',textTransform:'uppercase' as const,letterSpacing:1.5}}>{children}</span>
  </div>
}

function Live({label, value, color=C.green, sub=''}:any) {
  return <div style={{background:'rgba(0,0,0,0.3)',border:`1px solid ${color}33`,borderRadius:8,padding:'10px 14px',marginBottom:8}}>
    <div style={{fontSize:10,color:C.muted,textTransform:'uppercase' as const,letterSpacing:1}}>{label}</div>
    <div style={{fontSize:18,fontWeight:800,color,fontFamily:'monospace',marginTop:2}}>{value??<span style={{color:C.dim}}>—</span>}</div>
    {sub&&<div style={{fontSize:11,color:C.muted,marginTop:2}}>{sub}</div>}
  </div>
}

function Pill({text, color=C.green}:any) {
  return <span style={{background:`${color}15`,border:`1px solid ${color}33`,color,borderRadius:20,padding:'3px 10px',fontSize:11,fontWeight:600,marginRight:6,marginBottom:6,display:'inline-block'}}>{text}</span>
}

function Json({data}:any) {
  if(!data) return <span style={{color:C.dim,fontSize:12}}>loading…</span>
  return <pre style={{fontSize:11,color:'#9ca3af',fontFamily:'monospace',background:'rgba(0,0,0,0.5)',borderRadius:6,padding:10,overflow:'auto',maxHeight:160,margin:0}}>{JSON.stringify(data,null,2)}</pre>
}

function Btn({label,onClick,color=C.blue}:any) {
  return <button onClick={onClick} style={{background:`${color}22`,border:`1px solid ${color}44`,color,borderRadius:6,padding:'5px 12px',cursor:'pointer',fontSize:11,fontWeight:600,marginTop:8}}>{label}</button>
}

export default function AIStackPage() {
  const [data, setData] = useState<Record<string,any>>({})
  const [uid, setUid] = useState(1)

  const load = useCallback((key:string, url:string, method='GET', body?:any) => {
    fetch(API+url, {method, headers:body?{'Content-Type':'application/json'}:undefined, body:body?JSON.stringify(body):undefined})
      .then(r=>r.json()).then(json=>setData(d=>({...d,[key]:json}))).catch(e=>setData(d=>({...d,[key]:{error:String(e)}})))
  }, [])

  useEffect(()=>{
    load('health','/healthz')
    load('rlStats','/rl/stats')
    load('abExps','/ab/experiments')
    load('train','/model/train_metrics')
    load('arch','/architecture')
    load('drift','/drift')
    load('latency','/metrics/latency')
    load('fresh','/eval/freshness')
    load('slice','/eval/slice_ndcg')
  },[load])

  useEffect(()=>{
    load('rlRec',`/rl/recommend/${uid}`)
    load('session',`/session/intent/${uid}`)
    load('page',`/page/${uid}`)
    load('shadow',`/shadow/${uid}`)
    load('userFeat',`/features/user/${uid}`)
  },[uid,load])

  const h=data.health||{}
  const rl=data.rlStats||{}
  const t=data.train||{}

  return (
    <div style={{background:C.bg,minHeight:'100vh',color:'#fff',fontFamily:'system-ui,sans-serif',paddingBottom:60}}>

      {/* Header */}
      <div style={{background:'rgba(229,9,20,0.06)',borderBottom:`1px solid ${C.red}33`,padding:'24px 40px'}}>
        <div style={{fontSize:11,color:C.muted,letterSpacing:3,textTransform:'uppercase' as const,marginBottom:6}}>CineWave · Live ML System</div>
        <div style={{fontSize:28,fontWeight:900,letterSpacing:-1,marginBottom:12}}>AI Stack — Everything Running Live</div>
        <div style={{display:'flex',flexWrap:'wrap' as const,gap:6}}>
          {['Offline RL','Off-Policy RL','Doubly-Robust IPS','Imitation Learning','Multi-Task Learning','GRU Sequence Model','CLIP ViT-B/32 Foundation Model','Apache Spark','Kubernetes HPA','SQL Schema'].map(t=>(
            <Pill key={t} text={t} color={t.includes('GRU')||t.includes('Multi')?C.purple:t.includes('CLIP')||t.includes('Found')?C.teal:t.includes('Spark')||t.includes('SQL')?C.amber:C.green}/>
          ))}
        </div>
      </div>

      {/* User picker */}
      <div style={{padding:'14px 40px',borderBottom:`1px solid ${C.border}`,display:'flex',alignItems:'center',gap:10}}>
        <span style={{fontSize:11,color:C.muted}}>Demo User:</span>
        {[1,7,42,99,256].map(u=>(
          <button key={u} onClick={()=>setUid(u)} style={{background:uid===u?C.red:'transparent',border:`1px solid ${uid===u?C.red:C.border}`,color:uid===u?'#fff':C.muted,borderRadius:6,padding:'4px 12px',cursor:'pointer',fontSize:12,fontWeight:600}}>{u}</button>
        ))}
        <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:6,background:h.ok?`${C.green}15`:`${C.red}15`,border:`1px solid ${h.ok?C.green:C.red}44`,borderRadius:20,padding:'5px 14px'}}>
          <div style={{width:6,height:6,borderRadius:'50%',background:h.ok?C.green:C.red,animation:'pulse 2s infinite'}}/>
          <span style={{fontSize:12,fontWeight:700,color:h.ok?C.green:C.red}}>{h.ok?'ALL SYSTEMS LIVE':'OFFLINE'}</span>
        </div>
      </div>

      <div style={{padding:'28px 40px',display:'grid',gap:20}}>

        {/* ── Row 1: NDCG baseline comparison ── */}
        <div>
          <div style={{fontSize:13,fontWeight:700,color:C.muted,textTransform:'uppercase' as const,letterSpacing:2,marginBottom:12}}>📊 Offline RL / Off-Policy Evaluation — Doubly-Robust IPS</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12}}>
            {[
              {method:'Popularity',ndcg:'0.0292',mrr:'0.0649',color:C.dim,tag:'baseline'},
              {method:'Co-occurrence',ndcg:'0.0362',mrr:'0.0781',color:C.muted,tag:'baseline'},
              {method:'ALS Only',ndcg:'0.0399',mrr:'0.0885',color:C.amber,tag:'collaborative'},
              {method:'ALS + LightGBM',ndcg:'0.1409',mrr:'0.2826',color:C.green,tag:'✅ +253% deployed'},
            ].map(r=>(
              <Card key={r.method} glow={r.color===C.green?C.green:''} style={{borderLeft:`3px solid ${r.color}`}}>
                <div style={{fontSize:10,color:C.muted,marginBottom:6,textTransform:'uppercase' as const,letterSpacing:1}}>{r.method}</div>
                <div style={{fontSize:28,fontWeight:900,color:r.color,fontFamily:'monospace'}}>{r.ndcg}</div>
                <div style={{fontSize:11,color:C.muted}}>NDCG@10</div>
                <div style={{fontSize:13,color:'#9ca3af',fontFamily:'monospace',marginTop:4}}>{r.mrr} MRR@10</div>
                <div style={{marginTop:8}}><Tag color={r.color===C.green?C.green:C.dim}>{r.tag}</Tag></div>
              </Card>
            ))}
          </div>
        </div>

        {/* ── Row 2: RL Stack + GRU + Multi-Task ── */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:16}}>

          {/* REINFORCE + Imitation Learning */}
          <Card glow={C.purple}>
            <STitle color={C.purple}>🤖 REINFORCE + Imitation Learning</STitle>
            <Live label="Policy Updates" value={rl.n_updates??rl.total_updates??'—'} color={C.purple}/>
            <Live label="W Norm" value={rl.W_norm?.toFixed(4)??'—'} color={C.blue} sub="policy weight magnitude"/>
            <div style={{background:'rgba(245,158,11,0.08)',border:'1px solid rgba(245,158,11,0.2)',borderRadius:6,padding:10,marginTop:10}}>
              <div style={{fontSize:10,color:C.amber,fontWeight:700,marginBottom:4}}>IMITATION LEARNING WARM-START</div>
              <div style={{fontSize:11,color:'#9ca3af',lineHeight:1.5}}>train_offline() pre-trains on logged sessions via behavioral cloning before live REINFORCE updates begin.</div>
              <Btn label="▶ Trigger Offline Training" color={C.amber}
                onClick={()=>load('rlTrain','/rl/train/offline','POST',{logged_sessions:[],user_activities:{}})}/>
              {data.rlTrain&&<div style={{marginTop:8}}><Json data={data.rlTrain}/></div>}
            </div>
          </Card>

          {/* GRU Sequence Model */}
          <Card glow={C.purple}>
            <STitle color={C.purple}>🔁 GRU Sequence Model</STitle>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8,marginBottom:12}}>
              {[{l:'Hidden Dim',v:'16'},{l:'Input Dim',v:'8'},{l:'Cell',v:'Single GRU'},{l:'Accuracy',v:'0.927'}].map(s=>(
                <div key={s.l} style={{background:'rgba(167,139,250,0.08)',border:'1px solid rgba(167,139,250,0.2)',borderRadius:6,padding:'8px 10px',textAlign:'center' as const}}>
                  <div style={{fontSize:18,fontWeight:800,color:C.purple,fontFamily:'monospace'}}>{s.v}</div>
                  <div style={{fontSize:9,color:C.muted,textTransform:'uppercase' as const,letterSpacing:1}}>{s.l}</div>
                </div>
              ))}
            </div>
            <STitle color={C.blue}>Live Session Intent</STitle>
            <Json data={data.session}/>
            <Btn label="↻ Refresh /session/intent" onClick={()=>load('session',`/session/intent/${uid}`)} color={C.blue}/>
          </Card>

          {/* Multi-Task Reward Model */}
          <Card glow={C.teal}>
            <STitle color={C.teal}>⚡ Multi-Task Reward Model</STitle>
            <div style={{fontSize:11,color:'#9ca3af',marginBottom:12,lineHeight:1.5}}>
              Shared-bottom network — 1 encoder → 4 task heads trained jointly via IPS-weighted backprop.
            </div>
            {[
              {task:'Click (play_start)',w:'+1.0',color:C.green},
              {task:'Completion (watch_90%)',w:'+2.0',color:C.green},
              {task:'Add to List',w:'+1.0',color:C.blue},
              {task:'Skip',w:'-0.5',color:C.red},
            ].map(r=>(
              <div key={r.task} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'6px 0',borderBottom:`1px solid ${C.border}`}}>
                <span style={{fontSize:12,color:'#9ca3af'}}>{r.task}</span>
                <span style={{fontSize:13,fontWeight:700,color:r.color,fontFamily:'monospace'}}>{r.w}</span>
              </div>
            ))}
            <div style={{marginTop:10,fontSize:10,color:C.teal,fontFamily:'monospace'}}>multi_task_reward.py · shared_bottom_multi_task ✅</div>
          </Card>
        </div>

        {/* ── Row 3: LinUCB + Live RL Recs ── */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          <Card>
            <STitle color={C.blue}>🎰 LinUCB Off-Policy Bandit — 8 Arms (α=1.0)</STitle>
            <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:8}}>
              {['Action','Comedy','Drama','Horror','Sci-Fi','Romance','Thriller','Documentary'].map((g,i)=>(
                <div key={g} style={{background:'rgba(255,255,255,0.04)',border:`1px solid ${C.border}`,borderRadius:6,padding:'8px 10px',textAlign:'center' as const}}>
                  <div style={{fontSize:9,color:C.muted,marginBottom:3}}>arm_{i}</div>
                  <div style={{fontSize:12,fontWeight:700,color:'#fff'}}>{g}</div>
                  <div style={{fontSize:9,color:C.dim,marginTop:2,fontFamily:'monospace'}}>UCB=μ+α√(xᵀA⁻¹x)</div>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <STitle color={C.green}>🎬 Live RL Recommendations — /rl/recommend/{uid}</STitle>
            <Json data={data.rlRec}/>
            <Btn label="↻ Refresh" onClick={()=>load('rlRec',`/rl/recommend/${uid}`)} color={C.green}/>
          </Card>
        </div>

        {/* ── Row 4: CLIP Foundation Model ── */}
        <div style={{display:'grid',gridTemplateColumns:'2fr 1fr',gap:16}}>
          <Card glow={C.teal}>
            <STitle color={C.teal}>🖼️ CLIP ViT-B/32 — Multimodal Vision-Language Foundation Model</STitle>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
              <div>
                <div style={{fontSize:11,color:C.muted,marginBottom:8}}>Architecture</div>
                {[
                  'Vision Transformer (ViT-B/32) backbone',
                  'Patch embeddings: 32×32 patches → 512-dim tokens',
                  'Multi-head self-attention (12 layers)',
                  '[CLS] token → 512-dim visual embedding',
                  'Projected into shared text-image space',
                  'Pre-trained on 400M image-text pairs (OpenAI)',
                ].map(s=>(<div key={s} style={{fontSize:11,color:'#9ca3af',marginBottom:4,paddingLeft:8,borderLeft:`2px solid ${C.teal}44`}}>· {s}</div>))}
              </div>
              <div>
                <div style={{fontSize:11,color:C.muted,marginBottom:8}}>Live Status</div>
                <Live label="CLIP Available" value={h.clip_available?'✅ loaded':'⚠️ fallback'} color={h.clip_available?C.green:C.amber} sub="colour histogram fallback when not installed"/>
                <Live label="Embedding Dim" value="512-dim" color={C.teal} sub="shared text-image space"/>
                <div style={{marginTop:8,fontSize:10,color:C.teal,fontFamily:'monospace'}}>Foundation model — zero-shot poster understanding</div>
              </div>
            </div>
          </Card>

          <Card>
            <STitle color={C.amber}>☁️ Kubernetes HPA</STitle>
            {[
              {trigger:'CPU',threshold:'> 70%',color:C.red},
              {trigger:'Memory',threshold:'> 80%',color:C.amber},
              {trigger:'RPS/pod',threshold:'> 100',color:C.blue},
            ].map(r=>(
              <div key={r.trigger} style={{display:'flex',justifyContent:'space-between',padding:'8px 0',borderBottom:`1px solid ${C.border}`}}>
                <span style={{fontSize:12,color:'#9ca3af'}}>{r.trigger}</span>
                <Tag color={r.color}>{r.threshold} → scale up</Tag>
              </div>
            ))}
            <div style={{marginTop:10,fontSize:11,color:C.muted}}>Min: 2 replicas · Max: 10 replicas</div>
            <div style={{marginTop:4,fontSize:10,color:C.amber,fontFamily:'monospace'}}>k8s/hpa.yaml · PDB minAvailable=2</div>
          </Card>
        </div>

        {/* ── Row 5: Apache Spark + SQL ── */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          <Card glow={C.amber}>
            <STitle color={C.amber}>⚡ Apache Spark — PySpark Feature Engineering</STitle>
            <div style={{fontSize:11,color:'#9ca3af',marginBottom:12,lineHeight:1.5}}>
              800k ratings processed via PySpark local[*] mode. Columnar groupBy faster than Python dict loops at this scale.
            </div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8,marginBottom:12}}>
              {[{l:'Ratings',v:'800k'},{l:'Feature Sets',v:'5'},{l:'Co-occurrence',v:'PySpark self-join'},{l:'Mode',v:'local[*]'}].map(s=>(
                <div key={s.l} style={{background:'rgba(245,158,11,0.08)',border:'1px solid rgba(245,158,11,0.2)',borderRadius:6,padding:'8px 10px'}}>
                  <div style={{fontSize:16,fontWeight:800,color:C.amber,fontFamily:'monospace'}}>{s.v}</div>
                  <div style={{fontSize:9,color:C.muted,textTransform:'uppercase' as const,letterSpacing:1,marginTop:2}}>{s.l}</div>
                </div>
              ))}
            </div>
            <STitle color={C.green}>Live Feature Store</STitle>
            <Json data={data.userFeat}/>
            <Btn label={`↻ /features/user/${uid}`} onClick={()=>load('userFeat',`/features/user/${uid}`)} color={C.amber}/>
          </Card>

          <Card glow={C.blue}>
            <STitle color={C.blue}>🗄️ SQL Schema — 4 Tables</STitle>
            <div style={{fontFamily:'monospace',fontSize:11,color:'#9ca3af',background:'rgba(0,0,0,0.4)',borderRadius:6,padding:12,marginBottom:12}}>
              {`CREATE TABLE users (
  user_id BIGINT PRIMARY KEY,
  activity_decile SMALLINT,
  top_genres TEXT[]
);
CREATE TABLE ratings (
  user_id BIGINT REFERENCES users,
  item_id BIGINT,
  rating NUMERIC(3,1)
);
CREATE TABLE recommendations (
  user_id BIGINT, item_id BIGINT,
  rank SMALLINT, als_score NUMERIC,
  rl_score NUMERIC, policy_version TEXT
);
CREATE TABLE events (
  event_type VARCHAR(32), -- play|skip|add
  reward NUMERIC(4,2),
  session_id UUID
);`}
            </div>
            <div style={{fontSize:10,color:C.blue,fontFamily:'monospace'}}>sql/schema.sql · sql/queries.sql · SELECT+JOIN+GROUP BY</div>
          </Card>
        </div>

        {/* ── Row 6: SRE + A/B ── */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:16}}>
          <Card>
            <STitle color={C.red}>🛡️ SRE — Policy Gate (27 Checks)</STitle>
            <Live label="p95 SLO" value="< 50ms" color={C.green} sub="plain /recommend endpoint"/>
            <Live label="Policy Gates" value="27 checks" color={C.amber} sub="all must pass before promotion"/>
            <Live label="Drift (PSI)" value={data.drift?.psi_score?.toFixed(4)??'—'} color={C.blue} sub="training vs serving skew"/>
            <Btn label="↻ /drift" onClick={()=>load('drift','/drift')} color={C.red}/>
          </Card>

          <Card>
            <STitle color={C.green}>⚗️ A/B Experiments</STitle>
            {Array.isArray(data.abExps)&&data.abExps.length>0?(
              data.abExps.slice(0,3).map((e:any,i:number)=>(
                <div key={i} style={{padding:'8px 0',borderBottom:`1px solid ${C.border}`}}>
                  <div style={{fontSize:12,fontWeight:600,color:'#fff',marginBottom:2}}>{e.name||e.experiment_id}</div>
                  <div style={{display:'flex',gap:8}}>
                    <Tag color={C.blue}>{e.control_policy}</Tag>
                    <span style={{color:C.muted,fontSize:11}}>vs</span>
                    <Tag color={C.green}>{e.treatment_policy}</Tag>
                  </div>
                </div>
              ))
            ):<Json data={data.abExps}/>}
            <Btn label="↻ /ab/experiments" onClick={()=>load('abExps','/ab/experiments')} color={C.green}/>
          </Card>

          <Card>
            <STitle color={C.purple}>📊 Latency (p50/p95/p99)</STitle>
            <Json data={data.latency}/>
            <Btn label="↻ /metrics/latency" onClick={()=>load('latency','/metrics/latency')} color={C.purple}/>
          </Card>
        </div>

        {/* ── Row 7: Live page recs ── */}
        <Card>
          <STitle color={C.blue}>🎯 Live Homepage Feed — /page/{uid} (ALS + LightGBM + RL + Bandit)</STitle>
          <Json data={data.page}/>
          <Btn label={`↻ Refresh /page/${uid}`} onClick={()=>load('page',`/page/${uid}`)} color={C.blue}/>
        </Card>

      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}`}</style>
    </div>
  )
}
