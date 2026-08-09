import { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowRight, BellRing, BrainCircuit, CheckCircle2, ChevronRight, Clock3, Database, Filter, Gauge, LayoutDashboard, Mail, Menu, MessageSquareText, MoreHorizontal, Search, Send, Settings2, ShieldCheck, Sparkles } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Area, AreaChart, Bar, BarChart, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { analyzeGmail, analyzeMessage, connectGmail, disconnectGmail, getAnalytics, getDashboardStats, getGmailDashboard, getGmailResults, getGmailStatus, getMessages, getSystemStatus, submitFeedback } from './services/api';
import type { GmailDashboard, GmailResult, GmailStatus } from './services/api';
import type { Message, DashboardStats } from './types/message';
import type { Action, AnalysisResult, MessageType } from './types/routing';
import './styles.css';

type Page = 'Dashboard'|'Message Router'|'Gmail Intelligence'|'Message History'|'Analytics'|'Rules & Intelligence'|'System';
const nav: { name: Page; icon: typeof LayoutDashboard }[] = [{name:'Dashboard',icon:LayoutDashboard},{name:'Message Router',icon:Send},{name:'Gmail Intelligence',icon:Mail},{name:'Message History',icon:MessageSquareText},{name:'Analytics',icon:Activity},{name:'Rules & Intelligence',icon:BrainCircuit},{name:'System',icon:Settings2}];
const actionClass: Record<Action,string> = { NOTIFY:'notify', DIGEST:'digest', MUTE:'mute' };
const actionDot: Record<Action,string> = { NOTIFY:'bg-teal-400', DIGEST:'bg-indigo-400', MUTE:'bg-rose-400' };
function Badge({ action }: { action: Action }) { return <span className={`badge ${actionClass[action]}`}><i className={actionDot[action]} />{action}</span> }
function Confidence({ value }: { value: number }) { return <div className="flex items-center gap-2"><div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-teal-500" style={{width:`${value*100}%`}} /></div><span className="text-xs font-semibold text-slate-600">{Math.round(value*100)}%</span></div> }
function Card({children, className='' }: {children: React.ReactNode; className?:string}) { return <section className={`card ${className}`}>{children}</section> }
function ChartTitle({title, detail}: {title:string;detail:string}) { return <div className="mb-5 flex items-start justify-between"><div><h3 className="font-semibold text-slate-900">{title}</h3><p className="mt-1 text-xs text-slate-500">{detail}</p></div><button className="icon-button"><MoreHorizontal size={18}/></button></div> }
function App() {
 const [page,setPage]=useState<Page>('Dashboard'), [open,setOpen]=useState(false), [toast,setToast]=useState('');
 const showToast=(s:string)=>{setToast(s);setTimeout(()=>setToast(''),3000)};
 return <div className="min-h-screen bg-[#f7f9fc] text-slate-800"><aside className={`sidebar ${open?'mobile-open':''}`}><div className="flex items-center gap-3 px-5 py-6"><div className="grid h-10 w-10 place-items-center rounded-xl bg-teal-400 text-navy shadow-lg shadow-teal-200"><BellRing size={21}/></div><div><p className="text-sm font-bold tracking-tight text-white">Router<span className="text-teal-300">AI</span></p><p className="text-[10px] uppercase tracking-widest text-slate-400">Notification OS</p></div></div><nav className="mt-5 px-3">{nav.map(({name,icon:Icon})=><button key={name} onClick={()=>{setPage(name);setOpen(false)}} className={`nav-link ${page===name?'active':''}`}><Icon size={18}/><span>{name}</span>{name==='Message Router'&&<span className="ml-auto h-1.5 w-1.5 rounded-full bg-teal-300"/>}</button>)}</nav><div className="mx-5 mt-auto mb-6 rounded-xl border border-slate-700/70 bg-slate-800 p-4"><div className="flex gap-2"><Sparkles size={16} className="mt-0.5 text-teal-300"/><div><p className="text-xs font-semibold text-white">Intelligence active</p><p className="mt-1 text-[11px] leading-4 text-slate-400">All routing systems are healthy.</p></div></div></div></aside>{open&&<button aria-label="Close menu" onClick={()=>setOpen(false)} className="fixed inset-0 z-30 bg-slate-950/40 lg:hidden"/>}<main className="lg:ml-64"><header className="flex h-[76px] items-center justify-between border-b border-slate-200/80 bg-white px-5 sm:px-8"><div className="flex items-center gap-3"><button className="icon-button lg:hidden" onClick={()=>setOpen(true)}><Menu size={20}/></button><div><p className="text-[11px] font-semibold uppercase tracking-[.18em] text-teal-600">Workspace / {page}</p><h1 className="text-lg font-bold tracking-tight text-slate-900">{page}</h1></div></div><div className="flex items-center gap-4"><div className="hidden items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 sm:flex"><span className="h-2 w-2 rounded-full bg-emerald-500"/>System operational</div><div className="grid h-9 w-9 place-items-center rounded-full bg-gradient-to-br from-indigo-500 to-teal-400 text-xs font-bold text-white">OM</div></div></header><div className="mx-auto max-w-[1550px] p-5 sm:p-8">{page==='Dashboard'&&<Dashboard setPage={setPage}/>} {page==='Message Router'&&<Router showToast={showToast}/>} {page==='Gmail Intelligence'&&<Gmail showToast={showToast}/>} {page==='Message History'&&<History/>} {page==='Analytics'&&<Analytics/>} {page==='Rules & Intelligence'&&<Intelligence/>} {page==='System'&&<System/>}</div></main>{toast&&<div className="toast"><CheckCircle2 size={18}/>{toast}</div>}</div>
}
function Dashboard({setPage}:{setPage:(p:Page)=>void}) {
  const [mode, setMode] = useState<'gmail' | 'dataset'>('gmail');

  // Dataset states
  const [stats, setStats] = useState<DashboardStats>();
  const [data, setData] = useState<any>();
  const [messages, setMessages] = useState<Message[]>([]);

  // Gmail states
  const [gmailStatus, setGmailStatus] = useState<GmailStatus>();
  const [gmailStats, setGmailStats] = useState<GmailDashboard>();
  const [gmailResults, setGmailResults] = useState<GmailResult[]>([]);

  const [limit, setLimit] = useState(25);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const loadGmail = async () => {
    try {
      const status = await getGmailStatus();
      setGmailStatus(status);
      if (status.connected) {
        const dashboard = await getGmailDashboard();
        setGmailStats(dashboard);
        if (dashboard.analyzed) {
          const results = await getGmailResults();
          setGmailResults(results.results);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const loadDataset = async () => {
    try {
      const s = await getDashboardStats();
      setStats(s);
      const d = await getAnalytics();
      setData(d);
      const m = await getMessages();
      setMessages(m);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadGmail();
    loadDataset();
  }, []);

  const runGmailAnalysis = async () => {
    setBusy(true);
    setError('');
    try {
      const res = await analyzeGmail(limit);
      setGmailResults(res.results);
      const dashboard = await getGmailDashboard();
      setGmailStats(dashboard);
    } catch (e) {
      setError('Unable to analyze Gmail. Check the connection and try again.');
    } finally {
      setBusy(false);
    }
  };

  const gmailAnalytics = useMemo(() => {
    if (!gmailResults || gmailResults.length === 0) return null;
    const counts = { NOTIFY: 0, DIGEST: 0, MUTE: 0 };
    const colors = { NOTIFY: "#2dd4bf", DIGEST: "#818cf8", MUTE: "#fb7185" };
    const trendMap: Record<string, { day: string; notify: number; digest: number; mute: number }> = {};
    
    gmailResults.forEach(r => {
      const act = r.action.toUpperCase() as 'NOTIFY' | 'DIGEST' | 'MUTE';
      if (act in counts) counts[act]++;
      const day = r.timestamp ? r.timestamp.slice(0, 10) : new Date().toISOString().slice(0, 10);
      if (!trendMap[day]) trendMap[day] = { day, notify: 0, digest: 0, mute: 0 };
      const actLower = r.action.toLowerCase() as 'notify' | 'digest' | 'mute';
      if (actLower in trendMap[day]) trendMap[day][actLower]++;
    });

    const routing = Object.keys(counts).map(key => ({
      name: key.charAt(0) + key.slice(1).toLowerCase(),
      value: counts[key as 'NOTIFY' | 'DIGEST' | 'MUTE'],
      color: colors[key as 'NOTIFY' | 'DIGEST' | 'MUTE']
    }));

    const trend = Object.values(trendMap).sort((a, b) => a.day.localeCompare(b.day));
    return { routing, trend };
  }, [gmailResults]);

  const datasetValues = stats ? [
    ['Total processed', stats.total.toLocaleString(), MessageSquareText, 'Dataset'],
    ['Notify', stats.notify.toLocaleString(), BellRing, 'Actual output'],
    ['Digest', stats.digest.toLocaleString(), Clock3, 'Actual output'],
    ['Mute', stats.mute.toLocaleString(), ShieldCheck, 'Actual output']
  ] : [];

  const gmailValues = gmailStats && gmailStats.connected ? [
    ['Total processed', gmailStats.total.toLocaleString(), MessageSquareText, 'Gmail Inbox'],
    ['Notify', gmailStats.notify.toLocaleString(), BellRing, 'Routed action'],
    ['Digest', gmailStats.digest.toLocaleString(), Clock3, 'Routed action'],
    ['Mute', gmailStats.mute.toLocaleString(), ShieldCheck, 'Routed action']
  ] : [];

  const gmailMessages: Message[] = useMemo(() => {
    return gmailResults.map(r => ({
      id: r.id,
      sender: r.sender,
      conversation: r.subject,
      text: r.preview || r.message,
      type: 'text' as const,
      action: r.action,
      confidence: r.confidence,
      timestamp: r.timestamp,
      source: 'gmail' as const
    }));
  }, [gmailResults]);

  const values = mode === 'gmail' ? gmailValues : datasetValues;
  const currentData = mode === 'gmail' ? gmailAnalytics : data;
  const currentMessages = mode === 'gmail' ? gmailMessages : messages;

  if (mode === 'gmail' && gmailStatus && !gmailStatus.connected) {
    return <>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">Good morning, Omar</h2>
          <p className="mt-1 text-sm text-slate-500">Here’s how your notification intelligence is performing.</p>
          <div className="mt-4 flex gap-2">
            <button onClick={() => setMode('gmail')} className="filter selected border-teal-200 bg-teal-50 text-teal-700">📊 Live Gmail</button>
            <button onClick={() => setMode('dataset')} className="filter">🧪 Demo Dataset</button>
          </div>
        </div>
      </div>
      
      <Card className="flex flex-col items-center justify-center p-12 text-center min-h-[400px]">
        <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-teal-50 text-teal-600">
          <Mail size={32} />
        </div>
        <h3 className="mt-6 text-lg font-semibold text-slate-900">Connect Your Gmail Account</h3>
        <p className="mt-2 max-w-md text-sm text-slate-500">
          RouterAI can parse and route real emails using Google's secure OAuth flow. Connect your Gmail account to view your live inbox routing stats.
        </p>
        <button onClick={connectGmail} className="primary mt-6">
          <Mail size={16} /> Connect Gmail
        </button>
      </Card>
    </>;
  }

  if (mode === 'gmail' && gmailStatus?.connected && (!gmailStats?.analyzed || gmailResults.length === 0)) {
    return <>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">Good morning, Omar</h2>
          <p className="mt-1 text-sm text-slate-500">Here’s how your notification intelligence is performing.</p>
          <div className="mt-4 flex gap-2">
            <button onClick={() => setMode('gmail')} className="filter selected border-teal-200 bg-teal-50 text-teal-700">📊 Live Gmail</button>
            <button onClick={() => setMode('dataset')} className="filter">🧪 Demo Dataset</button>
          </div>
        </div>
      </div>
      
      <Card className="flex flex-col items-center justify-center p-12 text-center min-h-[400px]">
        <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-teal-50 text-teal-600">
          <Sparkles size={32} />
        </div>
        <h3 className="mt-6 text-lg font-semibold text-slate-900">Gmail Connected</h3>
        <p className="mt-1 text-xs text-emerald-600 font-medium">Connected as {gmailStatus.email}</p>
        <p className="mt-3 max-w-md text-sm text-slate-500">
          Your Gmail account is connected. Analyze your recent messages to classify them as Notify, Digest, or Mute.
        </p>
        
        <div className="mt-6 flex flex-col items-center gap-3 sm:flex-row">
          <label className="mt-0 flex items-center gap-2 text-sm text-slate-600">
            Emails to fetch:
            <select value={limit} onChange={e => setLimit(Number(e.target.value))} className="mt-0 p-1.5 border border-slate-200 rounded-lg bg-white">
              <option value={10}>Last 10</option>
              <option value={25}>Last 25</option>
              <option value={50}>Last 50</option>
            </select>
          </label>
          <button onClick={runGmailAnalysis} disabled={busy} className="primary">
            {busy ? <span className="loader small" /> : <Sparkles size={16} />}
            {busy ? 'Running intelligence routing...' : 'Analyze Gmail Emails'}
          </button>
        </div>
        {error && <p className="mt-4 text-sm text-rose-600">{error}</p>}
      </Card>
    </>;
  }

  return <>
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-slate-900">Good morning, Omar</h2>
        <p className="mt-1 text-sm text-slate-500">Here’s how your notification intelligence is performing.</p>
        <div className="mt-4 flex gap-2">
          <button 
            onClick={() => setMode('gmail')} 
            className={`filter ${mode === 'gmail' ? 'selected border-teal-200 bg-teal-50 text-teal-700' : ''}`}
          >
            📊 Live Gmail
          </button>
          <button 
            onClick={() => setMode('dataset')} 
            className={`filter ${mode === 'dataset' ? 'selected border-teal-200 bg-teal-50 text-teal-700' : ''}`}
          >
            🧪 Demo Dataset
          </button>
        </div>
      </div>
      <div className="flex gap-2">
        {mode === 'gmail' && gmailStatus?.connected && (
          <button onClick={() => setPage('Gmail Intelligence')} className="filter">
            <Mail size={16} className="mr-1 inline" /> Manage Gmail
          </button>
        )}
        <button onClick={() => setPage('Message Router')} className="primary">
          <Sparkles size={17} />Analyze a message<ArrowRight size={16} />
        </button>
      </div>
    </div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {values.map(([label,value,Icon,sub]:any)=>(
        <Card key={label} className="stat">
          <div className="flex justify-between">
            <span className="stat-icon"><Icon size={19}/></span>
            <span className="text-xs font-medium text-emerald-600">{sub}</span>
          </div>
          <p className="mt-5 text-2xl font-bold tracking-tight text-slate-900">{value}</p>
          <p className="mt-1 text-xs text-slate-500">{label} · loaded from {mode === 'gmail' ? 'Gmail API' : 'output.csv'}</p>
        </Card>
      ))}
    </div>

    <div className="mt-5 grid gap-5 xl:grid-cols-3">
      <Card className="xl:col-span-2">
        <ChartTitle 
          title={mode === 'gmail' ? "Gmail routing activity" : "Routing activity"} 
          detail={mode === 'gmail' ? "Live emails routed across the inbox timeline" : "Messages routed across the dataset timeline"}
        />
        {currentData ? (
          <div className="h-64">
            <ResponsiveContainer>
              <AreaChart data={currentData.trend}>
                <defs>
                  <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                    <stop stopColor="#2dd4bf" stopOpacity=".3"/>
                    <stop offset="1" stopColor="#2dd4bf" stopOpacity="0"/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{fontSize:12,fill:'#94a3b8'}}/>
                <YAxis hide/>
                <Tooltip/>
                <Area type="monotone" dataKey="notify" stroke="#14b8a6" fill="url(#g)" strokeWidth={3}/>
                <Area type="monotone" dataKey="digest" stroke="#818cf8" fill="none" strokeWidth={2}/>
                <Area type="monotone" dataKey="mute" stroke="#fb7185" fill="none" strokeWidth={1.5}/>
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : <Loading/>}
      </Card>
      
      <Card>
        <ChartTitle 
          title={mode === 'gmail' ? "Gmail routing distribution" : "Routing distribution"} 
          detail={mode === 'gmail' ? "Live routing action mix" : "Actual output.csv decision mix"}
        />
        {currentData ? (
          <div className="relative h-64">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={currentData.routing} dataKey="value" innerRadius={62} outerRadius={86} paddingAngle={4}>
                  {currentData.routing.map((x:any)=><Cell key={x.name} fill={x.color}/>)}
                </Pie>
                <Tooltip/>
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
              <b className="text-xl">{mode === 'gmail' ? gmailStats?.total : stats?.total}</b>
              <span className="-mt-5 text-[10px] text-slate-400">{mode === 'gmail' ? 'emails' : 'messages'}</span>
            </div>
          </div>
        ) : <Loading/>}
        <div className="flex justify-center gap-4">
          {currentData?.routing.map((x:any)=><span key={x.name} className="legend"><i style={{background:x.color}}/>{x.name}</span>)}
        </div>
      </Card>
    </div>

    <div className="mt-5 grid gap-5 xl:grid-cols-3">
      <Card className="xl:col-span-2">
        <ChartTitle 
          title={mode === 'gmail' ? "Recent Gmail emails" : "Recent messages"} 
          detail={mode === 'gmail' ? "Latest routing decisions for your inbox" : "Latest decisions made by the routing engine"}
        />
        {currentMessages && currentMessages.length > 0 ? (
          <MessageTable messages={currentMessages.slice(0,4)}/>
        ) : (
          <div className="py-12 text-center text-sm text-slate-500">No messages analyzed.</div>
        )}
        <button 
          onClick={() => setPage(mode === 'gmail' ? 'Gmail Intelligence' : 'Message History')} 
          className="mt-3 text-sm font-semibold text-teal-600"
        >
          View all {mode === 'gmail' ? 'Gmail' : 'history'} <ChevronRight className="inline" size={15}/>
        </button>
      </Card>
      
      <Card>
        <ChartTitle title="System health" detail="Live operational telemetry"/>
        <div className="space-y-5">
          <Metric 
            label="Average confidence" 
            value={`${Math.round(((mode === 'gmail' ? gmailStats?.confidence : stats?.confidence) || 0)*100)}%`} 
            detail="Signal agreement"
          />
          <Metric 
            label={mode === 'gmail' ? "Connected account" : "Processing time"} 
            value={mode === 'gmail' ? (gmailStats?.email || 'N/A') : (stats?.processingTime == null ? '—' : `${stats.processingTime}s`)} 
            detail={mode === 'gmail' ? "Gmail OAuth status" : "Not stored in output.csv"}
          />
          <Metric 
            label="Pipeline status" 
            value="Healthy" 
            detail={mode === 'gmail' ? "FastAPI live adapter" : "6 / 6 services online"} 
            green
          />
        </div>
      </Card>
    </div>
  </>;
}
function Metric({label,value,detail,green}:{label:string;value:string;detail:string;green?:boolean}){return <div className="border-b border-slate-100 pb-4 last:border-0"><p className="text-xs text-slate-500">{label}</p><div className="mt-1 flex items-baseline justify-between"><b className={green?'text-emerald-600':'text-slate-900'}>{value}</b><span className="text-[11px] text-slate-400">{detail}</span></div></div>}
function Loading(){return <div className="grid h-56 place-items-center"><div className="loader"/></div>}
function MessageTable({messages}:{messages:Message[]}){return <div className="overflow-x-auto"><table><thead><tr><th>Sender</th><th>Message</th><th>Source</th><th>Action</th><th>Confidence</th><th>Time</th></tr></thead><tbody>{messages.map(m=><tr key={m.id}><td><b className="block text-sm">{m.sender}</b><span className="text-xs text-slate-400">{m.conversation}</span></td><td className="max-w-[280px] truncate text-sm text-slate-600">{m.text}</td><td><span className="text-xs text-slate-500">{m.source==='gmail'?'Gmail':'Dataset'}</span></td><td><Badge action={m.action}/></td><td><Confidence value={m.confidence}/></td><td className="whitespace-nowrap text-xs text-slate-400">{m.timestamp}</td></tr>)}</tbody></table></div>}
function FeedbackCell({result}:{result:GmailResult}) {
  const [state,setState]=useState<'idle'|'wrong'|'done'>('idle');
  const [saving,setSaving]=useState(false);
  const submit=async(correct:string)=>{
    setSaving(true);
    try{
      await submitFeedback({message_id:result.id,original_action:result.action,correct_action:correct,subject:result.subject,sender:result.sender});
      setState('done');
    }finally{setSaving(false);}
  };
  if(state==='done') return <span className="text-xs font-medium text-emerald-600">✓ Saved</span>;
  if(state==='wrong') return <div className="flex flex-wrap gap-1">{(['NOTIFY','DIGEST','MUTE'] as const).filter(a=>a!==result.action.toUpperCase()).map(a=><button key={a} disabled={saving} onClick={()=>submit(a)} className={`filter text-[11px] px-2 py-0.5 ${a==='NOTIFY'?'!text-teal-700 !border-teal-200':a==='DIGEST'?'!text-indigo-700 !border-indigo-200':'!text-rose-700 !border-rose-200'}`}>{a}</button>)}<button onClick={()=>setState('idle')} className="filter text-[11px] px-2 py-0.5">✕</button></div>;
  return <div className="flex gap-1"><button onClick={()=>submit(result.action)} title="Correct" className="filter text-[11px] px-2 py-0.5 !text-emerald-700 !border-emerald-200">✓</button><button onClick={()=>setState('wrong')} title="Wrong" className="filter text-[11px] px-2 py-0.5 !text-rose-700 !border-rose-200">✗</button></div>;
}
function Router({showToast}:{showToast:(s:string)=>void}) { const [text,setText]=useState('Can you review the launch deck before our call tomorrow?'),[sender,setSender]=useState('Maya Chen'),[type,setType]=useState<MessageType>('text'),[result,setResult]=useState<AnalysisResult>(),[busy,setBusy]=useState(false); const analyze=async()=>{if(!text.trim())return showToast('Enter a message to analyze');setBusy(true);setResult(undefined);try{setResult(await analyzeMessage({text,sender,type}));showToast('Message analysis complete')}finally{setBusy(false)}};return <><div className="mb-7"><h2 className="text-2xl font-bold tracking-tight text-slate-900">Test the routing engine</h2><p className="mt-1 text-sm text-slate-500">Simulate the full intelligence pipeline before connecting your live API.</p></div><div className="grid gap-5 xl:grid-cols-5"><Card className="xl:col-span-2"><h3 className="font-semibold text-slate-900">Message details</h3><p className="mt-1 text-xs text-slate-500">Provide the context used by the Python router.</p><label>Message content<textarea value={text} onChange={e=>setText(e.target.value)} rows={6} placeholder="Type a WhatsApp message..."/></label><div className="grid grid-cols-2 gap-3"><label>Sender<input value={sender} onChange={e=>setSender(e.target.value)} placeholder="Name or business"/></label><label>Conversation<input defaultValue="Maya Chen" placeholder="Group or direct chat"/></label></div><div className="grid grid-cols-2 gap-3"><label>Message type<select value={type} onChange={e=>setType(e.target.value as MessageType)}><option value="text">Text</option><option value="image">Image</option><option value="voice">Voice note</option><option value="document">Document</option></select></label><label>Timestamp<input type="datetime-local" defaultValue="2026-08-09T10:30"/></label></div><button onClick={analyze} disabled={busy} className="primary mt-5 w-full justify-center">{busy?<span className="loader small"/>:<Sparkles size={17}/>} {busy?'Analyzing intelligence signals…':'Analyze message'}</button></Card><div className="xl:col-span-3">{busy?<Card><Loading/></Card>:result?<Analysis result={result}/>:<Card className="grid min-h-[440px] place-items-center text-center"><div className="max-w-sm"><div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-teal-50 text-teal-600"><BrainCircuit size={26}/></div><h3 className="mt-5 font-semibold text-slate-900">Awaiting a message</h3><p className="mt-2 text-sm leading-6 text-slate-500">Enter a message and the router will surface its action, confidence, evidence, and rule signals.</p></div></Card>}</div></div></>}
function Analysis({result}:{result:AnalysisResult}){return <div className="space-y-5"><Card className="overflow-hidden"><div className={`result-top ${actionClass[result.action]}`}><div><p className="text-xs font-semibold uppercase tracking-widest opacity-70">Final routing action</p><h2 className="mt-2 text-3xl font-bold">{result.action}</h2><p className="mt-2 text-sm opacity-80">{result.reasoning}</p></div><div className="rounded-xl bg-white/20 px-5 py-3 text-center backdrop-blur"><b className="text-2xl">{Math.round(result.confidence*100)}%</b><span className="block text-[10px] uppercase tracking-wider">confidence</span></div></div><div className="grid grid-cols-3 divide-x divide-slate-100 p-4 text-center"><div><p className="text-[11px] text-slate-400">Priority</p><b className="text-sm">{result.priority}</b></div><div><p className="text-[11px] text-slate-400">Message type</p><b className="text-sm capitalize">{result.message_type}</b></div><div><p className="text-[11px] text-slate-400">Processing</p><b className="text-sm">{result.processing_time}s</b></div></div></Card><div className="grid gap-5 lg:grid-cols-2"><Card><h3 className="font-semibold">Retrieved evidence</h3><div className="mt-4 space-y-3">{result.evidence.map(x=><div key={x.id} className="rounded-lg bg-slate-50 p-3"><span className="text-[10px] font-bold uppercase tracking-wider text-teal-600">{x.signal}</span><p className="mt-1 text-xs leading-5 text-slate-600">{x.text}</p></div>)}</div></Card><Card><h3 className="font-semibold">Rules triggered</h3><div className="mt-4 flex flex-wrap gap-2">{result.rules_triggered.map(x=><span key={x} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">{x}</span>)}</div><div className="mt-6 rounded-lg border border-teal-100 bg-teal-50 p-3 text-xs leading-5 text-teal-800"><CheckCircle2 className="mr-1 inline" size={14}/>Result is structured to map directly to the Python router response.</div></Card></div></div>}
function Gmail({showToast}:{showToast:(s:string)=>void}) {const [status,setStatus]=useState<GmailStatus>(),[limit,setLimit]=useState(25),[results,setResults]=useState<GmailResult[]>([]),[busy,setBusy]=useState(false),[error,setError]=useState('');const load=()=>getGmailStatus().then(setStatus).catch(()=>setStatus({connected:false}));useEffect(()=>{load();getGmailResults().then(x=>setResults(x.results)).catch(()=>{})},[]);const run=async()=>{setBusy(true);setError('');try{const x=await analyzeGmail(limit);setResults(x.results);showToast(`${x.total} Gmail messages analyzed`)}catch{setError('Unable to analyze Gmail. Check the connection and try again.')}finally{setBusy(false)}};if(!status)return <Loading/>;return <><div className="mb-7"><h2 className="text-2xl font-bold tracking-tight">Gmail Intelligence</h2><p className="mt-1 text-sm text-slate-500">Route real inbox messages without changing your Gmail account.</p></div><Card><div className="flex flex-wrap items-center justify-between gap-4"><div className="flex items-center gap-3"><span className={`stat-icon ${status.connected?'':'!bg-slate-100 !text-slate-400'}`}><Mail size={19}/></span><div><h3 className="font-semibold">{status.connected?'Gmail connected':'Connect Gmail'}</h3><p className="text-xs text-slate-500">{status.connected?status.email:'Authorize read-only Gmail access to analyze your inbox.'}</p></div></div>{status.connected?<div className="flex gap-2"><button onClick={load} className="filter">Refresh status</button><button onClick={async()=>{const x=await disconnectGmail();setStatus({connected:false});setResults([]);showToast(x.message)}} className="filter">Disconnect</button></div>:<button onClick={connectGmail} className="primary"><Mail size={16}/>Connect Gmail</button>}</div><p className="mt-5 rounded-lg border border-teal-100 bg-teal-50 p-3 text-xs leading-5 text-teal-800">RouterAI requests read-only Gmail access to analyze your email messages and classify them as Notify, Digest, or Mute. RouterAI does not send, delete, modify, or move your emails.</p></Card>{status.connected&&<><Card className="mt-5"><div className="flex flex-wrap items-end justify-between gap-4"><div><h3 className="font-semibold">Analyze inbox</h3><p className="mt-1 text-xs text-slate-500">Fetches only the selected recent messages through Gmail’s read-only API.</p></div><div className="flex items-end gap-3"><label className="mt-0">Emails<select value={limit} onChange={e=>setLimit(Number(e.target.value))}><option value={10}>Last 10</option><option value={25}>Last 25</option><option value={50}>Last 50</option></select></label><button onClick={run} disabled={busy} className="primary">{busy?<span className="loader small"/>:<Sparkles size={16}/>}Analyze emails</button></div></div>{error&&<p className="mt-4 text-sm text-rose-600">{error}</p>}</Card><Card className="mt-5"><ChartTitle title="Gmail routing results" detail={results.length?`${results.length} real Gmail messages analyzed`:'No Gmail messages analyzed yet'}/>{results.length?<div className="overflow-x-auto"><table><thead><tr><th>Sender</th><th>Subject & preview</th><th>Decision</th><th>Confidence</th><th>Date</th><th>Feedback</th></tr></thead><tbody>{results.map(x=><tr key={x.id}><td className="max-w-40 truncate text-sm font-semibold">{x.sender}</td><td className="max-w-[360px]"><b className="block truncate text-sm">{x.subject}</b><span className="block truncate text-xs text-slate-500">{x.preview}</span></td><td><Badge action={x.action}/></td><td><Confidence value={x.confidence}/></td><td className="whitespace-nowrap text-xs text-slate-400">{x.timestamp}</td><td><FeedbackCell result={x}/></td></tr>)}</tbody></table></div>:<div className="py-12 text-center text-sm text-slate-500">Analyze your inbox to see real Gmail decisions and analytics.</div>}</Card></>}</>}
function History(){const [items,setItems]=useState<Message[]>([]),[query,setQuery]=useState(''),[filter,setFilter]=useState<'ALL'|Action>('ALL'),[source,setSource]=useState<'all'|'dataset'|'gmail'>('all');useEffect(()=>{getMessages().then(dataset=>getGmailResults().then(g=>setItems([...dataset,...g.results.map(x=>({id:x.id,sender:x.sender,conversation:x.subject,text:x.message,type:'text' as MessageType,action:x.action,confidence:x.confidence,timestamp:x.timestamp,source:'gmail' as const}))])).catch(()=>setItems(dataset)))},[]);const shown=useMemo(()=>items.filter(x=>(filter==='ALL'||x.action===filter)&&(source==='all'||x.source===source)&&`${x.sender} ${x.text}`.toLowerCase().includes(query.toLowerCase())),[items,query,filter,source]);return <><div className="mb-7"><h2 className="text-2xl font-bold tracking-tight">Message history</h2><p className="mt-1 text-sm text-slate-500">Dataset and analyzed Gmail routing decisions.</p></div><Card><div className="mb-5 flex flex-col justify-between gap-3 md:flex-row"><div className="search"><Search size={16}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search senders or messages..."/></div><div className="flex gap-2 overflow-x-auto">{(['all','dataset','gmail'] as const).map(x=><button key={x} onClick={()=>setSource(x)} className={`filter ${source===x?'selected':''}`}>{x[0].toUpperCase()+x.slice(1)}</button>)}{(['ALL','NOTIFY','DIGEST','MUTE'] as const).map(x=><button key={x} onClick={()=>setFilter(x)} className={`filter ${filter===x?'selected':''}`}>{x==='ALL'?<Filter size={14}/>:null}{x==='ALL'?'All':x}</button>)}</div></div>{shown.length?<MessageTable messages={shown}/>:<div className="py-16 text-center text-sm text-slate-500">No messages match these filters.</div>}<div className="mt-5 flex justify-between text-xs text-slate-500"><span>Showing {shown.length} messages</span><span>Page 1 of 1</span></div></Card></>}
function Analytics(){const [data,setData]=useState<any>();useEffect(()=>{getAnalytics().then(setData)},[]);if(!data)return <Loading/>;return <><div className="mb-7"><h2 className="text-2xl font-bold tracking-tight">Analytics</h2><p className="mt-1 text-sm text-slate-500">Understand routing quality and notification volume.</p></div><div className="grid gap-5 lg:grid-cols-2"><Card><ChartTitle title="Routing trends" detail="Daily decisions by action"/><div className="h-72"><ResponsiveContainer><LineChart data={data.trend}><XAxis dataKey="day" axisLine={false} tickLine={false}/><YAxis axisLine={false} tickLine={false}/><Tooltip/><Line dataKey="notify" stroke="#14b8a6" strokeWidth={3}/><Line dataKey="digest" stroke="#818cf8" strokeWidth={3}/><Line dataKey="mute" stroke="#fb7185" strokeWidth={3}/></LineChart></ResponsiveContainer></div></Card><Card><ChartTitle title="Message type distribution" detail="How inbound content is classified"/><div className="h-72"><ResponsiveContainer><BarChart data={[{name:'Text',value:72},{name:'Image',value:13},{name:'Voice',value:9},{name:'Document',value:6}]}><XAxis dataKey="name" axisLine={false} tickLine={false}/><YAxis hide/><Tooltip/><Bar dataKey="value" fill="#2dd4bf" radius={[6,6,0,0]}/></BarChart></ResponsiveContainer></div></Card><Card><ChartTitle title="Confidence distribution" detail="Decision confidence across recent messages"/><div className="grid grid-cols-4 gap-3 pt-6">{[['90–100%',57],['75–89%',28],['50–74%',11],['< 50%',4]].map(([x,y])=><div key={String(x)} className="rounded-xl bg-slate-50 p-4"><b className="text-xl">{y}%</b><p className="mt-2 text-xs text-slate-500">{x}</p></div>)}</div></Card><Card><ChartTitle title="Processing performance" detail="Pipeline stage latency"/><div className="space-y-5 pt-2">{[['Context & features','8 ms',22],['Retrieval','12 ms',35],['Rules & reasoning','15 ms',44],['Confidence','3 ms',10]].map(([x,y,w])=><div key={String(x)}><div className="flex justify-between text-xs"><span>{x}</span><b>{y}</b></div><div className="mt-2 h-2 rounded-full bg-slate-100"><div className="h-full rounded-full bg-indigo-400" style={{width:`${w}%`}}/></div></div>)}</div></Card></div></>}
function Intelligence(){const stages: [string,LucideIcon,string][]=[['Message',MessageSquareText,'Incoming content & media'],['Context',Database,'Relationships and history'],['Features',Gauge,'100+ routing signals'],['Retrieval',Search,'Relevant past evidence'],['Rules + AI',BrainCircuit,'Deterministic and LLM reasoning'],['Confidence',ShieldCheck,'Signal agreement'],['Final action',BellRing,'Notify, digest, or mute']];const cards: [string,string,LucideIcon][]=[['Context Builder','Builds a connected view of the user, sender, group, history, and quiet-hours preferences.',Database],['Feature Engineering','Extracts engagement, trust, urgency, media, relationship, and behavioral signals.',Gauge],['Evidence Retriever','Scores historical messages for relevance, outcome, similarity, and recency.',Search],['Deterministic Rule Engine','Applies explicit protections for scams, opt-outs, business trust, and urgent signals.',ShieldCheck],['LLM Reasoning Engine','Handles nuanced fallthrough decisions when rules alone are not decisive.',BrainCircuit],['Confidence Calibration','Measures agreement between evidence, context, rules, and reasoning.',Sparkles]];return <><div className="mb-7"><h2 className="text-2xl font-bold tracking-tight">Rules & intelligence</h2><p className="mt-1 text-sm text-slate-500">A transparent view of the decision pipeline that powers every route.</p></div><Card><h3 className="font-semibold">Intelligence pipeline</h3><p className="mt-1 text-xs text-slate-500">Each message travels through a layered, explainable decision system.</p><div className="pipeline">{stages.map(([name,I,detail],i)=><div className="flex items-center" key={name}><div className="pipeline-step"><span><I size={18}/></span><b>{name}</b><small>{detail}</small></div>{i<stages.length-1&&<ArrowRight className="pipeline-arrow" size={18}/>}</div>)}</div></Card><div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">{cards.map(([title,desc,I])=><Card key={title} className="intelligence"><span className="stat-icon"><I size={18}/></span><h3>{title}</h3><p>{desc}</p></Card>)}</div><Card className="mt-5"><ChartTitle title="Example routing rules" detail="Representative deterministic decision logic"/><div className="grid gap-3 md:grid-cols-3"><div className="rule"><Badge action="MUTE"/><p>Suspicious URL, pressure language, or domain mismatch.</p></div><div className="rule"><Badge action="NOTIFY"/><p>Verified, time-sensitive update from a trusted contact.</p></div><div className="rule"><Badge action="DIGEST"/><p>Useful non-urgent group or subscribed business update.</p></div></div></Card></>}
function System(){const [status,setStatus]=useState<any>();useEffect(()=>{getSystemStatus().then(setStatus)},[]);return <><div className="mb-7"><h2 className="text-2xl font-bold tracking-tight">System</h2><p className="mt-1 text-sm text-slate-500">Architecture, connection readiness, and current health.</p></div><div className="grid gap-5 lg:grid-cols-3"><Card className="lg:col-span-2"><ChartTitle title="System architecture" detail="Frontend presentation layer prepared for the existing Python router"/><div className="architecture"><div>React + Vite UI</div><ArrowRight/><div>API service layer</div><ArrowRight/><div className="highlight">Future FastAPI adapter</div><ArrowRight/><div>Python router pipeline</div></div><p className="mt-5 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">The frontend only uses mock service functions today. Replacing those calls in <code>src/services/api.ts</code> will connect this interface to a FastAPI endpoint without changing any UI components.</p></Card><Card><ChartTitle title="Current status" detail="Environment telemetry"/>{status?<div className="space-y-4"><Metric label="Router status" value={status.status} detail="All checks passing" green/><Metric label="Frontend version" value={status.version} detail="UI release"/><Metric label="Connection" value="Mock mode" detail={status.backend}/><Metric label="Average latency" value={status.latency} detail="Simulated response"/></div>:<Loading/>}</Card></div><div className="mt-5 grid gap-5 md:grid-cols-2"><Card><h3 className="font-semibold">Technology</h3><div className="mt-4 flex flex-wrap gap-2">{['React','Vite','TypeScript','Tailwind CSS','Lucide','Recharts','Python pipeline','Gemini fallback'].map(x=><span className="chip" key={x}>{x}</span>)}</div></Card><Card><h3 className="font-semibold">Integration contract</h3><p className="mt-3 text-sm leading-6 text-slate-600">The analysis screen expects <code>action</code>, <code>confidence</code>, <code>message_type</code>, <code>reasoning</code>, evidence, rules, and processing time—the same core concepts used by the existing router.</p></Card></div></>}
createRoot(document.getElementById('root')!).render(<App/>);
