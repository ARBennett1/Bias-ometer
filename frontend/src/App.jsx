import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Radio, Youtube, Upload, Mic, Users, Clock, BarChart2,
  ChevronRight, Plus, Link2, Search, RefreshCw, CheckCircle2,
  XCircle, Loader2, AlertCircle, Settings2, List, ArrowLeft,
  TrendingUp, TrendingDown, Minus, Edit3, Eye, Activity
} from 'lucide-react'

const API = '/api'

// ─── Colour helpers ───────────────────────────────────────────────────────────
const SPEAKER_PALETTE = [
  '#e8a020', '#4d7efa', '#34d37a', '#f25757',
  '#a78bfa', '#38bdf8', '#fb923c', '#e879f9',
  '#4ade80', '#f87171',
]

function speakerColor(id) {
  const n = parseInt(id.replace(/\D/g, '') || '0', 10)
  return SPEAKER_PALETTE[n % SPEAKER_PALETTE.length]
}

function sentimentColor(score) {
  if (score === null || score === undefined) return '#556080'
  if (score > 0.1) return '#34d37a'
  if (score < -0.1) return '#f25757'
  return '#8892b0'
}

function sentimentIcon(score) {
  if (score === null || score === undefined) return <Minus size={12} />
  if (score > 0.1) return <TrendingUp size={12} />
  if (score < -0.1) return <TrendingDown size={12} />
  return <Minus size={12} />
}

function fmtTime(s) {
  if (s == null) return '—'
  const m = Math.floor(s / 60)
  const sec = Math.round(s % 60)
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`
}

function fmtTimestamp(s) {
  if (s == null) return '00:00'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = Math.floor(s % 60)
  const mm = String(m).padStart(2, '0')
  const ss = String(sec).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

function fmtDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #080c14;
    --surf:    #0e1422;
    --surf2:   #151d2e;
    --surf3:   #1c2540;
    --bord:    #1e2840;
    --bord2:   #2a3555;
    --accent:  #e8a020;
    --accentd: #b87a10;
    --blue:    #4d7efa;
    --text:    #c8d4e8;
    --dim:     #556080;
    --dimmer:  #2e3a55;
    --pos:     #34d37a;
    --neg:     #f25757;
    --warn:    #f0c040;
    --font:    'DM Sans', system-ui, sans-serif;
    --mono:    'JetBrains Mono', 'Fira Mono', monospace;
    --head:    'Syne', system-ui, sans-serif;
    --radius:  8px;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
  }

  /* Layout */
  .shell { display: flex; min-height: 100vh; }
  .sidebar {
    width: 220px; min-width: 220px;
    background: var(--surf);
    border-right: 1px solid var(--bord);
    display: flex; flex-direction: column;
    position: fixed; top: 0; left: 0; bottom: 0;
    z-index: 100;
  }
  .main { margin-left: 220px; flex: 1; display: flex; flex-direction: column; min-height: 100vh; }
  .topbar {
    height: 56px; border-bottom: 1px solid var(--bord);
    display: flex; align-items: center;
    padding: 0 28px; gap: 12px;
    background: var(--surf); position: sticky; top: 0; z-index: 50;
  }
  .content { padding: 28px; flex: 1; }

  /* Sidebar logo */
  .logo {
    height: 56px; border-bottom: 1px solid var(--bord);
    display: flex; align-items: center; padding: 0 20px; gap: 10px;
  }
  .logo-mark {
    width: 28px; height: 28px; border-radius: 6px;
    background: var(--accent); display: flex; align-items: center; justify-content: center;
  }
  .logo-text { font-family: var(--head); font-size: 15px; font-weight: 700; color: var(--text); letter-spacing: -0.3px; }

  /* Nav */
  .nav { padding: 12px 10px; flex: 1; }
  .nav-label { font-size: 10px; font-weight: 500; letter-spacing: 0.08em; color: var(--dimmer); text-transform: uppercase; padding: 8px 10px 4px; }
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 10px; border-radius: var(--radius); cursor: pointer;
    color: var(--dim); font-size: 13.5px; transition: all 0.15s;
    border: none; background: none; width: 100%; text-align: left;
  }
  .nav-item:hover { background: var(--surf2); color: var(--text); }
  .nav-item.active { background: var(--surf3); color: var(--accent); }
  .nav-item.active svg { color: var(--accent); }

  /* Status badge in sidebar */
  .job-badge {
    margin: 0 10px 14px; padding: 8px 10px; border-radius: var(--radius);
    background: var(--surf2); border: 1px solid var(--bord);
    font-size: 12px; color: var(--dim);
  }
  .job-badge-dot {
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
  }

  /* Cards */
  .card {
    background: var(--surf); border: 1px solid var(--bord);
    border-radius: 10px; padding: 18px 20px;
  }
  .card-sm { padding: 12px 16px; }
  .card-title { font-family: var(--head); font-weight: 600; font-size: 13px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }

  /* Stat row */
  .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }
  .stat { background: var(--surf); border: 1px solid var(--bord); border-radius: 10px; padding: 16px 18px; }
  .stat-label { font-size: 11px; font-weight: 500; letter-spacing: 0.07em; text-transform: uppercase; color: var(--dim); margin-bottom: 6px; }
  .stat-val { font-family: var(--head); font-size: 26px; font-weight: 700; color: var(--text); line-height: 1; }
  .stat-sub { font-size: 11px; color: var(--dim); margin-top: 4px; font-family: var(--mono); }

  /* Forms */
  .form-group { margin-bottom: 16px; }
  .form-label { display: block; font-size: 12px; font-weight: 500; color: var(--dim); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
  .form-input {
    width: 100%; background: var(--surf2); border: 1px solid var(--bord2);
    border-radius: var(--radius); padding: 9px 12px; color: var(--text); font-size: 14px;
    font-family: var(--font); outline: none; transition: border 0.15s;
  }
  .form-input:focus { border-color: var(--accent); }
  .form-input::placeholder { color: var(--dimmer); }

  .toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 8px 0; }
  .toggle-label { font-size: 13px; color: var(--text); }
  .toggle-desc { font-size: 11px; color: var(--dim); margin-top: 1px; }
  .toggle {
    width: 40px; height: 22px; border-radius: 11px; border: none; cursor: pointer; position: relative;
    transition: background 0.2s; flex-shrink: 0;
  }
  .toggle.on { background: var(--accent); }
  .toggle.off { background: var(--surf3); }
  .toggle::after {
    content: ''; position: absolute; top: 3px; width: 16px; height: 16px;
    border-radius: 50%; background: var(--text); transition: left 0.2s;
  }
  .toggle.on::after { left: 21px; }
  .toggle.off::after { left: 3px; }

  /* Buttons */
  .btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 9px 16px; border-radius: var(--radius); font-size: 13.5px; font-weight: 500;
    border: none; cursor: pointer; transition: all 0.15s; font-family: var(--font);
    white-space: nowrap;
  }
  .btn-primary { background: var(--accent); color: #080c14; }
  .btn-primary:hover { background: var(--accentd); }
  .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-ghost { background: transparent; color: var(--dim); border: 1px solid var(--bord2); }
  .btn-ghost:hover { background: var(--surf2); color: var(--text); }
  .btn-danger { background: transparent; color: var(--neg); border: 1px solid #3a1a1a; }
  .btn-danger:hover { background: #2a1010; }
  .btn-sm { padding: 6px 11px; font-size: 12px; }
  .btn-icon { padding: 7px; border-radius: var(--radius); }

  /* Tabs */
  .tabs { display: flex; gap: 4px; background: var(--surf2); padding: 4px; border-radius: 10px; margin-bottom: 20px; }
  .tab {
    flex: 1; padding: 8px 12px; border-radius: 7px; font-size: 13px; font-weight: 500;
    color: var(--dim); cursor: pointer; border: none; background: none; transition: all 0.15s;
    font-family: var(--font); text-align: center;
  }
  .tab.active { background: var(--surf); color: var(--text); border: 1px solid var(--bord); }
  .tab:hover:not(.active) { color: var(--text); }

  /* Table */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th { font-size: 11px; font-weight: 500; letter-spacing: 0.07em; text-transform: uppercase; color: var(--dim); padding: 8px 12px; border-bottom: 1px solid var(--bord); text-align: left; }
  td { padding: 10px 12px; border-bottom: 1px solid var(--bord); font-size: 13px; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surf2); }
  .mono { font-family: var(--mono); font-size: 12px; }

  /* Status chips */
  .chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; border-radius: 20px; font-size: 11px; font-weight: 500; }
  .chip-queued  { background: #1e2840; color: var(--dim); }
  .chip-running { background: #1a2040; color: var(--blue); }
  .chip-complete{ background: #0d2a1a; color: var(--pos); }
  .chip-error   { background: #2a1010; color: var(--neg); }

  /* Speaker pill */
  .spk-pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px; border-radius: 20px; font-family: var(--mono); font-size: 11px; font-weight: 500; }

  /* Timeline */
  .timeline { position: relative; padding: 8px 0; }
  .tl-track { height: 28px; border-radius: 4px; background: var(--surf2); position: relative; margin: 2px 0; overflow: hidden; }
  .tl-seg { position: absolute; top: 0; bottom: 0; border-radius: 2px; transition: opacity 0.15s; cursor: pointer; }
  .tl-seg:hover { opacity: 0.8; }
  .tl-label { font-family: var(--mono); font-size: 11px; color: var(--dim); width: 96px; flex-shrink: 0; padding-right: 10px; display: flex; align-items: center; }
  .tl-row { display: flex; align-items: stretch; margin: 3px 0; }

  /* Turns list */
  .turn-item { border-left: 3px solid; padding: 10px 14px; border-radius: 0 var(--radius) var(--radius) 0; margin: 6px 0; background: var(--surf2); }
  .turn-meta { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
  .turn-time { font-family: var(--mono); font-size: 11px; color: var(--dim); }
  .turn-text { font-size: 13px; color: var(--text); line-height: 1.55; }
  .turn-no-text { font-size: 12px; color: var(--dimmer); font-style: italic; }

  /* Drop zone */
  .dropzone {
    border: 2px dashed var(--bord2); border-radius: 10px;
    padding: 32px 24px; text-align: center; transition: all 0.2s; cursor: pointer;
  }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: rgba(232,160,32,0.04); }
  .dropzone-icon { color: var(--dimmer); margin-bottom: 10px; }
  .dropzone-text { font-size: 14px; color: var(--dim); }
  .dropzone-sub { font-size: 12px; color: var(--dimmer); margin-top: 4px; }

  /* Divider */
  .divider { border: none; border-top: 1px solid var(--bord); margin: 18px 0; }

  /* Errors / notices */
  .notice { display: flex; gap: 10px; padding: 12px 14px; border-radius: var(--radius); margin-bottom: 16px; font-size: 13px; }
  .notice-error { background: #1e0a0a; border: 1px solid #3a1010; color: #f08080; }
  .notice-info  { background: #0a1220; border: 1px solid #1e3060; color: #80a8f0; }
  .notice-ok    { background: #0a1e10; border: 1px solid #1a4020; color: #60d090; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--surf3); border-radius: 3px; }

  /* Spinner animation */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spin { animation: spin 1s linear infinite; }

  /* Progress bar */
  .progress-bar-wrap { background: var(--surf3); border-radius: 4px; height: 4px; margin-top: 8px; }
  .progress-bar { height: 4px; border-radius: 4px; background: var(--accent); }
  @keyframes progress-indeterminate {
    0% { left: -40%; width: 40%; }
    100% { left: 100%; width: 40%; }
  }
  .progress-bar-indeterminate {
    position: relative; overflow: hidden;
  }
  .progress-bar-indeterminate::after {
    content: ''; position: absolute; height: 100%;
    background: var(--accent); border-radius: 4px;
    animation: progress-indeterminate 1.5s ease-in-out infinite;
  }

  /* Pulse for live jobs */
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .pulse { animation: pulse 2s ease-in-out infinite; }

  /* Link button */
  .link-btn { background: none; border: none; color: var(--blue); cursor: pointer; font-family: var(--font); font-size: 13px; padding: 0; text-decoration: underline; }
  .link-btn:hover { color: var(--text); }

  /* Modal */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.7);
    display: flex; align-items: center; justify-content: center; z-index: 999;
  }
  .modal {
    background: var(--surf); border: 1px solid var(--bord2); border-radius: 12px;
    padding: 24px; width: 480px; max-width: 95vw;
  }
  .modal-title { font-family: var(--head); font-size: 17px; font-weight: 700; margin-bottom: 16px; }

  .section-title { font-family: var(--head); font-size: 20px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
  .section-sub { font-size: 13px; color: var(--dim); margin-bottom: 22px; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
`

// ─── StatusChip ───────────────────────────────────────────────────────────────
function StatusChip({ status }) {
  const icons = {
    queued: <Clock size={11} />,
    running: <Loader2 size={11} className="spin" />,
    complete: <CheckCircle2 size={11} />,
    error: <XCircle size={11} />,
  }
  return <span className={`chip chip-${status}`}>{icons[status]} {status}</span>
}

// ─── Toggle ───────────────────────────────────────────────────────────────────
function Toggle({ value, onChange }) {
  return (
    <button className={`toggle ${value ? 'on' : 'off'}`} onClick={() => onChange(!value)} />
  )
}

// ─── SpeakerTimeline ─────────────────────────────────────────────────────────
function SpeakerTimeline({ turns, duration, onTurnClick }) {
  if (!turns || !duration) return null
  const speakers = [...new Set(turns.map(t => t.speaker_id))].sort()

  return (
    <div className="timeline">
      {speakers.map(spk => (
        <div key={spk} className="tl-row">
          <div className="tl-label">
            <span className="spk-pill" style={{ background: speakerColor(spk) + '22', color: speakerColor(spk), borderColor: speakerColor(spk) + '44', border: '1px solid' }}>
              {spk.replace('SPEAKER_', 'S')}
            </span>
          </div>
          <div className="tl-track" style={{ flex: 1 }}>
            {turns.filter(t => t.speaker_id === spk).map((t, i) => (
              <div
                key={i}
                className="tl-seg"
                title={`${t.start.toFixed(1)}s – ${t.end.toFixed(1)}s\n${t.transcript || ''}`}
                style={{
                  left: `${(t.start / duration) * 100}%`,
                  width: `${Math.max((t.duration / duration) * 100, 0.4)}%`,
                  background: speakerColor(spk),
                  opacity: 0.8,
                }}
                onClick={() => onTurnClick?.(t)}
              />
            ))}
          </div>
        </div>
      ))}
      <div style={{ display: 'flex', marginTop: 6, paddingLeft: 106 }}>
        {[0, 0.25, 0.5, 0.75, 1].map(p => (
          <span key={p} style={{ flex: p === 0 ? 0 : 1, fontSize: 10, color: 'var(--dimmer)', fontFamily: 'var(--mono)' }}>
            {p === 0 ? '0s' : fmtTime(duration * p)}
          </span>
        ))}
      </div>
    </div>
  )
}

// ─── NewJobView ───────────────────────────────────────────────────────────────
function NewJobView({ onJobSubmitted }) {
  const [tab, setTab] = useState('youtube')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState(null)
  const [sourceName, setSourceName] = useState('')
  const [numSpeakers, setNumSpeakers] = useState('')
  const [minSpeakers, setMinSpeakers] = useState('')
  const [maxSpeakers, setMaxSpeakers] = useState('')
  const [transcription, setTranscription] = useState(true)
  const [sentiment, setSentiment] = useState(true)
  const [saveVideo, setSaveVideo] = useState(false)
  const [broadcastChannel, setBroadcastChannel] = useState('')
  const [broadcastDate, setBroadcastDate] = useState('')
  const [cookiesBrowser, setCookiesBrowser] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const fileRef = useRef()

  async function submit() {
    setError('')
    setSubmitting(true)
    try {
      const fd = new FormData()
      if (tab === 'youtube') {
        if (!url.trim()) { setError('Please enter a YouTube URL.'); return }
        fd.append('url', url.trim())
      } else {
        if (!file) { setError('Please select an audio file.'); return }
        fd.append('audio_file', file)
      }
      if (sourceName) fd.append('source_name', sourceName)
      if (numSpeakers) fd.append('num_speakers', numSpeakers)
      if (minSpeakers) fd.append('min_speakers', minSpeakers)
      if (maxSpeakers) fd.append('max_speakers', maxSpeakers)
      fd.append('enable_transcription', transcription)
      fd.append('enable_sentiment', sentiment)
      if (broadcastChannel) fd.append('broadcast_channel', broadcastChannel)
      if (broadcastDate) fd.append('broadcast_date', broadcastDate)
      fd.append('save_video', saveVideo)

      const res = await apiFetch('/jobs', { method: 'POST', body: fd })
      onJobSubmitted(res.job_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  function onDrop(e) {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  return (
    <div>
      <div className="section-title">New Job</div>
      <div className="section-sub">Submit a YouTube video or local audio file for diarization.</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20, alignItems: 'start' }}>

        {/* Left — source */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="tabs">
              <button className={`tab ${tab === 'youtube' ? 'active' : ''}`} onClick={() => setTab('youtube')}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}><Youtube size={14} /> YouTube</span>
              </button>
              <button className={`tab ${tab === 'file' ? 'active' : ''}`} onClick={() => setTab('file')}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}><Upload size={14} /> Local file</span>
              </button>
            </div>

            {tab === 'youtube' ? (
              <>
                <div className="form-group">
                  <label className="form-label">YouTube URL</label>
                  <input className="form-input" placeholder="https://www.youtube.com/watch?v=..." value={url} onChange={e => setUrl(e.target.value)} />
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">Cookies from browser <span style={{ color: 'var(--dimmer)', textTransform: 'none', letterSpacing: 0 }}>(optional — fixes 403 errors)</span></label>
                  <select className="form-input" value={cookiesBrowser} onChange={e => setCookiesBrowser(e.target.value)}
                    style={{ background: 'var(--surf2)', cursor: 'pointer' }}>
                    <option value="">None</option>
                    {['safari','chrome','firefox','edge','chromium','brave','opera','vivaldi'].map(b => (
                      <option key={b} value={b}>{b.charAt(0).toUpperCase() + b.slice(1)}</option>
                    ))}
                  </select>
                </div>
              </>
            ) : (
              <div
                className={`dropzone ${drag ? 'drag' : ''}`}
                onDragOver={e => { e.preventDefault(); setDrag(true) }}
                onDragLeave={() => setDrag(false)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
              >
                <input ref={fileRef} type="file" accept=".wav,.mp3,.m4a,.flac,.ogg" style={{ display: 'none' }}
                  onChange={e => setFile(e.target.files[0])} />
                <div className="dropzone-icon"><Upload size={28} /></div>
                {file ? (
                  <div>
                    <div style={{ fontWeight: 500, color: 'var(--accent)' }}>{file.name}</div>
                    <div className="dropzone-sub">{(file.size / 1024 / 1024).toFixed(1)} MB</div>
                  </div>
                ) : (
                  <>
                    <div className="dropzone-text">Drop audio file here or click to browse</div>
                    <div className="dropzone-sub">WAV · MP3 · M4A · FLAC · OGG</div>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="card">
            <div className="form-group">
              <label className="form-label">Programme name <span style={{ color: 'var(--dimmer)', textTransform: 'none', letterSpacing: 0 }}>(optional)</span></label>
              <input className="form-input" placeholder="e.g. Politics Live" value={sourceName} onChange={e => setSourceName(e.target.value)} />
            </div>
            <div className="form-group">
              <label className="form-label">Broadcast channel <span style={{ color: 'var(--dimmer)', textTransform: 'none', letterSpacing: 0 }}>(optional)</span></label>
              <input className="form-input" placeholder="e.g. BBC Two" value={broadcastChannel} onChange={e => setBroadcastChannel(e.target.value)} />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Date of broadcast <span style={{ color: 'var(--dimmer)', textTransform: 'none', letterSpacing: 0 }}>(optional)</span></label>
              <input className="form-input" type="date" value={broadcastDate} onChange={e => setBroadcastDate(e.target.value)} />
            </div>
          </div>
        </div>

        {/* Right — config */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title"><Settings2 size={12} style={{ marginRight: 5, verticalAlign: 'middle' }} />Speaker hints</div>
            <div className="form-group">
              <label className="form-label">Exact count <span style={{ color: 'var(--dimmer)', textTransform: 'none', letterSpacing: 0 }}>— overrides min/max</span></label>
              <input className="form-input" type="number" min="1" max="20" placeholder="Auto-detect" value={numSpeakers} onChange={e => setNumSpeakers(e.target.value)} />
            </div>
            <div className="grid-2">
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Min</label>
                <input className="form-input" type="number" min="1" placeholder="—" value={minSpeakers} onChange={e => setMinSpeakers(e.target.value)} />
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Max</label>
                <input className="form-input" type="number" min="1" placeholder="—" value={maxSpeakers} onChange={e => setMaxSpeakers(e.target.value)} />
              </div>
            </div>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title"><Activity size={12} style={{ marginRight: 5, verticalAlign: 'middle' }} />Pipeline stages</div>
            <div className="toggle-row">
              <div>
                <div className="toggle-label">Transcription</div>
                <div className="toggle-desc">Whisper base.en — adds transcript per turn</div>
              </div>
              <Toggle value={transcription} onChange={setTranscription} />
            </div>
            <hr className="divider" style={{ margin: '10px 0' }} />
            <div className="toggle-row">
              <div>
                <div className="toggle-label">Sentiment</div>
                <div className="toggle-desc">DistilBERT SST-2 — requires transcription</div>
              </div>
              <Toggle value={sentiment} onChange={setSentiment} />
            </div>
            <hr className="divider" style={{ margin: '10px 0' }} />
            <div className="toggle-row">
              <div>
                <div className="toggle-label">Save video</div>
                <div className="toggle-desc">Download and keep the original video file</div>
              </div>
              <Toggle value={saveVideo} onChange={setSaveVideo} />
            </div>
          </div>

          {error && (
            <div className="notice notice-error">
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
              <span>{error}</span>
            </div>
          )}

          <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center', padding: '11px' }}
            onClick={submit} disabled={submitting}>
            {submitting ? <><Loader2 size={15} className="spin" /> Submitting…</> : <><Radio size={15} /> Start Processing</>}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── JobsView ─────────────────────────────────────────────────────────────────
// ─── Stage pipeline definitions ──────────────────────────────────────────────
// Each entry: { key, label, youtubeOnly? }
// The active stage is highlighted; completed stages get a check mark.
const STAGES_YOUTUBE = [
  { key: 'download',   label: 'Download'    },
  { key: 'models',     label: 'Models'      },
  { key: 'diarize',    label: 'Diarize'     },
  { key: 'transcribe', label: 'Transcribe'  },
  { key: 'sentiment',  label: 'Sentiment'   },
  { key: 'saving',     label: 'Save'        },
]
const STAGES_FILE = STAGES_YOUTUBE.filter(s => s.key !== 'download')

function stageIndex(stages, key) {
  return stages.findIndex(s => s.key === key)
}

// ─── JobProgressBar ───────────────────────────────────────────────────────────
function JobProgressBar({ job }) {
  const isYt     = job.source_type === 'youtube'
  const stages   = isYt ? STAGES_YOUTUBE : STAGES_FILE
  const running  = job.status === 'running'
  const complete = job.status === 'complete'
  const errored  = job.status === 'error'
  const pct      = complete ? 100 : (job.progress_pct || 0)
  const curIdx   = complete ? stages.length
                 : errored  ? -1
                 : stageIndex(stages, job.progress_stage)

  // Elapsed time
  const [elapsed, setElapsed] = useState('')
  useEffect(() => {
    if (!running) { setElapsed(''); return }
    const tick = () => {
      const ms = Date.now() - new Date(job.created_at).getTime()
      const s  = Math.floor(ms / 1000)
      const m  = Math.floor(s / 60)
      setElapsed(m > 0 ? `${m}m ${s % 60}s` : `${s}s`)
    }
    tick()
    const iv = setInterval(tick, 1000)
    return () => clearInterval(iv)
  }, [running, job.created_at])

  return (
    <div>
      {/* Stage pills */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
        {stages.map((s, i) => {
          const done    = complete || i < curIdx
          const active  = !complete && i === curIdx
          const pending = !complete && !errored && i > curIdx
          const color   = done    ? 'var(--pos)'
                        : active  ? 'var(--accent)'
                        : errored && i === curIdx ? 'var(--neg)'
                        : 'var(--dimmer)'
          const bg      = done    ? 'rgba(52,211,122,0.10)'
                        : active  ? 'rgba(232,160,32,0.12)'
                        : 'transparent'
          return (
            <span key={s.key} style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '2px 8px', borderRadius: 20, fontSize: 11, fontWeight: 500,
              color, background: bg,
              border: `1px solid ${done ? 'rgba(52,211,122,0.25)' : active ? 'rgba(232,160,32,0.3)' : 'var(--bord)'}`,
              transition: 'all 0.3s',
            }}>
              {done
                ? <CheckCircle2 size={10} />
                : active
                  ? <Loader2 size={10} style={{ animation: 'spin 1s linear infinite' }} />
                  : <span style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--bord2)', display: 'inline-block' }} />
              }
              {s.label}
            </span>
          )
        })}
      </div>

      {/* Percentage bar */}
      {(running || complete) && (
        <div style={{ marginBottom: 4 }}>
          <div style={{
            height: 5, background: 'var(--surf3)', borderRadius: 3, overflow: 'hidden',
          }}>
            <div style={{
              height: '100%', borderRadius: 3,
              width: `${pct}%`,
              background: complete ? 'var(--pos)' : 'var(--accent)',
              transition: 'width 0.4s ease',
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
            <span style={{ fontSize: 11, color: 'var(--dim)', fontFamily: 'var(--mono)' }}>
              {job.progress_detail || job.progress || ''}
            </span>
            <span style={{ fontSize: 11, color: 'var(--dim)', fontFamily: 'var(--mono)', display: 'flex', gap: 10 }}>
              {elapsed && <span>{elapsed}</span>}
              <span style={{ color: complete ? 'var(--pos)' : 'var(--accent)' }}>{pct}%</span>
            </span>
          </div>
        </div>
      )}

      {/* Error message */}
      {errored && (
        <div style={{ fontSize: 12, color: 'var(--neg)', marginTop: 4, fontFamily: 'var(--mono)', wordBreak: 'break-all' }}>
          {job.error}
        </div>
      )}
    </div>
  )
}

// ─── JobsView ─────────────────────────────────────────────────────────────────
function JobsView({ onViewSession }) {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try { setJobs(await apiFetch('/jobs')) } catch {}
    setLoading(false)
  }, [])

  useEffect(() => {
    refresh()
    // Poll every 2s while any job is running, otherwise 10s
    let iv = setInterval(async () => {
      const updated = await apiFetch('/jobs').catch(() => null)
      if (updated) setJobs(updated)
      const hasActive = updated?.some(j => j.status === 'running' || j.status === 'queued')
      clearInterval(iv)
      iv = setInterval(refresh, hasActive ? 2000 : 10000)
    }, 2000)
    return () => clearInterval(iv)
  }, [refresh])

  if (loading) return (
    <div style={{ color: 'var(--dim)', padding: 40, textAlign: 'center' }}>
      <Loader2 size={20} className="spin" />
    </div>
  )

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <div className="section-title">Jobs</div>
          <div className="section-sub">Processing queue — live progress updates.</div>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={refresh}><RefreshCw size={13} /> Refresh</button>
      </div>

      {jobs.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '40px', color: 'var(--dim)' }}>
          No jobs yet. Submit a YouTube URL or audio file to get started.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {jobs.map(j => (
            <div key={j.job_id} className="card" style={{
              borderLeft: `3px solid ${
                j.status === 'complete' ? 'var(--pos)'
                : j.status === 'error'  ? 'var(--neg)'
                : j.status === 'running'? 'var(--accent)'
                : 'var(--bord2)'
              }`,
              borderRadius: '0 10px 10px 0',
            }}>
              {/* Header row */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <StatusChip status={j.status} />
                  <div>
                    <div style={{ fontWeight: 500, fontSize: 14 }}>{j.source_name || j.source_ref}</div>
                    <div style={{ fontSize: 11, color: 'var(--dim)', fontFamily: 'var(--mono)', marginTop: 1 }}>
                      {j.job_id} · {j.source_type} · {fmtDate(j.created_at)}
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
                  {j.status === 'complete' && j.session_id && (
                    <button className="btn btn-ghost btn-sm" onClick={() => onViewSession(j.session_id)}>
                      <Eye size={13} /> View results
                    </button>
                  )}
                </div>
              </div>

              {/* Progress pipeline */}
              <JobProgressBar job={j} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── TranscriptView ───────────────────────────────────────────────────────────
function TranscriptView({ turns, linkedName }) {
  const [copied, setCopied] = useState(false)

  function speakerLabel(spk) {
    return linkedName(spk) || spk.replace('SPEAKER_', 'Speaker ')
  }

  function copyText() {
    const lines = turns.map(t => {
      const name = speakerLabel(t.speaker_id)
      const time = fmtTimestamp(t.start)
      return `[${time}]  ${name}\n${t.transcript || '(no transcript)'}`
    })
    navigator.clipboard.writeText(lines.join('\n\n')).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  if (!turns.length) return (
    <div className="card" style={{ color: 'var(--dim)', textAlign: 'center', padding: 32 }}>No turns available.</div>
  )

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <button className="btn btn-ghost btn-sm" onClick={copyText}>
          {copied ? <CheckCircle2 size={13} /> : <List size={13} />}
          {copied ? 'Copied!' : 'Copy transcript'}
        </button>
      </div>
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ maxHeight: 620, overflowY: 'auto', padding: '8px 0' }}>
          {turns.map((t, i) => {
            const color = speakerColor(t.speaker_id)
            const name = speakerLabel(t.speaker_id)
            return (
              <div key={i} style={{
                display: 'grid',
                gridTemplateColumns: '72px 160px 1fr',
                gap: '0 16px',
                padding: '8px 20px',
                borderBottom: '1px solid var(--bord)',
                alignItems: 'baseline',
              }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--dimmer)', paddingTop: 1 }}>
                  {fmtTimestamp(t.start)}
                </span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600, color, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {name}
                </span>
                <span style={{ fontSize: 13, color: t.transcript ? 'var(--text)' : 'var(--dimmer)', fontStyle: t.transcript ? 'normal' : 'italic', lineHeight: 1.55 }}>
                  {t.transcript || 'no transcript'}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ─── SessionDetail ────────────────────────────────────────────────────────────
function SessionDetail({ sessionId, onBack, onReview, speakers, onRefreshSpeakers }) {
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('timeline')
  const [selectedSpeaker, setSelectedSpeaker] = useState(null)
  const [linkModal, setLinkModal] = useState(null) // { ephemeralId }
  const [linkCatId, setLinkCatId] = useState('')
  const [linkErr, setLinkErr] = useState('')
  const [linkOk, setLinkOk] = useState('')
  const [filterSpk, setFilterSpk] = useState(null)
  const [editingMeta, setEditingMeta] = useState(false)
  const [metaForm, setMetaForm] = useState({ source_name: '', broadcast_date: '', broadcast_channel: '' })
  const [metaSaving, setMetaSaving] = useState(false)
  const [metaErr, setMetaErr] = useState('')

  useEffect(() => {
    apiFetch(`/sessions/${sessionId}`)
      .then(d => { setSession(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [sessionId])

  function openMetaEdit() {
    setMetaForm({
      source_name: session.source_name || '',
      broadcast_date: session.broadcast_date || '',
      broadcast_channel: session.broadcast_channel || '',
    })
    setMetaErr('')
    setEditingMeta(true)
  }

  async function saveMeta() {
    setMetaSaving(true)
    setMetaErr('')
    try {
      await apiFetch(`/sessions/${sessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metaForm),
      })
      const updated = await apiFetch(`/sessions/${sessionId}`)
      setSession(updated)
      setEditingMeta(false)
    } catch (e) {
      setMetaErr(e.message)
    } finally {
      setMetaSaving(false)
    }
  }

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}><Loader2 size={20} className="spin" /></div>
  if (!session) return <div className="notice notice-error"><AlertCircle size={16} /> Session not found.</div>

  const result = session.result || {}
  const turns = result.turns || []
  const stats = result.speaker_stats || {}
  const spkIds = Object.keys(stats).sort()
  const links = session.links || []

  function linkedName(ephId) {
    const l = links.find(x => x.ephemeral_id === ephId)
    return l?.display_name ? l.display_name : null
  }

  async function doLink() {
    setLinkErr('')
    try {
      const fd = new FormData()
      fd.append('session_id', sessionId)
      fd.append('ephemeral_id', linkModal.ephemeralId)
      fd.append('catalogue_id', linkCatId.trim())
      await apiFetch('/link', { method: 'POST', body: fd })
      setLinkOk(`Linked ${linkModal.ephemeralId} → ${linkCatId.trim()}`)
      setLinkModal(null)
      setLinkCatId('')
      const updated = await apiFetch(`/sessions/${sessionId}`)
      setSession(updated)
      onRefreshSpeakers?.()
    } catch (e) {
      setLinkErr(e.message)
    }
  }

  const displayTurns = filterSpk ? turns.filter(t => t.speaker_id === filterSpk) : turns

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <button className="btn btn-ghost btn-sm" onClick={onBack}><ArrowLeft size={14} /> Back</button>
        <div style={{ flex: 1 }}>
          <div className="section-title" style={{ marginBottom: 0 }}>{session.source_name || result.source_name}</div>
          <div style={{ fontSize: 12, color: 'var(--dim)', fontFamily: 'var(--mono)' }}>{sessionId}</div>
          {(session.broadcast_date || session.broadcast_channel) && (
            <div style={{ fontSize: 12, color: 'var(--dim)', marginTop: 2 }}>
              {[session.broadcast_channel, session.broadcast_date].filter(Boolean).join(' · ')}
            </div>
          )}
        </div>
        <button className="btn btn-ghost btn-sm" onClick={openMetaEdit}><Edit3 size={14} /> Edit info</button>
        {onReview && <button className="btn btn-ghost btn-sm" onClick={() => onReview(sessionId)}><Eye size={14} /> Review</button>}
      </div>

      {editingMeta && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="section-title" style={{ fontSize: 13, marginBottom: 12 }}>Edit session info</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginBottom: 12 }}>
            <label style={{ fontSize: 12 }}>
              <div style={{ marginBottom: 4, color: 'var(--dim)' }}>Programme name</div>
              <input className="input" value={metaForm.source_name}
                onChange={e => setMetaForm(f => ({ ...f, source_name: e.target.value }))} />
            </label>
            <label style={{ fontSize: 12 }}>
              <div style={{ marginBottom: 4, color: 'var(--dim)' }}>Date of broadcast</div>
              <input className="input" type="date" value={metaForm.broadcast_date}
                onChange={e => setMetaForm(f => ({ ...f, broadcast_date: e.target.value }))} />
            </label>
            <label style={{ fontSize: 12 }}>
              <div style={{ marginBottom: 4, color: 'var(--dim)' }}>Broadcast channel</div>
              <input className="input" value={metaForm.broadcast_channel}
                onChange={e => setMetaForm(f => ({ ...f, broadcast_channel: e.target.value }))} />
            </label>
          </div>
          {metaErr && <div className="notice notice-error" style={{ marginBottom: 8 }}><AlertCircle size={14} />{metaErr}</div>}
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-primary btn-sm" onClick={saveMeta} disabled={metaSaving}>
              {metaSaving ? <Loader2 size={13} className="spin" /> : <CheckCircle2 size={13} />} Save
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setEditingMeta(false)}>Cancel</button>
          </div>
        </div>
      )}

      {linkOk && <div className="notice notice-ok"><CheckCircle2 size={16} />{linkOk}</div>}

      {/* Stats row */}
      <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(4,1fr)', marginBottom: 20 }}>
        <div className="stat">
          <div className="stat-label">Duration</div>
          <div className="stat-val">{fmtTime(result.total_duration)}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Speakers</div>
          <div className="stat-val">{result.num_speakers || spkIds.length}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Turns</div>
          <div className="stat-val">{turns.length}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Recorded</div>
          <div className="stat-val" style={{ fontSize: 16 }}>{fmtDate(result.processed_at).split(',')[0]}</div>
        </div>
      </div>

      {/* Speaker cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px,1fr))', gap: 10, marginBottom: 20 }}>
        {spkIds.map(spk => {
          const s = stats[spk]
          const color = speakerColor(spk)
          const name = linkedName(spk)
          const link = links.find(x => x.ephemeral_id === spk)
          return (
            <div key={spk} className="card card-sm" style={{ borderLeft: `3px solid ${color}`, cursor: 'pointer', borderRadius: '0 var(--radius) var(--radius) 0' }}
              onClick={() => setFilterSpk(filterSpk === spk ? null : spk)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color }}>
                    {spk.replace('SPEAKER_', 'SPEAKER_')}
                  </div>
                  {name
                    ? <div style={{ fontWeight: 500, fontSize: 13, marginTop: 2 }}>{name}</div>
                    : <button className="link-btn" style={{ fontSize: 11, marginTop: 2 }}
                        onClick={e => { e.stopPropagation(); setLinkModal({ ephemeralId: spk }); setLinkErr(''); setLinkCatId('') }}>
                        <Link2 size={10} style={{ marginRight: 3 }} /> Link speaker
                      </button>
                  }
                  {link?.affiliation && <div style={{ fontSize: 11, color: 'var(--dim)' }}>{link.affiliation}</div>}
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 500 }}>{s.pct_of_audio?.toFixed(1)}%</div>
                  <div style={{ fontSize: 11, color: 'var(--dim)' }}>{fmtTime(s.total_speaking_time)}</div>
                </div>
              </div>
              <div style={{ marginTop: 8, display: 'flex', gap: 10, fontSize: 11, color: 'var(--dim)' }}>
                <span>{s.turn_count} turns</span>
                {s.avg_sentiment != null && (
                  <span style={{ color: sentimentColor(s.avg_sentiment), display: 'flex', alignItems: 'center', gap: 3 }}>
                    {sentimentIcon(s.avg_sentiment)} {s.avg_sentiment > 0 ? '+' : ''}{s.avg_sentiment?.toFixed(2)}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Content tabs */}
      <div className="tabs">
        {['timeline','turns','transcript'].map(t => (
          <button key={t} className={`tab ${activeTab === t ? 'active' : ''}`} onClick={() => setActiveTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
        {filterSpk && activeTab !== 'transcript' && (
          <span style={{ alignSelf: 'center', fontSize: 12, color: 'var(--accent)', marginLeft: 8 }}>
            Filtered: {filterSpk} — <button className="link-btn" onClick={() => setFilterSpk(null)}>clear</button>
          </span>
        )}
      </div>

      {activeTab === 'timeline' && (
        <div className="card">
          <SpeakerTimeline turns={turns} duration={result.total_duration} onTurnClick={t => setFilterSpk(t.speaker_id)} />
        </div>
      )}

      {activeTab === 'turns' && (
        <div style={{ maxHeight: 520, overflowY: 'auto' }}>
          {displayTurns.map((t, i) => (
            <div key={i} className="turn-item" style={{ borderLeftColor: speakerColor(t.speaker_id) }}>
              <div className="turn-meta">
                <span className="spk-pill" style={{ background: speakerColor(t.speaker_id) + '22', color: speakerColor(t.speaker_id), border: '1px solid ' + speakerColor(t.speaker_id) + '44' }}>
                  {t.speaker_id.replace('SPEAKER_', 'S')}
                  {linkedName(t.speaker_id) && <span style={{ marginLeft: 5, fontFamily: 'var(--font)', fontWeight: 500 }}>{linkedName(t.speaker_id)}</span>}
                </span>
                <span className="turn-time">{t.start?.toFixed(1)}s – {t.end?.toFixed(1)}s</span>
                <span className="turn-time">{fmtTime(t.duration)}</span>
                {t.sentiment_score != null && (
                  <span style={{ color: sentimentColor(t.sentiment_score), fontSize: 11, display: 'flex', alignItems: 'center', gap: 3 }}>
                    {sentimentIcon(t.sentiment_score)} {t.sentiment}
                  </span>
                )}
              </div>
              {t.transcript
                ? <div className="turn-text">"{t.transcript}"</div>
                : <div className="turn-no-text">no transcript</div>
              }
            </div>
          ))}
        </div>
      )}

      {activeTab === 'transcript' && (
        <TranscriptView turns={turns} linkedName={linkedName} />
      )}

      {/* Link modal */}
      {linkModal && (
        <div className="modal-overlay" onClick={() => setLinkModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-title">Link <span style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>{linkModal.ephemeralId}</span></div>
            <p style={{ fontSize: 13, color: 'var(--dim)', marginBottom: 16 }}>
              Enter an existing catalogue ID (e.g. <span className="mono">SPK-0001</span>), or go to the Speakers tab to create a new entry first.
            </p>
            <div className="form-group">
              <label className="form-label">Catalogue ID</label>
              <input className="form-input" placeholder="SPK-0001" value={linkCatId} onChange={e => setLinkCatId(e.target.value)} />
            </div>
            {linkErr && <div className="notice notice-error" style={{ marginBottom: 12 }}><AlertCircle size={14} />{linkErr}</div>}
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost" onClick={() => setLinkModal(null)}>Cancel</button>
              <button className="btn btn-primary" onClick={doLink}><Link2 size={14} /> Link</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── ReviewView ───────────────────────────────────────────────────────────────
const REVIEW_PALETTE = [
  '#6366f1','#22c55e','#f59e0b','#ec4899','#14b8a6',
  '#f97316','#a78bfa','#34d399','#fb923c','#38bdf8',
]
const COMPRESS_GAP = 9999

function ReviewView({ sessionId, onBack }) {
  const [reviewData, setReviewData]     = useState(null)
  const [loading, setLoading]           = useState(true)
  const [toast, setToast]               = useState(null)
  const [activeFilter, setActiveFilter] = useState(null)
  const [playingIdx, setPlayingIdx]     = useState(null)
  const audioEls                        = useRef({})
  const [reassignOpen, setReassignOpen] = useState(null)
  const [mergeExpandEph, setMergeExpandEph] = useState(null)
  const [mergeTargetVal, setMergeTargetVal] = useState('')
  const [subTurnsOpen, setSubTurnsOpen] = useState({})
  const [compressMode, setCompressMode] = useState(false)
  const [showMergePanel, setShowMergePanel] = useState(false)
  const [mergeGap, setMergeGap]         = useState(1.0)
  const [mergeCounts, setMergeCounts]   = useState(null)
  const [mergePreviewActive, setMergePreviewActive] = useState(false)
  const [previewTurns, setPreviewTurns] = useState(null)
  const mergeDebounce                   = useRef(null)
  const [confirmModal, setConfirmModal] = useState(null)
  const [confirmForm, setConfirmForm]   = useState({ name: '', affiliation: '', role: '', notes: '' })
  const [searchModal, setSearchModal]   = useState(null)
  const [searchQuery, setSearchQuery]   = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [frameModal, setFrameModal]     = useState(null)
  const [frameStatus, setFrameStatus]   = useState('loading')

  function showToast(msg, type = 'success') {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  async function reload() {
    try {
      const data = await apiFetch(`/review/${sessionId}`)
      setReviewData(data)
    } catch (e) { showToast('Reload failed: ' + e.message, 'error') }
  }

  useEffect(() => {
    apiFetch(`/review/${sessionId}`)
      .then(d => { setReviewData(d); setLoading(false) })
      .catch(e => { showToast('Failed to load: ' + e.message, 'error'); setLoading(false) })
  }, [sessionId])

  // Keyboard shortcuts
  useEffect(() => {
    function handle(e) {
      if (frameModal) {
        if (e.key === 'ArrowLeft')  { e.preventDefault(); doStepFrame(-1) }
        if (e.key === 'ArrowRight') { e.preventDefault(); doStepFrame(1) }
        if (e.key === 'Escape' || e.key === 'f' || e.key === 'F') setFrameModal(null)
        return
      }
      if (e.key === 'Escape') { setReassignOpen(null); setConfirmModal(null); setSearchModal(null) }
    }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [frameModal]) // eslint-disable-line

  function spkColor(ephId, allIds) {
    const idx = (allIds || []).indexOf(ephId)
    return REVIEW_PALETTE[Math.max(0, idx) % REVIEW_PALETTE.length]
  }

  function fmtRev(s) {
    const m = Math.floor(s / 60), sec = Math.floor(s % 60)
    return `${m}:${String(sec).padStart(2, '0')}`
  }

  function buildCompressedRuns(allTurns) {
    const result = []
    let i = 0
    while (i < allTurns.length) {
      const first = allTurns[i]
      if (first.deleted) { result.push({ ...first, _runLen: 1 }); i++; continue }
      let j = i + 1
      while (j < allTurns.length && !allTurns[j].deleted && allTurns[j].original_speaker === first.original_speaker) j++
      const run = allTurns.slice(i, j)
      if (run.length === 1) {
        result.push({ ...first, _runLen: 1 })
      } else {
        const parts = run.map(t => (t.transcript || '').trim()).filter(Boolean)
        const last = run[run.length - 1]
        result.push({ ...first, end: last.end, duration: last.end - first.start, transcript: parts.join('...') || null, _compressed: true, _runLen: run.length })
      }
      i = j
    }
    return result
  }

  function scrollToSpeaker(ephId) {
    if (!reviewData) return
    const sp = reviewData.speakers.find(s => s.ephemeral_id === ephId)
    if (!sp || sp.turns.length === 0) return
    const firstIdx = sp.turns.reduce((a, b) => a.index < b.index ? a : b).index
    const el = document.getElementById(`rv-turn-${firstIdx}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  function stopAudio() {
    if (playingIdx === null) return
    const el = audioEls.current[playingIdx]
    if (el) { el.pause(); el.currentTime = 0 }
    setPlayingIdx(null)
  }

  async function togglePlay(idx, start, end) {
    if (playingIdx === idx) { stopAudio(); return }
    stopAudio()
    try {
      const url = `/api/review/${sessionId}/audio/${idx}?start=${start}&end=${end}`
      let el = audioEls.current[idx]
      if (!el) { el = new Audio(); audioEls.current[idx] = el }
      if (el.dataset.src !== url) { el.src = url; el.dataset.src = url; el.load() }
      el.currentTime = 0
      el.onended = () => setPlayingIdx(null)
      await el.play()
      setPlayingIdx(idx)
    } catch { showToast('Audio not available for this source', 'error') }
  }

  async function submitReassign(idx, newSpeaker) {
    setReassignOpen(null)
    try {
      const fd = new FormData()
      fd.append('assigned_speaker', newSpeaker)
      await apiFetch(`/review/${sessionId}/turns/${idx}/assign`, { method: 'POST', body: fd })
      showToast(`Turn ${idx} reassigned to ${newSpeaker}`)
      await reload()
    } catch (e) { showToast('Reassign failed: ' + e.message, 'error') }
  }

  async function deleteTurn(idx) {
    try {
      await apiFetch(`/review/${sessionId}/turns/${idx}`, { method: 'DELETE' })
      showToast(`Turn ${idx} deleted`)
      await reload()
    } catch (e) { showToast('Delete failed: ' + e.message, 'error') }
  }

  async function restoreTurn(idx) {
    try {
      await apiFetch(`/review/${sessionId}/turns/${idx}/restore`, { method: 'POST' })
      showToast(`Turn ${idx} restored`)
      await reload()
    } catch (e) { showToast('Restore failed: ' + e.message, 'error') }
  }

  async function submitSpeakerMerge(ephId) {
    if (!mergeTargetVal) return
    try {
      const fd = new FormData()
      fd.append('target_speaker', mergeTargetVal)
      const res = await apiFetch(`/review/${sessionId}/speakers/${ephId}/merge`, { method: 'POST', body: fd })
      showToast(`Merged ${res.merged_turns} turns: ${ephId} → ${mergeTargetVal}`)
      setMergeExpandEph(null)
      await reload()
    } catch (e) { showToast('Merge failed: ' + e.message, 'error') }
  }

  function openConfirm(ephId) {
    const sp = reviewData.speakers.find(s => s.ephemeral_id === ephId)
    setConfirmForm({
      name: sp?.display_name || sp?.suggested_name || '',
      affiliation: sp?.affiliation || sp?.suggested_org || '',
      role: sp?.role || sp?.suggested_title || '',
      notes: '',
    })
    setConfirmModal({ ephId, sp })
  }

  async function submitConfirm() {
    const { ephId, sp } = confirmModal
    setConfirmModal(null)
    try {
      let catalogueId = sp?.catalogue_id
      const fd = new FormData()
      if (confirmForm.name)        fd.append('name', confirmForm.name)
      if (confirmForm.affiliation) fd.append('affiliation', confirmForm.affiliation)
      if (confirmForm.role)        fd.append('role', confirmForm.role)
      if (confirmForm.notes)       fd.append('notes', confirmForm.notes)
      if (!catalogueId) {
        const res = await apiFetch('/speakers', { method: 'POST', body: fd })
        catalogueId = res.catalogue_id
      } else {
        await apiFetch(`/speakers/${catalogueId}`, { method: 'PUT', body: fd })
      }
      const lfd = new FormData()
      lfd.append('session_id', sessionId)
      lfd.append('ephemeral_id', ephId)
      lfd.append('catalogue_id', catalogueId)
      await apiFetch('/link', { method: 'POST', body: lfd })
      showToast(`Speaker confirmed: ${confirmForm.name || ephId}`)
      await reload()
    } catch (e) { showToast('Error confirming speaker: ' + e.message, 'error') }
  }

  function openSearch(ephId) {
    setSearchModal({ ephId })
    setSearchQuery('')
    setSearchResults([])
  }

  async function doSearch(q) {
    setSearchQuery(q)
    if (!q.trim()) { setSearchResults([]); return }
    try {
      const results = await apiFetch(`/speakers?search=${encodeURIComponent(q)}`)
      setSearchResults(results.slice(0, 10))
    } catch {}
  }

  async function selectAndLink(catalogueId) {
    const ephId = searchModal.ephId
    setSearchModal(null)
    try {
      const fd = new FormData()
      fd.append('session_id', sessionId)
      fd.append('ephemeral_id', ephId)
      fd.append('catalogue_id', catalogueId)
      await apiFetch('/link', { method: 'POST', body: fd })
      showToast(`Linked ${ephId} → ${catalogueId}`)
      await reload()
    } catch (e) { showToast('Link error: ' + e.message, 'error') }
  }

  function doStepFrame(delta) {
    if (!reviewData) return
    const allSorted = reviewData.speakers.flatMap(s => s.turns).sort((a, b) => a.index - b.index)
    setFrameModal(prev => {
      if (!prev) return null
      const pos = allSorted.findIndex(t => t.index === prev.idx)
      const next = allSorted[pos + delta]
      if (!next) return prev
      setFrameStatus('loading')
      return { idx: next.index }
    })
  }

  async function fetchMergePreview(gap) {
    try {
      const fd = new FormData()
      fd.append('merge_gap_secs', gap)
      const res = await apiFetch(`/review/${sessionId}/remerge`, { method: 'POST', body: fd })
      setMergeCounts({ original: res.original_count, merged: res.merged_count })
      if (mergePreviewActive) setPreviewTurns(res.turns)
    } catch {}
  }

  function scheduleMergePreview(gap) {
    clearTimeout(mergeDebounce.current)
    mergeDebounce.current = setTimeout(() => fetchMergePreview(gap), 280)
  }

  useEffect(() => {
    if (showMergePanel && reviewData) {
      const gap = reviewData.session.merge_gap_secs ?? 1.0
      setMergeGap(gap)
      scheduleMergePreview(gap)
    }
  }, [showMergePanel]) // eslint-disable-line

  async function applyRemerge() {
    const gap = compressMode ? COMPRESS_GAP : mergeGap
    const hasOverrides = reviewData?.speakers.some(s => s.turns.some(t => t.overridden || t.deleted))
    let msg = `Apply merge gap ${gap >= COMPRESS_GAP ? '∞ (compress)' : gap + 's'} to this session?`
    if (hasOverrides) msg += '\n\nWARNING: This will clear all speaker reassignments and deletions.'
    if (!confirm(msg)) return
    try {
      const fd = new FormData()
      fd.append('merge_gap_secs', gap)
      const res = await apiFetch(`/review/${sessionId}/apply-remerge`, { method: 'POST', body: fd })
      showToast(`Applied: ${res.original_count} → ${res.merged_count} turns`)
      setMergePreviewActive(false)
      setPreviewTurns(null)
      setCompressMode(false)
      await reload()
    } catch (e) { showToast('Apply failed: ' + e.message, 'error') }
  }

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}><Loader2 size={20} className="spin" /></div>
  if (!reviewData) return <div className="notice notice-error"><AlertCircle size={16} /> Failed to load review data.</div>

  const { session, speakers, all_ephemeral_ids } = reviewData
  const speakerLookup = Object.fromEntries(speakers.map(s => [s.ephemeral_id, s]))
  const allTurns = speakers.flatMap(s => s.turns).sort((a, b) => a.index - b.index)
  const displayTurns = compressMode ? buildCompressedRuns(allTurns) : allTurns
  const filteredTurns = activeFilter
    ? displayTurns.filter(t => (t.deleted ? t.original_speaker : t.effective_speaker) === activeFilter)
    : displayTurns
  const shownTurns = previewTurns || filteredTurns
  const totalTurns = speakers.reduce((a, s) => a + s.turn_count, 0)
  const totalOverrides = allTurns.filter(t => t.overridden && !t.deleted).length
  const totalDeleted   = allTurns.filter(t => t.deleted).length

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 56px)', overflow: 'hidden' }}>

      {/* Toast */}
      {toast && (
        <div style={{
          position: 'fixed', bottom: 20, right: 20, zIndex: 9999,
          background: toast.type === 'error' ? '#2a1010' : '#0d2a1a',
          border: `1px solid ${toast.type === 'error' ? '#5a2020' : '#1a4020'}`,
          color: toast.type === 'error' ? '#f08080' : '#60d090',
          padding: '10px 16px', borderRadius: 8, fontSize: 13, boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        }}>{toast.msg}</div>
      )}

      {/* ── Left panel: speakers ── */}
      <div style={{ width: 290, minWidth: 290, background: 'var(--surf)', borderRight: '1px solid var(--bord)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--bord)', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <button className="btn btn-ghost btn-sm" onClick={onBack}><ArrowLeft size={13} /></button>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {session.source_name || sessionId}
              </div>
              <div style={{ fontSize: 11, color: 'var(--dim)' }}>{fmtTime(session.total_duration)} · {session.num_speakers} speakers</div>
            </div>
          </div>
          <button className="btn btn-ghost btn-sm" style={{ width: '100%', justifyContent: 'center', fontSize: 11, ...(showMergePanel ? { color: 'var(--accent)', borderColor: 'var(--accent)' } : {}) }}
            onClick={() => setShowMergePanel(v => !v)}>
            ⇒ Merge tuning
          </button>
        </div>

        {/* Speaker cards */}
        <div style={{ overflowY: 'auto', flex: 1, padding: 8 }}>
          {speakers.map(sp => {
            const color    = spkColor(sp.ephemeral_id, all_ephemeral_ids)
            const pct      = ((sp.total_speaking_time / (session.total_duration || 1)) * 100).toFixed(1)
            const initials = sp.ephemeral_id.replace('SPEAKER_', 'S')
            const isFiltered = activeFilter === sp.ephemeral_id
            return (
              <div key={sp.ephemeral_id}
                style={{ background: isFiltered ? 'var(--surf3)' : 'var(--surf2)', border: `1px solid ${isFiltered ? color + '66' : 'var(--bord)'}`, borderRadius: 8, padding: '10px 12px', marginBottom: 8, cursor: 'pointer' }}
                onClick={() => scrollToSpeaker(sp.ephemeral_id)}
              >
                {/* Top row */}
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 6 }}>
                  <div style={{ width: 36, height: 36, borderRadius: '50%', border: `2px solid ${color}`, overflow: 'hidden', flexShrink: 0, background: 'var(--surf3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color, fontWeight: 600 }}>
                    {sp.frame_url
                      ? <img src={sp.frame_url} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={e => { e.target.style.display = 'none' }} />
                      : initials
                    }
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 500, fontSize: 13, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {sp.display_name || sp.ephemeral_id}
                    </div>
                    {(sp.affiliation || sp.role) && (
                      <div style={{ fontSize: 11, color: 'var(--dim)' }}>{[sp.affiliation, sp.role].filter(Boolean).join(' · ')}</div>
                    )}
                    <div style={{ fontSize: 10, color: 'var(--dimmer)', fontFamily: 'var(--mono)' }}>{sp.ephemeral_id}</div>
                    {sp.suggested_name && sp.confidence && (
                      <span style={{ fontSize: 10, background: 'var(--surf)', border: '1px solid var(--bord2)', borderRadius: 4, padding: '1px 5px', color: sp.confidence === 'high' ? 'var(--pos)' : sp.confidence === 'medium' ? 'var(--warn)' : 'var(--dim)' }}>
                        {sp.suggested_name} [{sp.confidence}]
                      </span>
                    )}
                  </div>
                </div>
                {/* Time bar */}
                <div style={{ marginBottom: 8 }}>
                  <div style={{ height: 4, background: 'var(--surf3)', borderRadius: 2, overflow: 'hidden', marginBottom: 3 }}>
                    <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2 }} />
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--dim)' }}>{sp.total_speaking_time.toFixed(1)}s · {sp.turn_count} turns · {pct}%</div>
                </div>
                {/* Action buttons */}
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }} onClick={e => e.stopPropagation()}>
                  <button className="btn btn-sm" style={{ fontSize: 11, padding: '3px 8px', background: '#0d2a1a', color: 'var(--pos)', border: '1px solid #1a4020' }}
                    onClick={() => openConfirm(sp.ephemeral_id)}>✓ Confirm</button>
                  <button className="btn btn-sm btn-danger" style={{ fontSize: 11, padding: '3px 8px' }}
                    onClick={() => openSearch(sp.ephemeral_id)}>✗ Wrong</button>
                  <button className="btn btn-sm btn-ghost" style={{ fontSize: 11, padding: '3px 8px' }}
                    onClick={() => { setMergeExpandEph(v => v === sp.ephemeral_id ? null : sp.ephemeral_id); setMergeTargetVal('') }}>⇒ Merge</button>
                  <button className="btn btn-sm btn-ghost" style={{ fontSize: 11, padding: '3px 8px', ...(isFiltered ? { color: 'var(--accent)', borderColor: 'var(--accent)' } : {}) }}
                    onClick={() => setActiveFilter(f => f === sp.ephemeral_id ? null : sp.ephemeral_id)}>⊙ {isFiltered ? 'Clear' : 'Filter'}</button>
                </div>
                {/* Merge inline form */}
                {mergeExpandEph === sp.ephemeral_id && (
                  <div style={{ marginTop: 8, padding: 8, background: 'var(--surf)', borderRadius: 6, border: '1px solid var(--bord)' }} onClick={e => e.stopPropagation()}>
                    <div style={{ fontSize: 11, color: 'var(--dim)', marginBottom: 6 }}>Merge all turns of {sp.ephemeral_id} into:</div>
                    <select style={{ width: '100%', background: 'var(--surf2)', color: 'var(--text)', border: '1px solid var(--bord2)', borderRadius: 4, padding: '4px 8px', fontSize: 12, marginBottom: 6 }}
                      value={mergeTargetVal} onChange={e => setMergeTargetVal(e.target.value)}>
                      <option value="">— select —</option>
                      {speakers.filter(s => s.ephemeral_id !== sp.ephemeral_id).map(s => (
                        <option key={s.ephemeral_id} value={s.ephemeral_id}>{s.display_name || s.ephemeral_id}</option>
                      ))}
                    </select>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="btn btn-ghost btn-sm" style={{ fontSize: 11 }} onClick={() => setMergeExpandEph(null)}>Cancel</button>
                      <button className="btn btn-primary btn-sm" style={{ fontSize: 11 }} onClick={() => submitSpeakerMerge(sp.ephemeral_id)}>Merge</button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Right panel: transcript ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Transcript header */}
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--bord)', background: 'var(--surf)', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ fontSize: 12, color: 'var(--dim)', flex: 1 }}>
            {totalTurns} turns · {session.num_speakers} speakers · {totalOverrides} override{totalOverrides !== 1 ? 's' : ''}
            {totalDeleted > 0 && ` · ${totalDeleted} deleted`}
            {activeFilter && (
              <span style={{ color: 'var(--accent)', marginLeft: 8 }}>
                Filtered: {speakerLookup[activeFilter]?.display_name || activeFilter} —{' '}
                <button className="link-btn" style={{ fontSize: 12 }} onClick={() => setActiveFilter(null)}>clear</button>
              </span>
            )}
            {previewTurns && <span style={{ color: 'var(--warn)', marginLeft: 8 }}>● Preview mode</span>}
          </div>
          <button className="btn btn-ghost btn-sm" style={{ fontSize: 11, ...(compressMode ? { color: 'var(--accent)', borderColor: 'var(--accent)' } : {}) }}
            onClick={() => setCompressMode(v => !v)}>
            ⇒ {compressMode ? 'Compress (on)' : 'Compress'}
          </button>
        </div>

        {/* Merge tuning panel */}
        {showMergePanel && (
          <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--bord)', background: 'var(--surf2)', flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, color: 'var(--dim)' }}>Merge gap:</span>
              <input type="range" min="0" max="5" step="0.1" value={Math.min(mergeGap, 5)} disabled={compressMode}
                style={{ width: 100, accentColor: 'var(--accent)', opacity: compressMode ? 0.35 : 1 }}
                onChange={e => { const v = parseFloat(e.target.value); setMergeGap(v); scheduleMergePreview(v) }} />
              <input type="number" min="0" max="10" step="0.1" value={mergeGap} disabled={compressMode}
                style={{ width: 56, background: 'var(--surf)', color: 'var(--text)', border: '1px solid var(--bord2)', borderRadius: 4, padding: '3px 6px', fontSize: 12, opacity: compressMode ? 0.35 : 1 }}
                onChange={e => { const v = Math.max(0, Math.min(10, parseFloat(e.target.value) || 0)); setMergeGap(v); scheduleMergePreview(v) }} />
              <span style={{ fontSize: 11, color: 'var(--dim)', opacity: compressMode ? 0.35 : 1 }}>s</span>
              <button className="btn btn-ghost btn-sm" style={{ fontSize: 11, ...(compressMode ? { color: 'var(--accent)', borderColor: 'var(--accent)' } : {}) }}
                onClick={() => { const next = !compressMode; setCompressMode(next); scheduleMergePreview(next ? COMPRESS_GAP : mergeGap) }}>
                ⇒ {compressMode ? 'Compress (on)' : 'Compress'}
              </button>
              {mergeCounts && (
                <span style={{ fontSize: 11, color: 'var(--dim)' }}>
                  {mergeCounts.original} → <strong style={{ color: 'var(--text)' }}>{mergeCounts.merged}</strong>
                  {mergeCounts.original > 0 && (
                    <span style={{ color: mergeCounts.original > mergeCounts.merged ? 'var(--pos)' : 'var(--dim)', marginLeft: 4 }}>
                      ({mergeCounts.original > mergeCounts.merged ? '-' : '+'}{Math.abs(mergeCounts.original - mergeCounts.merged)}, {Math.round(Math.abs(mergeCounts.original - mergeCounts.merged) / mergeCounts.original * 100)}% reduction)
                    </span>
                  )}
                </span>
              )}
              <button className="btn btn-ghost btn-sm" style={{ fontSize: 11, ...(mergePreviewActive ? { color: 'var(--accent)' } : {}) }}
                onClick={() => {
                  const next = !mergePreviewActive
                  setMergePreviewActive(next)
                  if (next) fetchMergePreview(compressMode ? COMPRESS_GAP : mergeGap)
                  else setPreviewTurns(null)
                }}>
                {mergePreviewActive ? 'Hide preview' : 'Show preview'}
              </button>
              <button className="btn btn-primary btn-sm" style={{ fontSize: 11 }} onClick={applyRemerge}>Apply</button>
              <button className="btn btn-ghost btn-sm" style={{ fontSize: 11 }} onClick={() => setShowMergePanel(false)}>✕</button>
            </div>
          </div>
        )}

        {/* Turns list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '6px 10px' }}>
          {shownTurns.map((turn, i) => {
            const isPreview  = !!previewTurns
            const effId      = isPreview ? turn.speaker_id : turn.effective_speaker
            const color      = spkColor(effId, all_ephemeral_ids)
            const effSp      = speakerLookup[effId]
            const effName    = effSp?.display_name || effId
            const sentColor  = turn.sentiment === 'positive' ? 'var(--pos)' : turn.sentiment === 'negative' ? 'var(--neg)' : 'var(--dim)'
            const mergedCount = turn.merged_count || 1
            const isPlaying  = playingIdx === turn.index
            const origTurns  = reviewData.session.original_turns || []

            return (
              <div key={isPreview ? i : turn.index}
                id={isPreview ? undefined : `rv-turn-${turn.index}`}
                tabIndex={isPreview ? undefined : 0}
                onKeyDown={isPreview ? undefined : e => {
                  if (e.key === 'r' || e.key === 'R') { e.preventDefault(); setReassignOpen(v => v === turn.index ? null : turn.index) }
                  if (e.key === 'd' || e.key === 'D') { e.preventDefault(); deleteTurn(turn.index) }
                  if (e.key === 'f' || e.key === 'F') { e.preventDefault(); setFrameModal({ idx: turn.index }); setFrameStatus('loading') }
                }}
                style={{
                  borderLeft: `3px solid ${turn.deleted ? 'var(--bord2)' : (turn.overridden && !isPreview ? color : color + '55')}`,
                  padding: '7px 10px', borderRadius: '0 6px 6px 0', marginBottom: 4,
                  background: 'var(--surf2)', opacity: turn.deleted ? 0.5 : 1,
                  outline: 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    {/* Turn meta */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3, flexWrap: 'wrap' }}>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--dimmer)' }}>
                        [{fmtRev(turn.start)} → {fmtRev(turn.end)}]
                      </span>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                      {!isPreview && turn.overridden
                        ? <><span style={{ fontSize: 11, color: 'var(--dimmer)', textDecoration: 'line-through', fontFamily: 'var(--mono)' }}>{turn.original_speaker}</span>
                            <strong style={{ fontSize: 12, color }}>{effName}</strong></>
                        : <span style={{ fontSize: 12, color, fontWeight: 500 }}>{effName}</span>
                      }
                      {turn._compressed && <span style={{ fontSize: 10, color: 'var(--dimmer)' }}>(×{turn._runLen})</span>}
                      {turn.deleted && <span style={{ fontSize: 10, background: '#2a1010', color: 'var(--neg)', padding: '1px 5px', borderRadius: 3 }}>DELETED</span>}
                      {!turn.deleted && turn.sentiment && <span style={{ fontSize: 10, color: sentColor }}>{turn.sentiment}</span>}
                      {!turn.deleted && !isPreview && mergedCount > 1 && (
                        <button style={{ fontSize: 10, background: 'var(--surf)', border: '1px solid var(--bord2)', color: 'var(--dim)', borderRadius: 3, padding: '1px 5px', cursor: 'pointer' }}
                          onClick={() => setSubTurnsOpen(o => ({ ...o, [turn.index]: !o[turn.index] }))}>
                          merged {mergedCount} {subTurnsOpen[turn.index] ? '▴' : '▾'}
                        </button>
                      )}
                      {!isPreview && turn.overridden && <span style={{ fontSize: 10, color: 'var(--dimmer)' }}>overridden</span>}
                    </div>
                    {/* Transcript */}
                    <div style={{ fontSize: 13, color: turn.transcript ? 'var(--text)' : 'var(--dimmer)', fontStyle: turn.transcript ? 'normal' : 'italic', lineHeight: 1.5 }}>
                      {turn.transcript || '(no transcript)'}
                    </div>
                    {/* Sub-turns drawer */}
                    {!turn.deleted && !isPreview && mergedCount > 1 && subTurnsOpen[turn.index] && origTurns.length > 0 && (
                      <div style={{ marginTop: 6, borderLeft: '2px solid var(--bord)', paddingLeft: 8 }}>
                        {origTurns
                          .filter(ot => ot.speaker_id === turn.original_speaker && ot.start >= turn.start - 0.01 && ot.end <= turn.end + 0.01)
                          .map((st, si) => (
                            <div key={si} style={{ fontSize: 11, color: 'var(--dim)', padding: '2px 0' }}>
                              <span style={{ fontFamily: 'var(--mono)', color: 'var(--dimmer)', marginRight: 6 }}>{fmtRev(st.start)}→{fmtRev(st.end)}</span>
                              {(st.transcript || '').slice(0, 80) || <em style={{ color: 'var(--dimmer)' }}>(silent)</em>}
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                  {/* Turn actions */}
                  {!isPreview && (
                    <div style={{ display: 'flex', gap: 2, flexShrink: 0, alignItems: 'center', marginTop: 1 }}>
                      {turn.deleted ? (
                        <button title="Restore" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--dim)', fontSize: 14, padding: '2px 5px' }}
                          onClick={() => restoreTurn(turn.index)}>↩</button>
                      ) : (
                        <>
                          <div style={{ position: 'relative' }}>
                            <button title="Reassign (R)" style={{ background: 'none', border: 'none', cursor: 'pointer', color: reassignOpen === turn.index ? 'var(--accent)' : 'var(--dim)', fontSize: 14, padding: '2px 5px' }}
                              onClick={() => setReassignOpen(v => v === turn.index ? null : turn.index)}>↺</button>
                            {reassignOpen === turn.index && (
                              <select autoFocus
                                style={{ position: 'absolute', right: 0, top: '100%', zIndex: 100, background: 'var(--surf)', border: '1px solid var(--bord2)', borderRadius: 6, fontSize: 12, color: 'var(--text)', padding: 4, minWidth: 180, boxShadow: '0 4px 16px rgba(0,0,0,0.4)' }}
                                defaultValue={turn.effective_speaker}
                                onChange={e => submitReassign(turn.index, e.target.value)}
                                onBlur={() => setReassignOpen(null)}>
                                {all_ephemeral_ids.map(id => (
                                  <option key={id} value={id}>{speakerLookup[id]?.display_name || id}</option>
                                ))}
                              </select>
                            )}
                          </div>
                          <button title="Play/Stop" style={{ background: 'none', border: 'none', cursor: 'pointer', color: isPlaying ? 'var(--accent)' : 'var(--dim)', fontSize: 13, padding: '2px 5px' }}
                            onClick={() => togglePlay(turn.index, turn.start, turn.end)}>
                            {isPlaying ? '⏹' : '▶'}
                          </button>
                          <button title="View frame (F)" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--dim)', fontSize: 13, padding: '2px 5px' }}
                            onClick={() => { setFrameModal({ idx: turn.index }); setFrameStatus('loading') }}>📷</button>
                          <button title="Delete (D)" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--neg)', fontSize: 13, padding: '2px 5px' }}
                            onClick={() => deleteTurn(turn.index)}>🗑</button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Confirm speaker modal ── */}
      {confirmModal && (
        <div className="modal-overlay" onClick={() => setConfirmModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-title">Confirm Speaker — <span style={{ color: 'var(--accent)', fontFamily: 'var(--mono)', fontSize: 14 }}>{confirmModal.ephId}</span></div>
            {[['Name', 'name'], ['Affiliation', 'affiliation'], ['Role / title', 'role'], ['Notes', 'notes']].map(([label, key]) => (
              <div key={key} className="form-group">
                <label className="form-label">{label}</label>
                <input className="form-input" autoFocus={key === 'name'} value={confirmForm[key]}
                  onChange={e => setConfirmForm(f => ({ ...f, [key]: e.target.value }))} />
              </div>
            ))}
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost" onClick={() => setConfirmModal(null)}>Cancel</button>
              <button className="btn btn-primary" onClick={submitConfirm}><CheckCircle2 size={14} /> Confirm</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Search / link modal ── */}
      {searchModal && (
        <div className="modal-overlay" onClick={() => setSearchModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-title">Find Existing Speaker</div>
            <div className="form-group">
              <label className="form-label">Search by name</label>
              <input className="form-input" autoFocus placeholder="Type to search…" value={searchQuery}
                onChange={e => doSearch(e.target.value)} />
            </div>
            <div style={{ maxHeight: 260, overflowY: 'auto' }}>
              {searchResults.map(r => (
                <div key={r.catalogue_id}
                  style={{ padding: '8px 10px', borderRadius: 6, cursor: 'pointer', marginBottom: 4, background: 'var(--surf2)', border: '1px solid var(--bord)' }}
                  onClick={() => selectAndLink(r.catalogue_id)}>
                  <strong style={{ fontSize: 13 }}>{r.display_name || r.catalogue_id}</strong>
                  {r.affiliation && <span style={{ fontSize: 11, color: 'var(--dim)', marginLeft: 8 }}>{r.affiliation}</span>}
                  <span style={{ fontSize: 10, color: 'var(--accent)', marginLeft: 8, fontFamily: 'var(--mono)' }}>{r.catalogue_id}</span>
                </div>
              ))}
              {searchQuery && searchResults.length === 0 && (
                <div style={{ color: 'var(--dim)', fontSize: 12, padding: 8 }}>No results found.</div>
              )}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
              <button className="btn btn-ghost" onClick={() => setSearchModal(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Frame viewer modal ── */}
      {frameModal && (() => {
        const allSorted = reviewData.speakers.flatMap(s => s.turns).sort((a, b) => a.index - b.index)
        const pos  = allSorted.findIndex(t => t.index === frameModal.idx)
        const turn = allSorted[pos]
        return (
          <div className="modal-overlay" onClick={() => setFrameModal(null)}>
            <div style={{ background: 'var(--surf)', border: '1px solid var(--bord2)', borderRadius: 12, width: '90vw', maxWidth: 960, overflow: 'hidden' }}
              onClick={e => e.stopPropagation()}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid var(--bord)' }}>
                <span style={{ fontSize: 12, color: 'var(--dim)', fontFamily: 'var(--mono)' }}>
                  Turn {frameModal.idx}{turn && ` · ${fmtRev(turn.start)} → ${fmtRev(turn.end)} · ${turn.original_speaker}`}
                </span>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <span style={{ fontSize: 11, color: 'var(--dimmer)' }}>← → to step · Esc to close</span>
                  <button className="btn btn-ghost btn-sm" onClick={() => setFrameModal(null)}>✕</button>
                </div>
              </div>
              <div style={{ position: 'relative', minHeight: 200, background: '#0a0d14', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {frameStatus === 'loading' && <div style={{ color: 'var(--dim)', fontSize: 13 }}>Loading frame…</div>}
                {frameStatus === 'error' && (
                  <div style={{ color: 'var(--neg)', fontSize: 13, textAlign: 'center', padding: 40 }}>
                    No video frame available.<br /><span style={{ fontSize: 11, color: 'var(--dimmer)' }}>(Audio-only source or missing video file)</span>
                  </div>
                )}
                <img key={frameModal.idx}
                  src={`/api/review/${sessionId}/turns/${frameModal.idx}/frame?_=${Date.now()}`}
                  alt="Turn frame"
                  style={{ maxWidth: '100%', maxHeight: '72vh', objectFit: 'contain', display: frameStatus === 'loaded' ? 'block' : 'none' }}
                  onLoad={() => setFrameStatus('loaded')}
                  onError={() => setFrameStatus('error')} />
              </div>
              <div style={{ display: 'flex', gap: 8, padding: '8px 12px', borderTop: '1px solid var(--bord)' }}>
                <button className="btn btn-ghost btn-sm" disabled={pos <= 0} onClick={() => doStepFrame(-1)}>← Prev</button>
                <button className="btn btn-ghost btn-sm" disabled={pos >= allSorted.length - 1} onClick={() => doStepFrame(1)}>Next →</button>
                <div style={{ flex: 1 }} />
                {turn && !turn.deleted && (
                  <button className="btn btn-primary btn-sm"
                    onClick={() => { setFrameModal(null); setReassignOpen(turn.index) }}>↺ Reassign this turn</button>
                )}
              </div>
            </div>
          </div>
        )
      })()}
    </div>
  )
}

// ─── SessionsView ─────────────────────────────────────────────────────────────
function SessionsView({ onViewSession }) {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [editModal, setEditModal] = useState(null) // session object being edited
  const [editForm, setEditForm] = useState({ source_name: '', broadcast_date: '', broadcast_channel: '' })
  const [saving, setSaving] = useState(false)
  const [editErr, setEditErr] = useState('')

  const refresh = () =>
    apiFetch('/sessions').then(d => { setSessions(d); setLoading(false) }).catch(() => setLoading(false))

  useEffect(() => { refresh() }, [])

  function openEdit(e, s) {
    e.stopPropagation()
    setEditForm({ source_name: s.source_name || '', broadcast_date: s.broadcast_date || '', broadcast_channel: s.broadcast_channel || '' })
    setEditErr('')
    setEditModal(s)
  }

  async function saveEdit() {
    setSaving(true); setEditErr('')
    try {
      await apiFetch(`/sessions/${editModal.session_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editForm),
      })
      setEditModal(null)
      refresh()
    } catch (e) {
      setEditErr(e.message)
    } finally {
      setSaving(false)
    }
  }

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}><Loader2 size={20} className="spin" /></div>

  return (
    <div>
      <div className="section-title">Sessions</div>
      <div className="section-sub">All processed recordings with diarization results.</div>

      {sessions.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '40px', color: 'var(--dim)' }}>
          No sessions yet. Process a file to create a session.
        </div>
      ) : (
        <div className="card" style={{ padding: 0 }}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Programme</th><th>Channel</th><th>Broadcast date</th><th>Duration</th><th>Speakers</th><th></th></tr>
              </thead>
              <tbody>
                {sessions.map(s => (
                  <tr key={s.session_id} style={{ cursor: 'pointer' }} onClick={() => onViewSession(s.session_id)}>
                    <td>{s.source_name || <span style={{ color: 'var(--dim)' }}>—</span>}</td>
                    <td>{s.broadcast_channel || <span style={{ color: 'var(--dim)' }}>—</span>}</td>
                    <td>{s.broadcast_date || <span style={{ color: 'var(--dim)' }}>—</span>}</td>
                    <td>{fmtTime(s.total_duration)}</td>
                    <td>{s.num_speakers}</td>
                    <td style={{ whiteSpace: 'nowrap' }}>
                      <button className="btn btn-ghost btn-sm" style={{ marginRight: 4 }}
                        onClick={e => openEdit(e, s)}><Edit3 size={13} /> Edit</button>
                      <button className="btn btn-ghost btn-sm"
                        onClick={e => { e.stopPropagation(); onViewSession(s.session_id) }}><Eye size={13} /> View</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {editModal && (
        <div className="modal-overlay" onClick={() => setEditModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 460 }}>
            <div className="modal-title">Edit session info</div>
            <div style={{ fontSize: 11, color: 'var(--dim)', fontFamily: 'var(--mono)', marginBottom: 16 }}>{editModal.session_id}</div>
            <div className="form-group">
              <label className="form-label">Programme name</label>
              <input className="input" value={editForm.source_name}
                onChange={e => setEditForm(f => ({ ...f, source_name: e.target.value }))} />
            </div>
            <div className="form-group">
              <label className="form-label">Broadcast channel</label>
              <input className="input" value={editForm.broadcast_channel}
                onChange={e => setEditForm(f => ({ ...f, broadcast_channel: e.target.value }))} />
            </div>
            <div className="form-group">
              <label className="form-label">Date of broadcast</label>
              <input className="input" type="date" value={editForm.broadcast_date}
                onChange={e => setEditForm(f => ({ ...f, broadcast_date: e.target.value }))} />
            </div>
            {editErr && <div className="notice notice-error"><AlertCircle size={14} />{editErr}</div>}
            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
              <button className="btn btn-primary btn-sm" onClick={saveEdit} disabled={saving}>
                {saving ? <Loader2 size={13} className="spin" /> : <CheckCircle2 size={13} />} Save
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => setEditModal(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── SpeakersView ─────────────────────────────────────────────────────────────
function SpeakersView({ onRefresh }) {
  const [speakers, setSpeakers] = useState([])
  const [search, setSearch] = useState('')
  const [affFilter, setAffFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [addModal, setAddModal] = useState(false)
  const [editModal, setEditModal] = useState(null)
  const [form, setForm] = useState({ name: '', affiliation: '', role: '', notes: '' })
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')
  const [detail, setDetail] = useState(null)

  const refresh = useCallback(async () => {
    const params = new URLSearchParams()
    if (search) params.set('search', search)
    if (affFilter) params.set('affiliation', affFilter)
    try {
      setSpeakers(await apiFetch('/speakers?' + params))
    } catch {}
    setLoading(false)
  }, [search, affFilter])

  useEffect(() => { refresh() }, [refresh])

  async function saveNew() {
    setSaving(true); setErr('')
    try {
      const fd = new FormData()
      Object.entries(form).forEach(([k, v]) => v && fd.append(k === 'name' ? 'name' : k, v))
      await apiFetch('/speakers', { method: 'POST', body: fd })
      setAddModal(false); setForm({ name: '', affiliation: '', role: '', notes: '' })
      refresh(); onRefresh?.()
    } catch (e) { setErr(e.message) }
    setSaving(false)
  }

  async function saveEdit() {
    setSaving(true); setErr('')
    try {
      const fd = new FormData()
      Object.entries(form).forEach(([k, v]) => fd.append(k, v || ''))
      await apiFetch(`/speakers/${editModal.catalogue_id}`, { method: 'PUT', body: fd })
      setEditModal(null); setForm({ name: '', affiliation: '', role: '', notes: '' })
      refresh()
    } catch (e) { setErr(e.message) }
    setSaving(false)
  }

  function openEdit(spk) {
    setEditModal(spk)
    setForm({ name: spk.display_name || '', affiliation: spk.affiliation || '', role: spk.role || '', notes: spk.notes || '' })
    setErr('')
  }

  async function loadDetail(id) {
    try {
      setDetail(await apiFetch(`/speakers/${id}/appearances`))
    } catch {}
  }

  if (detail) return (
    <div>
      <button className="btn btn-ghost btn-sm" style={{ marginBottom: 16 }} onClick={() => setDetail(null)}><ArrowLeft size={14} /> Back</button>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 20 }}>
        <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'var(--surf3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18, fontWeight: 700, color: 'var(--accent)' }}>
          {(detail.profile.display_name || '?').charAt(0).toUpperCase()}
        </div>
        <div>
          <div className="section-title" style={{ marginBottom: 0 }}>{detail.profile.display_name || '(unnamed)'}</div>
          <div style={{ fontSize: 12, color: 'var(--dim)' }}>
            {[detail.profile.role, detail.profile.affiliation].filter(Boolean).join(' · ')}
          </div>
        </div>
      </div>
      <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(3,1fr)', marginBottom: 20 }}>
        <div className="stat"><div className="stat-label">Appearances</div><div className="stat-val">{detail.profile.total_appearances}</div></div>
        <div className="stat"><div className="stat-label">Total airtime</div><div className="stat-val">{fmtTime(detail.profile.total_speaking_time)}</div></div>
        <div className="stat"><div className="stat-label">Last seen</div><div className="stat-val" style={{ fontSize: 16 }}>{fmtDate(detail.profile.last_seen).split(',')[0]}</div></div>
      </div>
      <div className="card" style={{ padding: 0 }}>
        <table>
          <thead><tr><th>Date</th><th>Source</th><th>Speaking time</th><th>Turns</th><th>Avg sentiment</th></tr></thead>
          <tbody>
            {detail.appearances.map((a, i) => (
              <tr key={i}>
                <td><span className="mono" style={{ fontSize: 11 }}>{fmtDate(a.appeared_at)}</span></td>
                <td>{a.source_name || '—'}</td>
                <td>{fmtTime(a.speaking_time)}</td>
                <td>{a.turn_count}</td>
                <td>
                  {a.avg_sentiment != null
                    ? <span style={{ color: sentimentColor(a.avg_sentiment), display: 'flex', alignItems: 'center', gap: 3 }}>
                        {sentimentIcon(a.avg_sentiment)} {a.avg_sentiment > 0 ? '+' : ''}{a.avg_sentiment?.toFixed(3)}
                      </span>
                    : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <div className="section-title">Speakers</div>
          <div className="section-sub">Named speaker catalogue with cross-session history.</div>
        </div>
        <button className="btn btn-primary btn-sm" onClick={() => { setAddModal(true); setForm({ name: '', affiliation: '', role: '', notes: '' }); setErr('') }}>
          <Plus size={14} /> Add Speaker
        </button>
      </div>

      {/* Search bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--dim)' }} />
          <input className="form-input" style={{ paddingLeft: 32 }} placeholder="Search by name…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <input className="form-input" style={{ width: 200 }} placeholder="Filter by affiliation…" value={affFilter} onChange={e => setAffFilter(e.target.value)} />
        <button className="btn btn-ghost btn-sm" onClick={refresh}><RefreshCw size={13} /></button>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrap">
          <table>
            <thead><tr><th>ID</th><th>Name</th><th>Affiliation</th><th>Role</th><th>Appearances</th><th>Airtime</th><th></th></tr></thead>
            <tbody>
              {speakers.map(s => (
                <tr key={s.catalogue_id}>
                  <td><span className="mono" style={{ color: 'var(--accent)' }}>{s.catalogue_id}</span></td>
                  <td style={{ fontWeight: 500 }}>{s.display_name || <span style={{ color: 'var(--dim)', fontStyle: 'italic' }}>unnamed</span>}</td>
                  <td style={{ color: 'var(--dim)' }}>{s.affiliation || '—'}</td>
                  <td style={{ color: 'var(--dim)' }}>{s.role || '—'}</td>
                  <td>{s.total_appearances}</td>
                  <td>{fmtTime(s.total_speaking_time)}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="btn btn-ghost btn-sm" onClick={() => loadDetail(s.catalogue_id)}><Eye size={12} /></button>
                      <button className="btn btn-ghost btn-sm" onClick={() => openEdit(s)}><Edit3 size={12} /></button>
                    </div>
                  </td>
                </tr>
              ))}
              {speakers.length === 0 && !loading && (
                <tr><td colSpan={7} style={{ textAlign: 'center', color: 'var(--dim)', padding: 32 }}>No speakers found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add / Edit modal */}
      {(addModal || editModal) && (
        <div className="modal-overlay" onClick={() => { setAddModal(false); setEditModal(null) }}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-title">{addModal ? 'Register New Speaker' : `Edit ${editModal.display_name || editModal.catalogue_id}`}</div>
            {['name','affiliation','role','notes'].map(k => (
              <div className="form-group" key={k}>
                <label className="form-label">{k}</label>
                <input className="form-input" value={form[k]} onChange={e => setForm(f => ({ ...f, [k]: e.target.value }))} placeholder={k === 'name' ? 'Full name' : k === 'affiliation' ? 'e.g. BBC News' : k === 'role' ? 'e.g. Political Correspondent' : 'Any notes…'} />
              </div>
            ))}
            {err && <div className="notice notice-error" style={{ marginBottom: 12 }}><AlertCircle size={14} />{err}</div>}
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost" onClick={() => { setAddModal(false); setEditModal(null) }}>Cancel</button>
              <button className="btn btn-primary" onClick={addModal ? saveNew : saveEdit} disabled={saving}>
                {saving ? <><Loader2 size={14} className="spin" /> Saving…</> : addModal ? <><Plus size={14} /> Register</> : <><CheckCircle2 size={14} /> Save</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Dashboard ────────────────────────────────────────────────────────────────
function Dashboard({ onNavigate }) {
  const [stats, setStats] = useState({ sessions: 0, speakers: 0, running: 0 })
  const [recentSessions, setRecentSessions] = useState([])
  const [topSpeakers, setTopSpeakers] = useState([])
  const [health, setHealth] = useState(null)

  useEffect(() => {
    Promise.all([
      apiFetch('/health'),
      apiFetch('/sessions'),
      apiFetch('/speakers'),
      apiFetch('/jobs'),
    ]).then(([h, sessions, spk, jobs]) => {
      setHealth(h)
      setRecentSessions(sessions.slice(0, 5))
      setTopSpeakers(spk.slice(0, 5))
      setStats({
        sessions: sessions.length,
        speakers: spk.length,
        running: jobs.filter(j => j.status === 'running').length,
      })
    }).catch(() => {})
  }, [])

  return (
    <div>
      <div className="section-title">Overview</div>
      <div className="section-sub">News diarization pipeline — speaker tracking dashboard.</div>

      {health && !health.hf_token_configured && (
        <div className="notice notice-error" style={{ marginBottom: 20 }}>
          <AlertCircle size={16} style={{ flexShrink: 0 }} />
          <div><strong>HF_TOKEN not configured.</strong> Add it to your .env file and restart the server. Processing will fail without it.</div>
        </div>
      )}

      <div className="stat-grid">
        <div className="stat">
          <div className="stat-label">Sessions</div>
          <div className="stat-val">{stats.sessions}</div>
          <div className="stat-sub">recorded</div>
        </div>
        <div className="stat">
          <div className="stat-label">Speakers</div>
          <div className="stat-val">{stats.speakers}</div>
          <div className="stat-sub">in catalogue</div>
        </div>
        <div className="stat">
          <div className="stat-label">Active jobs</div>
          <div className="stat-val" style={{ color: stats.running > 0 ? 'var(--accent)' : undefined }}>{stats.running}</div>
          <div className="stat-sub">{stats.running > 0 ? <span className="pulse" style={{ color: 'var(--accent)' }}>● processing</span> : 'idle'}</div>
        </div>
        <div className="stat">
          <div className="stat-label">API</div>
          <div className="stat-val" style={{ fontSize: 16, color: health ? 'var(--pos)' : 'var(--neg)' }}>
            {health ? '● Online' : '○ Offline'}
          </div>
          <div className="stat-sub">localhost:8000</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div className="card-title"><Clock size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Recent sessions</div>
          {recentSessions.length === 0
            ? <div style={{ color: 'var(--dim)', fontSize: 13 }}>No sessions yet.</div>
            : recentSessions.map(s => (
              <div key={s.session_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: '1px solid var(--bord)', cursor: 'pointer' }}
                onClick={() => onNavigate('sessions', s.session_id)}>
                <div>
                  <div style={{ fontWeight: 500, fontSize: 13 }}>{s.source_name || s.session_id}</div>
                  <div style={{ fontSize: 11, color: 'var(--dim)' }}>{fmtDate(s.processed_at)} · {s.num_speakers} speakers</div>
                </div>
                <ChevronRight size={14} style={{ color: 'var(--dimmer)', alignSelf: 'center' }} />
              </div>
          ))}
          <button className="link-btn" style={{ marginTop: 10 }} onClick={() => onNavigate('sessions')}>View all →</button>
        </div>

        <div className="card">
          <div className="card-title"><Users size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Top speakers</div>
          {topSpeakers.length === 0
            ? <div style={{ color: 'var(--dim)', fontSize: 13 }}>No speakers catalogued yet.</div>
            : topSpeakers.map((s, i) => (
              <div key={s.catalogue_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: '1px solid var(--bord)' }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--accent)', minWidth: 52 }}>{s.catalogue_id}</span>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 500 }}>{s.display_name || '(unnamed)'}</div>
                    <div style={{ fontSize: 11, color: 'var(--dim)' }}>{s.affiliation || '—'}</div>
                  </div>
                </div>
                <div style={{ textAlign: 'right', fontSize: 12, color: 'var(--dim)' }}>
                  <div style={{ fontWeight: 500, color: 'var(--text)' }}>{s.total_appearances}×</div>
                  <div>{fmtTime(s.total_speaking_time)}</div>
                </div>
              </div>
          ))}
          <button className="link-btn" style={{ marginTop: 10 }} onClick={() => onNavigate('speakers')}>Manage speakers →</button>
        </div>
      </div>
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState('dashboard')
  const [selectedSession, setSelectedSession] = useState(null)
  const [activeJobs, setActiveJobs] = useState(0)

  // Poll for active jobs count
  useEffect(() => {
    const poll = async () => {
      try {
        const jobs = await apiFetch('/jobs')
        setActiveJobs(jobs.filter(j => j.status === 'running' || j.status === 'queued').length)
      } catch {}
    }
    poll()
    const iv = setInterval(poll, 5000)
    return () => clearInterval(iv)
  }, [])

  function navigate(v, sessionId = null) {
    setView(v)
    if (sessionId) setSelectedSession(sessionId)
  }

  function handleJobSubmitted(jobId) {
    setView('jobs')
  }

  function handleViewSession(sessionId) {
    setSelectedSession(sessionId)
    setView('session-detail')
  }

  function handleReviewSession(sessionId) {
    setSelectedSession(sessionId)
    setView('session-review')
  }

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <BarChart2 size={15} /> },
    { id: 'new-job', label: 'New Job', icon: <Radio size={15} /> },
    { id: 'jobs', label: 'Jobs', icon: <Activity size={15} />, badge: activeJobs },
    { id: 'sessions', label: 'Sessions', icon: <List size={15} /> },
    { id: 'speakers', label: 'Speakers', icon: <Users size={15} /> },
  ]

  const titles = { dashboard: 'Dashboard', 'new-job': 'New Job', jobs: 'Jobs', sessions: 'Sessions', 'session-detail': 'Session Detail', 'session-review': 'Review', speakers: 'Speakers' }

  return (
    <>
      <style>{css}</style>
      <div className="shell">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="logo">
            <div className="logo-mark"><Mic size={15} color="#080c14" /></div>
            <span className="logo-text">Diarizer</span>
          </div>
          <nav className="nav">
            <div className="nav-label">Navigation</div>
            {navItems.map(item => (
              <button key={item.id} className={`nav-item ${view === item.id || ((view === 'session-detail' || view === 'session-review') && item.id === 'sessions') ? 'active' : ''}`}
                onClick={() => { setView(item.id); if (item.id !== 'session-detail') setSelectedSession(null) }}>
                {item.icon}
                <span style={{ flex: 1 }}>{item.label}</span>
                {item.badge > 0 && (
                  <span style={{ background: 'var(--accent)', color: '#080c14', borderRadius: 10, padding: '1px 6px', fontSize: 10, fontWeight: 700 }}>
                    {item.badge}
                  </span>
                )}
              </button>
            ))}
          </nav>
          <div style={{ padding: '0 10px 14px', fontSize: 11, color: 'var(--dimmer)' }}>
            <div style={{ padding: '8px 10px', background: 'var(--surf2)', borderRadius: 6, border: '1px solid var(--bord)' }}>
              <span style={{ fontFamily: 'var(--mono)' }}>api</span> localhost:8000
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="main">
          <div className="topbar">
            <span style={{ fontFamily: 'var(--head)', fontWeight: 600, fontSize: 15 }}>{titles[view]}</span>
            {(view === 'session-detail' || view === 'session-review') && selectedSession && (
              <>
                <span style={{ color: 'var(--dimmer)' }}>/</span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--dim)' }}>{selectedSession}</span>
              </>
            )}
          </div>

          <div className="content" style={view === 'session-review' ? { padding: 0 } : {}}>
            {view === 'dashboard' && <Dashboard onNavigate={navigate} />}
            {view === 'new-job' && <NewJobView onJobSubmitted={handleJobSubmitted} />}
            {view === 'jobs' && <JobsView onViewSession={handleViewSession} />}
            {view === 'sessions' && <SessionsView onViewSession={handleViewSession} />}
            {view === 'session-detail' && selectedSession && (
              <SessionDetail
                sessionId={selectedSession}
                onBack={() => setView('sessions')}
                onReview={handleReviewSession}
                onRefreshSpeakers={() => {}}
              />
            )}
            {view === 'session-review' && selectedSession && (
              <ReviewView
                sessionId={selectedSession}
                onBack={() => setView('session-detail')}
              />
            )}
            {view === 'speakers' && <SpeakersView />}
          </div>
        </main>
      </div>
    </>
  )
}
