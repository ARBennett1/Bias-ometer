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

// ─── SessionDetail ────────────────────────────────────────────────────────────
function SessionDetail({ sessionId, onBack, speakers, onRefreshSpeakers }) {
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
        {['timeline','turns'].map(t => (
          <button key={t} className={`tab ${activeTab === t ? 'active' : ''}`} onClick={() => setActiveTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
        {filterSpk && (
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

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <BarChart2 size={15} /> },
    { id: 'new-job', label: 'New Job', icon: <Radio size={15} /> },
    { id: 'jobs', label: 'Jobs', icon: <Activity size={15} />, badge: activeJobs },
    { id: 'sessions', label: 'Sessions', icon: <List size={15} /> },
    { id: 'speakers', label: 'Speakers', icon: <Users size={15} /> },
  ]

  const titles = { dashboard: 'Dashboard', 'new-job': 'New Job', jobs: 'Jobs', sessions: 'Sessions', 'session-detail': 'Session Detail', speakers: 'Speakers' }

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
              <button key={item.id} className={`nav-item ${view === item.id || (view === 'session-detail' && item.id === 'sessions') ? 'active' : ''}`}
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
            {view === 'session-detail' && selectedSession && (
              <>
                <span style={{ color: 'var(--dimmer)' }}>/</span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--dim)' }}>{selectedSession}</span>
              </>
            )}
          </div>

          <div className="content">
            {view === 'dashboard' && <Dashboard onNavigate={navigate} />}
            {view === 'new-job' && <NewJobView onJobSubmitted={handleJobSubmitted} />}
            {view === 'jobs' && <JobsView onViewSession={handleViewSession} />}
            {view === 'sessions' && <SessionsView onViewSession={handleViewSession} />}
            {view === 'session-detail' && selectedSession && (
              <SessionDetail
                sessionId={selectedSession}
                onBack={() => setView('sessions')}
                onRefreshSpeakers={() => {}}
              />
            )}
            {view === 'speakers' && <SpeakersView />}
          </div>
        </main>
      </div>
    </>
  )
}
