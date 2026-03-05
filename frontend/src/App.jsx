import { useState } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'

const STACK = ['PyTorch', 'Gemini', 'LangChain', 'AutoGen', 'RDKit', 'VAE', 'GCN', 'ZINC', 'FAISS']

const PROP_LABELS = {
  MolecularWeight: 'Mol. Weight (Da)',
  logP: 'LogP',
  QED: 'QED',
  TPSA: 'TPSA (Å²)',
  NumHBondDonors: 'H-Bond Donors',
  NumHBondAcceptors: 'H-Bond Acceptors',
  NumRotatableBonds: 'Rotatable Bonds',
  NumAromaticRings: 'Aromatic Rings',
  FractionCSP3: 'Fsp3',
  LipinskiViolations: 'Lipinski Violations',
  SyntheticAccessibility: 'SA Score',
  EstimatedLogS: 'Est. LogS',
}

function traceClass(line) {
  if (line.startsWith('[CoT]')) return 'cot'
  if (line.includes('✅')) return 'valid'
  if (line.includes('❌') || line.includes('failed')) return 'error'
  if (line.includes('[RAG]')) return 'rag'
  return ''
}

/* ── Result card ─────────────────────────────────────────── */
function ResultCard({ result, index }) {
  const { smiles, is_valid, validation_message, properties, agent_trace } = result
  return (
    <div className="result-card" style={{ animationDelay: `${index * 0.07}s` }}>
      <div className="result-card-header">
        <span className="result-index">MOL-{String(index + 1).padStart(3, '0')}</span>
        <span className="smiles-badge" title={smiles}>{smiles}</span>
        <span className={`valid-badge ${is_valid ? 'valid' : 'invalid'}`}>
          {is_valid ? '✓ Valid' : '✗ Invalid'}
        </span>
      </div>

      <div className="result-card-body">
        {/* Property table */}
        <div>
          <p style={{ fontSize: '0.75rem', color: 'var(--color-muted)', marginBottom: 8 }}>
            Physicochemical Properties
          </p>
          {properties ? (
            <table className="prop-table">
              <tbody>
                {Object.entries(PROP_LABELS).map(([key, label]) =>
                  properties[key] !== undefined && properties[key] !== null ? (
                    <tr key={key}>
                      <td>{label}</td>
                      <td>
                        {typeof properties[key] === 'number'
                          ? properties[key].toFixed ? properties[key].toFixed(3) : properties[key]
                          : properties[key]}
                      </td>
                    </tr>
                  ) : null
                )}
              </tbody>
            </table>
          ) : (
            <p style={{ fontSize: '0.8rem', color: 'var(--color-error)' }}>
              {validation_message}
            </p>
          )}
        </div>

        {/* Molecular info  */}
        <div>
          <p style={{ fontSize: '0.75rem', color: 'var(--color-muted)', marginBottom: 8 }}>
            Molecular Formula &amp; Details
          </p>
          {properties && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <code style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '1.1rem',
                color: 'var(--color-accent)',
                background: 'var(--color-surface-2)',
                padding: '6px 10px',
                borderRadius: 6,
              }}>
                {properties.MolecularFormula}
              </code>
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8,
              }}>
                {[
                  ['QED', properties.QED?.toFixed(3), properties.QED >= 0.6 ? 'var(--color-success)' : 'var(--color-warn)'],
                  ['LogP', properties.logP?.toFixed(2), (properties.logP >= 1 && properties.logP <= 5) ? 'var(--color-success)' : 'var(--color-warn)'],
                  ['SA Score', properties.SyntheticAccessibility?.toFixed(2), properties.SyntheticAccessibility < 5 ? 'var(--color-success)' : 'var(--color-error)'],
                  ['Lipinski', `${properties.LipinskiViolations} viol.`, properties.LipinskiViolations === 0 ? 'var(--color-success)' : 'var(--color-warn)'],
                ].map(([label, value, color]) => (
                  <div key={label} style={{
                    background: 'var(--color-surface-2)', borderRadius: 8,
                    padding: '8px 10px', border: '1px solid var(--color-border)',
                  }}>
                    <div style={{ fontSize: '0.68rem', color: 'var(--color-muted)', marginBottom: 2 }}>{label}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', color }}>{value ?? '—'}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Agent trace */}
      {agent_trace && agent_trace.length > 0 && (
        <div className="trace-panel">
          <details>
            <summary className="trace-toggle">
              <span className="chevron">▶</span>
              Agent Reasoning Trace ({agent_trace.length} steps)
            </summary>
            <div className="trace-lines">
              {agent_trace.map((line, i) => (
                <div key={i} className={`trace-line ${traceClass(line)}`}>{line}</div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  )
}

/* ── Main App ────────────────────────────────────────────── */
export default function App() {
  const [description, setDescription] = useState(
    'A selective JAK1 inhibitor with high oral bioavailability, moderate lipophilicity, and good CNS penetration.'
  )
  const [constraints, setConstraints] = useState({
    logp_min: 1.0, logp_max: 4.0,
    mw_min: 200, mw_max: 500,
    qed_min: 0.5, qed_max: 0.9,
    num_molecules: 3,
  })
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const cx = (k) => ({ value: constraints[k], onChange: (e) => setConstraints(p => ({ ...p, [k]: parseFloat(e.target.value) })) })

  const handleGenerate = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResults([])
    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_description: description, ...constraints }),
      })
      if (!res.ok) {
        const detail = await res.json()
        throw new Error(detail.detail || 'Generation failed')
      }
      setResults(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const validCount = results.filter(r => r.is_valid).length
  const avgQED = results.filter(r => r.properties?.QED).reduce((s, r, _, a) => s + r.properties.QED / a.length, 0)

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-logo">
          <div className="header-logo-icon">⚗</div>
          <h1>MoleculeForge</h1>
        </div>
        <span className="header-badge">v2.0 · Agentic Pipeline</span>
      </header>

      {/* ── Hero ── */}
      <section className="hero">
        <p className="hero-eyebrow">AI-Powered Drug Discovery</p>
        <h2>Generate <span>Novel Molecules</span><br />from Natural Language</h2>
        <p className="hero-sub">
          Fine-tuned Gemini + Chain-of-Thought reasoning · AutoGen multi-agent validation ·
          Self-reflective RAG over ZINC
        </p>
        <div className="stack-pills">
          {STACK.map(s => <span key={s} className="pill">{s}</span>)}
        </div>
      </section>

      {/* ── Main layout ── */}
      <main className="main-content">

        {/* Sidebar form */}
        <aside className="sidebar">
          <form onSubmit={handleGenerate}>
            <div className="card">
              <div className="card-title"><span className="card-title-icon">🎯</span> Target Description</div>
              <div className="form-group">
                <label htmlFor="desc">Describe the molecule</label>
                <textarea
                  id="desc"
                  rows={5}
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="Describe the desired molecule and therapeutic target…"
                />
              </div>
            </div>

            <div className="card">
              <div className="card-title"><span className="card-title-icon">⚙️</span> Property Constraints</div>

              {[
                { label: 'LogP', min: 'logp_min', max: 'logp_max', step: 0.1 },
                { label: 'Molecular Weight (Da)', min: 'mw_min', max: 'mw_max', step: 10 },
                { label: 'QED (Drug-likeness)', min: 'qed_min', max: 'qed_max', step: 0.05 },
              ].map(({ label, min, max, step }) => (
                <div key={label} className="form-group">
                  <label>{label}</label>
                  <div className="range-row">
                    <div>
                      <div className="range-label">Min</div>
                      <input type="number" step={step} {...cx(min)} />
                    </div>
                    <div>
                      <div className="range-label">Max</div>
                      <input type="number" step={step} {...cx(max)} />
                    </div>
                  </div>
                </div>
              ))}

              <div className="form-group">
                <label htmlFor="num">Candidates to generate</label>
                <input id="num" type="number" min={1} max={8} {...cx('num_molecules')} />
              </div>
            </div>

            <button
              id="generate-btn"
              type="submit"
              className={`btn-generate${loading ? ' loading' : ''}`}
              disabled={loading || !description.trim()}
            >
              {loading ? 'Running Pipeline…' : '⚗ Generate Molecules'}
            </button>
          </form>
        </aside>

        {/* Results panel */}
        <section className="results-panel">
          {error && (
            <div className="card" style={{ borderColor: 'var(--color-error)', color: 'var(--color-error)' }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading && Array.from({ length: constraints.num_molecules }).map((_, i) => (
            <div key={i} className="skeleton skeleton-card" style={{ animationDelay: `${i * 0.1}s` }} />
          ))}

          {!loading && results.length === 0 && !error && (
            <div className="empty-state">
              <div className="empty-state-icon">⚗️</div>
              <h3>Ready to generate</h3>
              <p>Describe your target molecule and set property constraints, then click Generate.</p>
            </div>
          )}

          {results.map((r, i) => <ResultCard key={i} result={r} index={i} />)}
        </section>
      </main>

      {/* Metrics bar shown after results */}
      {results.length > 0 && (
        <footer className="metrics-bar">
          <div className="metric">
            <div className="metric-value">{results.length}</div>
            <div className="metric-label">Candidates</div>
          </div>
          <div className="metric">
            <div className="metric-value">{validCount}</div>
            <div className="metric-label">Valid</div>
          </div>
          <div className="metric">
            <div className="metric-value">{results.length ? Math.round(validCount / results.length * 100) : 0}%</div>
            <div className="metric-label">Success Rate</div>
          </div>
          {avgQED > 0 && (
            <div className="metric">
              <div className="metric-value">{avgQED.toFixed(3)}</div>
              <div className="metric-label">Avg QED</div>
            </div>
          )}
        </footer>
      )}
    </div>
  )
}
