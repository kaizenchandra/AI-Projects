import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const fields = [
  ['OverallQual', 'Quality (1–10)', '7'],
  ['GrLivArea', 'Living area (sq ft)', '1700'],
  ['GarageCars', 'Garage capacity', '2'],
  ['TotalBsmtSF', 'Basement area (sq ft)', '800'],
  ['YearBuilt', 'Year built', '2000'],
  ['Neighborhood', 'Neighborhood', 'CollgCr'],
];

type ModelInfo = { model_version: string; algorithm: string; currency: string; metrics: Record<string, number> };
type Prediction = { prediction: { price: number; currency: string; unit: string }; model: { name: string; version: string } };
const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [form, setForm] = useState<Record<string, string>>(Object.fromEntries(fields.map(([key, , value]) => [key, value])));
  const [result, setResult] = useState<Prediction>();
  const [modelInfo, setModelInfo] = useState<ModelInfo>();
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/api/v1/models/current`).then((response) => response.ok ? response.json() : Promise.reject()).then(setModelInfo).catch(() => undefined);
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault(); setLoading(true); setError('');
    try {
      const body = { ...form, OverallQual: +form.OverallQual, GrLivArea: +form.GrLivArea, GarageCars: +form.GarageCars, TotalBsmtSF: +form.TotalBsmtSF, YearBuilt: +form.YearBuilt };
      const response = await fetch(`${apiBase}/api/v1/predictions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!response.ok) throw new Error('Please check the highlighted values.');
      setResult(await response.json());
    } catch (cause) { setError((cause as Error).message); } finally { setLoading(false); }
  }

  return <main>
    <section className="hero"><p className="eyebrow">CLASSICAL ML · AMES HOUSING</p><h1>Estimate a home’s market value.</h1><p>Explore a reproducible regression model trained on property characteristics. Results are estimates, not appraisals.</p></section>
    <form onSubmit={submit}><h2>Property details</h2>{fields.map(([key, label]) => <label key={key}>{label}<input required value={form[key]} onChange={(event) => setForm({ ...form, [key]: event.target.value })} type={key === 'Neighborhood' ? 'text' : 'number'} /></label>)}<button disabled={loading}>{loading ? 'Estimating…' : 'Estimate price'}</button>{error && <p role="alert" className="error">{error}</p>}</form>
    {result && <aside><p className="eyebrow">MODEL ESTIMATE</p><div className="price">{new Intl.NumberFormat('en-US', { style: 'currency', currency: result.prediction.currency, maximumFractionDigits: 0 }).format(result.prediction.price)}</div><p>{result.prediction.unit.replaceAll('_', ' ')} · {result.model.name} v{result.model.version}</p><small>This is a statistical estimate and should not be treated as a guaranteed appraisal.</small></aside>}
    <section className="model-card"><p className="eyebrow">MODEL INSIGHTS</p>{modelInfo ? <><h2>{modelInfo.algorithm}</h2><p>Version {modelInfo.model_version} · {modelInfo.currency} · held-out test evaluation.</p><dl>{Object.entries(modelInfo.metrics).filter(([key]) => ['mae', 'rmse', 'r2'].includes(key)).map(([key, value]) => <div key={key}><dt>{key.toUpperCase()}</dt><dd>{value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</dd></div>)}</dl></> : <p>Model evaluation details are unavailable until the API is ready.</p>}</section>
  </main>;
}

createRoot(document.getElementById('root')!).render(<App />);
