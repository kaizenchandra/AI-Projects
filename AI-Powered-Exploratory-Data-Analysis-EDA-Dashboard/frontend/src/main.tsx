import React, { useEffect, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import { Chart } from "./Chart";
import { DataPreview } from "./DataPreview";
import type { Analysis, Config, Dataset } from "./types";
import "./style.css";
export const fmt = (v: unknown): string =>
  v === null || v === undefined
    ? "Unavailable"
    : typeof v === "number"
      ? v.toLocaleString(undefined, { maximumFractionDigits: 4 })
      : String(v);
const initial: Config = {
  filters: [],
  overrides: {},
  correlation: "pearson",
  outlier_method: "iqr",
  threshold: 1.5,
  bins: 0,
  columns: [],
};
const tabs = [
  "Overview",
  "Data & schema",
  "Statistics",
  "Distributions",
  "Relationships",
  "Outliers",
  "AI insights",
  "Reports",
];
export function App() {
  const [token, setToken] = useState(""),
    [datasets, setDatasets] = useState<Dataset[]>([]),
    [selected, setSelected] = useState(""),
    [analysis, setAnalysis] = useState<Analysis | null>(null),
    [tab, setTab] = useState("Overview"),
    [column, setColumn] = useState(""),
    [config, setConfig] = useState<Config>(initial),
    [busy, setBusy] = useState(""),
    [error, setError] = useState(""),
    [job, setJob] = useState(""),
    [progress, setProgress] = useState(0),
    [delimiter, setDelimiter] = useState(","),
    [encoding, setEncoding] = useState("utf-8"),
    [header, setHeader] = useState(true),
    [sheet, setSheet] = useState(""),
    [search, setSearch] = useState(""),
    [profileSort, setProfileSort] = useState("name"),
    [reportSections, setReportSections] = useState([
      "overview",
      "quality",
      "profiles",
      "correlations",
      "associations",
      "outliers",
      "insights",
    ]),
    [severity, setSeverity] = useState("all"),
    [pair, setPair] = useState(0),
    [chartType, setChartType] = useState("histogram"),
    [explanations, setExplanations] = useState("");
  const xhr = useRef<XMLHttpRequest | null>(null);
  async function api(path: string, options: RequestInit = {}) {
    const r = await fetch("/api/v1" + path, {
      ...options,
      headers: { Authorization: "Bearer " + token, ...options.headers },
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw Error(
        typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail),
      );
    }
    return r.status === 204 ? null : r.json();
  }
  async function load() {
    try {
      setDatasets(await api("/datasets"));
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  useEffect(() => {
    void load();
  }, []);
  useEffect(() => {
    if (!job) return;
    let stopped = false;
    const timer = setInterval(async () => {
      try {
        const j = await api("/jobs/" + job);
        if (stopped) return;
        setBusy(j.state);
        if (j.state === "completed") {
          const a = await api("/analyses/" + job);
          if (stopped) return;
          setAnalysis(a);
          setColumn(a.profiles[0]?.name || "");
          setPair(0);
          setBusy("");
          setJob("");
        } else if (["failed", "cancelled"].includes(j.state)) {
          setBusy("");
          setJob("");
          if (j.error) setError(j.error);
        }
      } catch (e) {
        setError(String(e));
        setJob("");
        setBusy("");
      }
    }, 900);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [job, token]);
  async function run(id = selected) {
    if (!id) return;
    setError("");
    setAnalysis(null);
    setBusy("queued");
    try {
      const j = await api("/datasets/" + id + "/analyses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      setJob(j.id);
    } catch (e) {
      setError(String(e));
      setBusy("");
    }
  }
  function upload(file?: File) {
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) {
      setError("Maximum upload size is 20 MiB.");
      return;
    }
    setBusy("uploading");
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("delimiter", delimiter);
    form.append("encoding", encoding);
    form.append("header", String(header));
    form.append("sheet", sheet);
    const req = new XMLHttpRequest();
    xhr.current = req;
    req.open("POST", "/api/v1/datasets");
    req.setRequestHeader("Authorization", "Bearer " + token);
    req.upload.onprogress = (e) =>
      setProgress(
        e.lengthComputable ? Math.round((100 * e.loaded) / e.total) : 0,
      );
    req.onerror = () => {
      setError("Upload failed. Check connection and retry.");
      setBusy("");
    };
    req.onabort = () => setBusy("");
    req.onload = () => {
      setBusy("");
      let d;
      try {
        d = JSON.parse(req.responseText);
      } catch {
        setError("Invalid server response");
        return;
      }
      if (req.status >= 400) {
        setError(d.detail || "Upload failed");
        return;
      }
      setSelected(d.id);
      setConfig(initial);
      void load();
      setAnalysis(null);
      setTab("Overview");
    };
    req.send(form);
  }
  async function download(format: string) {
    if (!analysis) return;
    try {
      const r = await fetch(
        "/api/v1/analyses/" +
          analysis.run_id +
          "/report?format=" +
          format +
          "&column=" +
          encodeURIComponent(column) +
          "&sections=" +
          (reportSections.join(",") || "none"),
        { headers: { Authorization: "Bearer " + token } },
      );
      if (!r.ok) throw Error("Export failed");
      const url = URL.createObjectURL(await r.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = "prism-analysis." + (format === "outliers" ? "csv" : format);
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setError(String(e));
    }
  }
  const p = analysis?.profiles.find((p) => p.name === column),
    rel = analysis?.correlations[pair];
  const profileSelect = (
    <label>
      Column
      <select
        aria-label="Column"
        value={column}
        onChange={(e) => setColumn(e.target.value)}
      >
        {analysis?.profiles.map((p) => (
          <option key={p.name}>{p.name}</option>
        ))}
      </select>
    </label>
  );
  return (
    <div className="app">
      <aside>
        <a className="brand" href="#main">
          <span>◈</span> prism
          <span className="brand-sub">DATA INTELLIGENCE</span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Analysis sections">
          {tabs.map((t, i) => (
            <button
              key={t}
              className={tab === t ? "active" : ""}
              onClick={() => setTab(t)}
            >
              <span aria-hidden="true">
                {["◫", "▤", "∑", "▥", "⌘", "◎", "✧", "↗"][i]}
              </span>
              {t}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="dot" /> Evidence before interpretation
          <p>
            Private by default.
            <br />
            Calculated from your data.
          </p>
        </div>
      </aside>
      <main id="main">
        <header>
          <div>
            <span className="eyebrow">ANALYTICAL WORKSPACE</span>
            <h1>{tab}</h1>
            <p>Understand your data. Find what deserves a closer look.</p>
          </div>
          <span className="badge">PRISM / EDA 1.0</span>
        </header>
        <details className="access">
          <summary>Connection & access</summary>
          <label>
            Access token
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Not needed in explicit local development mode"
            />
          </label>
          <button
            onClick={() => {
              setAnalysis(null);
              setSelected("");
              void load();
            }}
          >
            Connect
          </button>
          <small>
            Tokens stay in memory. Each token owns an isolated workspace.
          </small>
        </details>
        {error && (
          <div role="alert" className="error">
            {error}
            <button onClick={() => setError("")}>Dismiss</button>
          </div>
        )}
        <section className="workspace">
          <label>
            Active dataset
            <select
              value={selected}
              disabled={!!busy}
              onChange={(e) => {
                setSelected(e.target.value);
                setAnalysis(null);
                setConfig(initial);
              }}
            >
              <option value="">Select a dataset</option>
              {datasets.map((d) => (
                <option value={d.id} key={d.id}>
                  {d.name} · {d.rows.toLocaleString()} rows
                </option>
              ))}
            </select>
          </label>
          <button
            className="primary"
            disabled={!selected || !!busy}
            onClick={() => void run()}
          >
            Analyze dataset ↗
          </button>
          <button
            disabled={!selected || !!busy}
            onClick={async () => {
              if (!confirm("Permanently delete this dataset and its analyses?"))
                return;
              try {
                await api("/datasets/" + selected, { method: "DELETE" });
                setSelected("");
                setAnalysis(null);
                await load();
              } catch (e) {
                setError(String(e));
              }
            }}
          >
            Delete
          </button>
          <details>
            <summary>Upload dataset</summary>
            <div
              className="drop"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!busy) upload(e.dataTransfer.files[0]);
              }}
            >
              <strong>Drop a dataset here</strong>
              <p>CSV, XLSX or Parquet · up to 20 MiB</p>
              <input
                aria-label="Upload dataset"
                disabled={!!busy}
                type="file"
                accept=".csv,.xlsx,.parquet"
                onChange={(e) => upload(e.target.files?.[0])}
              />
            </div>
            <div className="controls">
              <label>
                Delimiter
                <input
                  value={delimiter}
                  maxLength={1}
                  onChange={(e) => setDelimiter(e.target.value)}
                />
              </label>
              <label>
                Encoding
                <select
                  value={encoding}
                  onChange={(e) => setEncoding(e.target.value)}
                >
                  <option>utf-8</option>
                  <option>utf-8-sig</option>
                  <option>latin-1</option>
                </select>
              </label>
              <label>
                Worksheet
                <input
                  value={sheet}
                  onChange={(e) => setSheet(e.target.value)}
                  placeholder="First sheet by default"
                  list="sheets"
                />
                <datalist id="sheets">
                  {JSON.parse(
                    datasets.find((d) => d.id === selected)?.sheets || "[]",
                  ).map((s: string) => (
                    <option key={s}>{s}</option>
                  ))}
                </datalist>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={header}
                  onChange={(e) => setHeader(e.target.checked)}
                />{" "}
                First row is header
              </label>
            </div>
            <small>
              After importing XLSX, its sheet names appear in the worksheet
              selector. Select a sheet and upload again to import it.
            </small>
          </details>
        </section>
        {busy && (
          <div role="status" className="status">
            <span className="spinner" />
            {busy === "uploading"
              ? `Uploading ${progress}%`
              : busy + " analysis…"}
            <button
              onClick={() => {
                if (xhr.current && busy === "uploading") xhr.current.abort();
                if (job)
                  void api("/jobs/" + job, { method: "DELETE" }).catch((e) =>
                    setError(String(e)),
                  );
              }}
            >
              Cancel
            </button>
          </div>
        )}
        {!analysis && !busy && (
          <section className="empty">
            <div>◈</div>
            <h2>Your next insight starts with a dataset.</h2>
            <p>
              Upload a file, select it, then run an analysis.
              <br />
              Explore quality, distributions and relationships with traceable
              evidence.
            </p>
            <button
              onClick={() => {
                document.querySelector<HTMLDetailsElement>(
                  ".workspace details",
                )!.open = true;
              }}
            >
              Import your data
            </button>
          </section>
        )}
        {analysis && (
          <>
            <div className="runline">
              <span className="dot" /> Analysis complete ·{" "}
              {analysis.overview.rows.toLocaleString()} of{" "}
              {analysis.overview.source_rows.toLocaleString()} rows
              <span>
                Run {analysis.run_id.slice(0, 8)} ·{" "}
                {new Date(analysis.created_at).toLocaleString()}
              </span>
            </div>
            <details className="panel">
              <summary>Analysis settings & filters</summary>
              <p>
                Changes apply only when you press Analyze dataset. Current
                charts belong to the completed run above.
              </p>
              <div className="controls">
                <label>
                  Correlation
                  <select
                    value={config.correlation}
                    onChange={(e) =>
                      setConfig({ ...config, correlation: e.target.value })
                    }
                  >
                    <option>pearson</option>
                    <option>spearman</option>
                  </select>
                </label>
                <label>
                  Outlier method
                  <select
                    value={config.outlier_method}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        outlier_method: e.target.value,
                        threshold: e.target.value === "iqr" ? 1.5 : 3.5,
                      })
                    }
                  >
                    <option>iqr</option>
                    <option>zscore</option>
                    <option>mad</option>
                  </select>
                </label>
                <label>
                  Threshold
                  <input
                    type="number"
                    min="0.1"
                    max="20"
                    step="0.1"
                    value={config.threshold}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        threshold: Number(e.target.value),
                      })
                    }
                  />
                </label>
                <label>
                  Histogram bins (0 = automatic)
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={config.bins}
                    onChange={(e) =>
                      setConfig({ ...config, bins: Number(e.target.value) })
                    }
                  />
                </label>
              </div>
              <label>
                Correlation columns (up to 20; none = automatic)
                <select
                  multiple
                  aria-label="Correlation columns"
                  value={config.columns}
                  onChange={(e) => {
                    const columns = Array.from(
                      e.target.selectedOptions,
                      (o) => o.value,
                    );
                    if (columns.length <= 20) setConfig({ ...config, columns });
                    else setError("Choose at most 20 correlation columns.");
                  }}
                >
                  {analysis.profiles
                    .filter((p) => p.semantic_type === "numeric")
                    .map((p) => (
                      <option key={p.name}>{p.name}</option>
                    ))}
                </select>
              </label>
              <button
                onClick={() =>
                  setConfig({
                    ...config,
                    filters: [
                      ...config.filters,
                      {
                        column: column || analysis.profiles[0].name,
                        op: "eq",
                        value: "",
                      },
                    ],
                  })
                }
                disabled={config.filters.length >= 10}
              >
                + Add filter
              </button>
              {config.filters.map((f, i) => (
                <div className="controls" key={i}>
                  <select
                    aria-label="Filter column"
                    value={f.column}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        filters: config.filters.map((x, j) =>
                          j === i ? { ...x, column: e.target.value } : x,
                        ),
                      })
                    }
                  >
                    {analysis.profiles.map((p) => (
                      <option key={p.name}>{p.name}</option>
                    ))}
                  </select>
                  <select
                    aria-label="Filter operator"
                    value={f.op}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        filters: config.filters.map((x, j) =>
                          j === i ? { ...x, op: e.target.value } : x,
                        ),
                      })
                    }
                  >
                    {[
                      "eq",
                      "ne",
                      "gt",
                      "ge",
                      "lt",
                      "le",
                      "contains",
                      "is_null",
                    ].map((v) => (
                      <option key={v}>{v}</option>
                    ))}
                  </select>
                  <input
                    aria-label="Filter value"
                    value={f.value || ""}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        filters: config.filters.map((x, j) =>
                          j === i ? { ...x, value: e.target.value } : x,
                        ),
                      })
                    }
                  />
                  <button
                    onClick={() =>
                      setConfig({
                        ...config,
                        filters: config.filters.filter((_, j) => i !== j),
                      })
                    }
                  >
                    Remove
                  </button>
                </div>
              ))}
              <small>
                Range comparisons support numbers and ISO dates. Use equality or
                contains (for example 2026-10). All filters are ANDed.
              </small>
            </details>
            {tab === "Overview" && (
              <>
                <div className="metrics">
                  {[
                    ["Records", analysis.overview.rows],
                    ["Columns", analysis.overview.columns],
                    [
                      "Missing cells",
                      fmt(analysis.overview.missing_percentage) + "%",
                    ],
                    ["Duplicate rows", analysis.overview.duplicate_rows],
                  ].map(([k, v]) => (
                    <article key={k}>
                      <span>{k}</span>
                      <strong>{fmt(v)}</strong>
                      <small>Full filtered dataset</small>
                    </article>
                  ))}
                </div>
                <div className="grid">
                  <section className="panel">
                    <h2>Completeness by column</h2>
                    <Chart
                      title="Missing values (%)"
                      layout={{
                        yaxis: {
                          range: [0, 100],
                          title: { text: "Missing cells (%)" },
                        },
                      }}
                      data={[
                        {
                          type: "bar",
                          x: analysis.profiles.map((p) => p.name),
                          y: analysis.profiles.map((p) => p.missing_percentage),
                          marker: { color: "#087e8b" },
                        },
                      ]}
                    />
                  </section>
                  <section className="panel">
                    <h2>
                      Signals worth exploring{" "}
                      <span className="badge">{analysis.insights.length}</span>
                    </h2>
                    {analysis.insights.slice(0, 4).map((i) => (
                      <article className="insight" key={i.id}>
                        <span className="tag">{i.type}</span>
                        <h3>{i.title}</h3>
                        <p>{i.recommendation}</p>
                        <small>
                          {i.evidence.metric}: {fmt(i.evidence.value)}
                        </small>
                      </article>
                    ))}
                    {!analysis.insights.length && (
                      <p>
                        No configured rules triggered. This does not establish
                        that the data is error-free.
                      </p>
                    )}
                  </section>
                </div>
              </>
            )}
            {tab === "Data & schema" && (
              <>
                <section className="panel">
                  <h2>Schema explorer</h2>
                  <label>
                    Sort profiles
                    <select
                      aria-label="Sort profiles"
                      value={profileSort}
                      onChange={(e) => setProfileSort(e.target.value)}
                    >
                      <option value="name">Column name</option>
                      <option value="missing">
                        Missingness (highest first)
                      </option>
                      <option value="distinct">
                        Distinct values (highest first)
                      </option>
                    </select>
                  </label>
                  <label>
                    Search columns
                    <input
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                  </label>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Physical</th>
                          <th>Inferred</th>
                          <th>Override for next run</th>
                          <th>Quality notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysis.profiles
                          .filter((p) =>
                            p.name.toLowerCase().includes(search.toLowerCase()),
                          )
                          .sort((a, b) =>
                            profileSort === "missing"
                              ? b.missing_percentage - a.missing_percentage
                              : profileSort === "distinct"
                                ? b.distinct_count - a.distinct_count
                                : a.name.localeCompare(b.name),
                          )
                          .map((p) => (
                            <tr key={p.name}>
                              <th>{p.name}</th>
                              <td>{p.physical_type}</td>
                              <td>{p.inferred_type}</td>
                              <td>
                                <select
                                  aria-label={"Override " + p.name}
                                  value={config.overrides[p.name] || ""}
                                  onChange={(e) => {
                                    const overrides = { ...config.overrides };
                                    if (e.target.value)
                                      overrides[p.name] = e.target.value;
                                    else delete overrides[p.name];
                                    setConfig({ ...config, overrides });
                                  }}
                                >
                                  <option value="">Automatic</option>
                                  {[
                                    "numeric",
                                    "categorical",
                                    "text",
                                    "temporal",
                                    "identifier",
                                    "boolean",
                                  ].map((t) => (
                                    <option key={t}>{t}</option>
                                  ))}
                                </select>
                              </td>
                              <td>
                                {[
                                  p.constant && "Constant",
                                  p.high_cardinality && "High cardinality",
                                  p.potential_sensitive && "Potential PII",
                                  p.precision_warning && "Float precision risk",
                                ]
                                  .filter(Boolean)
                                  .join(" · ") || "—"}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                <DataPreview
                  key={analysis.run_id}
                  analysis={analysis}
                  token={token}
                />
                <Chart
                  title={`Missingness matrix · first ${analysis.missingness.shown_rows} rows (1 = missing)`}
                  data={[
                    {
                      type: "heatmap",
                      x: analysis.missingness.columns,
                      z: analysis.missingness.matrix.map((r) => r.map(Number)),
                      colorscale: [
                        [0, "#e5f3f2"],
                        [1, "#df8e40"],
                      ],
                      zmin: 0,
                      zmax: 1,
                    },
                  ]}
                />
              </>
            )}
            {tab === "Statistics" && (
              <section className="panel">
                <div className="section-heading">
                  <h2>Column statistics</h2>
                  {profileSelect}
                </div>
                {p && (
                  <>
                    <p>
                      {p.semantic_type} · {p.non_null_count} non-null
                      observations · {p.null_count} missing · {p.distinct_count}{" "}
                      distinct
                    </p>
                    <div className="metrics compact">
                      {Object.entries(p.statistics).map(([k, v]) => (
                        <article key={k}>
                          <span>{k.replaceAll("_", " ")}</span>
                          <strong>{fmt(v)}</strong>
                        </article>
                      ))}
                    </div>
                    <p>
                      Invalid numerical values: {p.invalid_count ?? 0} ·
                      Nonfinite values: {p.nonfinite_count ?? 0}
                    </p>
                    {p.temporal && (
                      <p>
                        {p.temporal.earliest} → {p.temporal.latest}.{" "}
                        {p.temporal.cadence}
                      </p>
                    )}
                    <h3>Most frequent values</h3>
                    <table>
                      <thead>
                        <tr>
                          <th>Value</th>
                          <th>Count</th>
                          <th>Relative frequency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {p.top_values.map((v) => (
                          <tr key={v.value}>
                            <td>{v.value}</td>
                            <td>{v.count}</td>
                            <td>{fmt(100 * v.relative_frequency)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p>Other values: {p.other_count} records. Top 20 shown.</p>
                  </>
                )}
              </section>
            )}
            {tab === "Distributions" && (
              <section className="panel">
                <div className="section-heading">
                  <h2>Distribution explorer</h2>
                  {profileSelect}
                  <label>
                    View
                    <select
                      aria-label="View"
                      value={chartType}
                      onChange={(e) => setChartType(e.target.value)}
                    >
                      {[
                        "histogram",
                        "density",
                        "box",
                        "violin",
                        "bar",
                        "donut",
                        "time",
                      ].map((t) => (
                        <option key={t}>{t}</option>
                      ))}
                    </select>
                  </label>
                </div>
                {p && (
                  <>
                    {chartType === "histogram" && p.distribution.counts ? (
                      <Chart
                        title={`${column} · histogram, n=${p.non_null_count - (p.invalid_count || 0) - (p.nonfinite_count || 0)}`}
                        data={[
                          {
                            type: "bar",
                            x: p.distribution
                              .edges!.slice(0, -1)
                              .map(
                                (v, i) =>
                                  (v + p.distribution.edges![i + 1]) / 2,
                              ),
                            y: p.distribution.counts,
                            width: p.distribution
                              .edges!.slice(0, -1)
                              .map((v, i) => p.distribution.edges![i + 1] - v),
                          },
                        ]}
                      />
                    ) : chartType === "density" && p.distribution.kde ? (
                      <Chart
                        title={`${column} · KDE, sample n=${p.distribution.kde.n}`}
                        data={[
                          {
                            type: "scatter",
                            mode: "lines",
                            x: p.distribution.kde.x,
                            y: p.distribution.kde.y,
                            fill: "tozeroy",
                          },
                        ]}
                      />
                    ) : chartType === "box" && p.distribution.box ? (
                      <Chart
                        title={`${column} · quartiles with min/max whiskers`}
                        data={[
                          {
                            type: "box",
                            q1: [p.statistics.q1!],
                            median: [p.statistics.median!],
                            q3: [p.statistics.q3!],
                            lowerfence: [p.statistics.min!],
                            upperfence: [p.statistics.max!],
                            name: column,
                          },
                        ]}
                      />
                    ) : chartType === "violin" && p.distribution.kde ? (
                      <Chart
                        title={`${column} · mirrored KDE (violin density)`}
                        data={[
                          {
                            type: "scatter",
                            mode: "lines",
                            x: p.distribution.kde.y,
                            y: p.distribution.kde.x,
                            fill: "tozerox",
                          },
                          {
                            type: "scatter",
                            mode: "lines",
                            x: p.distribution.kde.y.map((y) => -y),
                            y: p.distribution.kde.x,
                            fill: "tozerox",
                          },
                        ]}
                      />
                    ) : ["bar", "donut"].includes(chartType) ? (
                      <Chart
                        title={`${column} · top 20 + Other`}
                        data={
                          chartType === "donut"
                            ? [
                                {
                                  type: "pie",
                                  labels: [
                                    ...p.top_values.map((v) => v.value),
                                    "Other",
                                  ],
                                  values: [
                                    ...p.top_values.map((v) => v.count),
                                    p.other_count,
                                  ],
                                  hole: 0.55,
                                },
                              ]
                            : [
                                {
                                  type: "bar",
                                  x: [
                                    ...p.top_values.map((v) => v.value),
                                    "Other",
                                  ],
                                  y: [
                                    ...p.top_values.map((v) => v.count),
                                    p.other_count,
                                  ],
                                },
                              ]
                        }
                      />
                    ) : chartType === "time" && p.temporal ? (
                      <Chart
                        title={`${column} · daily records (${p.temporal.daily_counts.length}/${p.temporal.periods_total} days shown)`}
                        data={[
                          {
                            type: "scatter",
                            mode: "lines",
                            x: p.temporal.daily_counts.map((v) => v[0]),
                            y: p.temporal.daily_counts.map((v) => v[1]),
                          },
                        ]}
                      />
                    ) : (
                      <p className="empty-small">
                        This view is unavailable for the selected column or
                        sample size. Try another view.
                      </p>
                    )}
                  </>
                )}
              </section>
            )}
            {tab === "Relationships" && (
              <>
                <section className="panel">
                  <h2>
                    Numerical relationships · {analysis.config.correlation}
                  </h2>
                  <p>
                    Finite pairwise deletion. Minimum 3 pairs, nonconstant
                    variables. Correlation does not imply causation.
                  </p>
                  {analysis.correlations.length > 0 ? (
                    <>
                      <Chart
                        title="Pairwise correlation heatmap"
                        data={[
                          {
                            type: "heatmap",
                            x: Array.from(
                              new Set(
                                analysis.correlations.flatMap((r) => [
                                  r.x,
                                  r.y,
                                ]),
                              ),
                            ),
                            y: Array.from(
                              new Set(
                                analysis.correlations.flatMap((r) => [
                                  r.x,
                                  r.y,
                                ]),
                              ),
                            ),
                            z: Array.from(
                              new Set(
                                analysis.correlations.flatMap((r) => [
                                  r.x,
                                  r.y,
                                ]),
                              ),
                            ).map((a) =>
                              Array.from(
                                new Set(
                                  analysis.correlations.flatMap((r) => [
                                    r.x,
                                    r.y,
                                  ]),
                                ),
                              ).map((b) =>
                                a === b
                                  ? null
                                  : (analysis.correlations.find(
                                      (r) =>
                                        (r.x === a && r.y === b) ||
                                        (r.x === b && r.y === a),
                                    )?.coefficient ?? null),
                              ),
                            ),
                            zmin: -1,
                            zmax: 1,
                            colorscale: "RdBu",
                          },
                        ]}
                      />
                      <label>
                        Variable pair
                        <select
                          value={pair}
                          onChange={(e) => setPair(Number(e.target.value))}
                        >
                          {analysis.correlations.map((r, i) => (
                            <option key={i} value={i}>
                              {r.x} × {r.y} · r={fmt(r.coefficient)} · n={r.n}
                            </option>
                          ))}
                        </select>
                      </label>
                      {rel && (
                        <Chart
                          title={`${rel.x} × ${rel.y} · ${rel.plotted_n}/${rel.n} pairs shown`}
                          data={[
                            {
                              type: "scatter",
                              mode: "markers",
                              x: rel.points.map((p) => p[0]),
                              y: rel.points.map((p) => p[1]),
                            },
                          ]}
                          layout={{
                            xaxis: { title: { text: rel.x } },
                            yaxis: { title: { text: rel.y } },
                          }}
                        />
                      )}
                    </>
                  ) : (
                    <p>No eligible numerical pairs. Check schema overrides.</p>
                  )}
                  <h3>Categorical associations</h3>
                  {analysis.associations.map((r) => (
                    <p key={r.x + r.y}>
                      {r.x} × {r.y}: Cramér’s V {fmt(r.cramers_v)} · n={r.n}
                    </p>
                  ))}
                </section>
              </>
            )}
            {tab === "Outliers" && (
              <section className="panel">
                <div className="section-heading">
                  <h2>Potential anomalies</h2>
                  {profileSelect}
                </div>
                <p>
                  Univariate {analysis.config.outlier_method}, threshold{" "}
                  {analysis.config.threshold}. Flags are investigation
                  candidates, not errors.
                </p>
                {p?.outliers ? (
                  <>
                    <div className="metrics">
                      <article>
                        <span>Flagged records</span>
                        <strong>{p.outliers.count}</strong>
                      </article>
                      <article>
                        <span>Lower fence</span>
                        <strong>{fmt(p.outliers.lower)}</strong>
                      </article>
                      <article>
                        <span>Upper fence</span>
                        <strong>{fmt(p.outliers.upper)}</strong>
                      </article>
                    </div>
                    <Chart
                      title={`${column} · flagged values (first 200)`}
                      data={[
                        {
                          type: "scatter",
                          mode: "markers",
                          x: p.outlier_records?.map((r) => r.row_index),
                          y: p.outlier_records?.map((r) => Number(r.value)),
                          marker: { color: "#c65e34" },
                        },
                      ]}
                    />
                    <table>
                      <thead>
                        <tr>
                          <th>Filtered row index (0-based)</th>
                          <th>Value</th>
                          <th>Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {p.outlier_records?.map((r) => (
                          <tr key={r.row_index}>
                            <td>{r.row_index}</td>
                            <td>{fmt(r.value)}</td>
                            <td>
                              Outside [{fmt(p.outliers!.lower)},{" "}
                              {fmt(p.outliers!.upper)}]
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p>
                      Showing {p.outlier_records?.length} of{" "}
                      {p.outlier_records_total} candidate records. Full counts
                      in JSON export.
                    </p>
                    <button
                      disabled={!p.outliers.eligible}
                      onClick={() => void download("outliers")}
                    >
                      Export all flagged records (CSV)
                    </button>
                  </>
                ) : (
                  <p>
                    Select a numerical column. Identifier columns are excluded
                    unless explicitly overridden.
                  </p>
                )}
              </section>
            )}
            {tab === "AI insights" && (
              <section className="panel">
                <div className="section-heading">
                  <h2>Evidence-backed findings</h2>
                  <label>
                    Category
                    <select
                      value={severity}
                      onChange={(e) => setSeverity(e.target.value)}
                    >
                      <option value="all">All</option>
                      {Array.from(
                        new Set(analysis.insights.map((i) => i.type)),
                      ).map((t) => (
                        <option key={t}>{t}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <p>
                  {analysis.insights.length} deterministic findings from{" "}
                  {analysis.overview.rows} records. Rules work without external
                  credentials.
                </p>
                {analysis.insights
                  .filter((i) => severity === "all" || i.type === severity)
                  .map((i) => (
                    <article className="insight" key={i.id}>
                      <span className="tag">
                        {i.severity} · {i.type}
                      </span>
                      <h3>{i.title}</h3>
                      <p>{i.recommendation}</p>
                      <code>
                        {i.evidence.metric} = {fmt(i.evidence.value)}
                      </code>
                      <p>
                        <small>
                          {i.limitations.join(" ")} · {i.method}
                        </small>
                      </p>
                      <button
                        onClick={() => {
                          if (i.affected_columns[0])
                            setColumn(i.affected_columns[0]);
                          setTab("Statistics");
                        }}
                      >
                        Inspect supporting statistics →
                      </button>
                    </article>
                  ))}
                <h3>Optional external explanation</h3>
                <p>
                  Server administrators must explicitly enable external AI. Only
                  anonymized aggregate evidence is sent. Generated prose is an
                  unverified interpretation.
                </p>
                <button
                  onClick={async () => {
                    try {
                      const r = await api(
                        "/analyses/" + analysis.run_id + "/explanations",
                        { method: "POST" },
                      );
                      setExplanations(JSON.stringify(r, null, 2));
                    } catch (e) {
                      setError(String(e));
                    }
                  }}
                >
                  Request explanation
                </button>
                {explanations && <pre>{explanations}</pre>}
              </section>
            )}
            {tab === "Reports" && (
              <section className="panel">
                <span className="eyebrow">REPRODUCIBLE BY DESIGN</span>
                <h2>Take your analysis with you.</h2>
                <fieldset>
                  <legend>HTML report sections</legend>
                  <div className="controls">
                    {[
                      "overview",
                      "quality",
                      "profiles",
                      "correlations",
                      "associations",
                      "outliers",
                      "insights",
                    ].map((section) => (
                      <label key={section}>
                        <input
                          type="checkbox"
                          checked={reportSections.includes(section)}
                          onChange={(e) =>
                            setReportSections(
                              e.target.checked
                                ? [...reportSections, section]
                                : reportSections.filter((s) => s !== section),
                            )
                          }
                        />
                        {section}
                      </label>
                    ))}
                  </div>
                  <small>
                    Provenance, configuration and limitations are always
                    included.
                  </small>
                </fieldset>
                <p>
                  All exports refer to this completed run, its fingerprint,
                  filters and method settings.
                </p>
                <div className="controls">
                  {["html", "json", "csv"].map((f) => (
                    <button
                      className="primary"
                      key={f}
                      onClick={() => void download(f)}
                    >
                      Download {f.toUpperCase()} ↗
                    </button>
                  ))}
                </div>
                <p>
                  HTML includes all analytical sections and supports browser
                  print-to-PDF. CSV contains column summaries. Download
                  individual PNG charts from their toolbars.
                </p>
                <h3>Methods & limitations</h3>
                <pre>{JSON.stringify(analysis.methodology, null, 2)}</pre>
                <p>
                  Fingerprint: <code>{analysis.fingerprint}</code>
                </p>
              </section>
            )}
          </>
        )}
        <footer>
          PRISM · Statistical evidence, clearly explained.
          <span>Original values preserved · No external AI by default</span>
        </footer>
      </main>
    </div>
  );
}
if (document.getElementById("root"))
  createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
