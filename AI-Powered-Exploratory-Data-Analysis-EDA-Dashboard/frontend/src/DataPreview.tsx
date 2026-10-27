import { useEffect, useState } from "react";
import type { Analysis } from "./types";
export function DataPreview({
  analysis,
  token,
}: {
  analysis: Analysis;
  token: string;
}) {
  const [offset, setOffset] = useState(0),
    [rows, setRows] = useState(analysis.preview),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    fetch(
      `/api/v1/analyses/${analysis.run_id}/rows?offset=${offset}&limit=50`,
      {
        headers: { Authorization: "Bearer " + token },
        signal: controller.signal,
      },
    )
      .then(async (r) => {
        if (!r.ok) throw Error("Unable to retrieve records.");
        return r.json();
      })
      .then((r) => setRows(r.rows))
      .catch((e) => {
        if (e.name !== "AbortError") setError(String(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [analysis.run_id, token, offset]);
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>Data preview</h2>
        <span>
          {analysis.overview.rows ? offset + 1 : 0}–
          {Math.min(offset + 50, analysis.overview.rows)} of{" "}
          {analysis.overview.rows} filtered rows
        </span>
        <button
          disabled={offset === 0 || loading}
          onClick={() => setOffset(offset - 50)}
        >
          Previous rows
        </button>
        <button
          disabled={offset + 50 >= analysis.overview.rows || loading}
          onClick={() => setOffset(offset + 50)}
        >
          Next rows
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
      {loading ? (
        <p role="status">Loading records…</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {analysis.profiles.map((p) => (
                  <th key={p.name}>{p.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>
                  {analysis.profiles.map((p) => (
                    <td key={p.name}>{row[p.name] ?? <em>null</em>}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
