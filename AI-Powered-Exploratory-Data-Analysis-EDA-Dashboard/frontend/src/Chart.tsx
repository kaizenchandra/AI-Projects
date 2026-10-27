import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";
export function Chart({
  data,
  title,
  layout = {},
}: {
  data: (
    | Data
    | {
        type: "box";
        q1: number[];
        median: number[];
        q3: number[];
        lowerfence: number[];
        upperfence: number[];
        name: string;
      }
  )[];
  title: string;
  layout?: Partial<Layout>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current!;
    void Plotly.react(
      el,
      data as Data[],
      {
        title: { text: title },
        autosize: true,
        height: 360,
        margin: { l: 65, r: 25, t: 55, b: 65 },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#fff",
        font: { family: "system-ui", color: "#334155" },
        colorway: ["#087e8b", "#f0a452", "#8b5cf6"],
        ...layout,
      },
      {
        responsive: true,
        displaylogo: false,
        toImageButtonOptions: {
          format: "png",
          filename: "prism-chart",
          height: 700,
          width: 1100,
          scale: 1,
        },
      },
    );
    const observer = new ResizeObserver(() => {
      void Plotly.Plots.resize(el);
    });
    observer.observe(el);
    return () => {
      observer.disconnect();
      Plotly.purge(el);
    };
  }, [data, title, layout]);
  return (
    <figure aria-label={title}>
      <div ref={ref} />
      <figcaption>
        {title}. Use the chart toolbar to zoom, pan, or download PNG. Numerical
        summaries are available in Statistics.
      </figcaption>
    </figure>
  );
}
