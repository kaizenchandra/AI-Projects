import { afterEach, describe, it, expect, vi } from "vitest";
import {
  cleanup,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@testing-library/react";
vi.mock("./Chart", () => ({
  Chart: ({ title }: { title: string }) => (
    <div role="img" aria-label={title} />
  ),
}));
import { App, fmt } from "./main";
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
describe("workspace", () => {
  it("formats unavailable values distinctly from zero", () => {
    expect(fmt(null)).toBe("Unavailable");
    expect(fmt(0)).toBe("0");
  });
  it("loads datasets and shows accessible analysis navigation", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => [
          {
            id: "d1",
            name: "measurements.csv",
            rows: 5,
            cols: 2,
            sheets: "[]",
          },
        ],
      }),
    );
    render(<App />);
    await screen.findByText("measurements.csv · 5 rows");
    expect(
      screen.getByRole("navigation", { name: "Analysis sections" }),
    ).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Active dataset"), {
      target: { value: "d1" },
    });
    expect(
      (screen.getByText("Analyze dataset ↗") as HTMLButtonElement).disabled,
    ).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Statistics" }));
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
      "Statistics",
    );
  });
  it("surfaces server errors and allows connection retry", async () => {
    const mock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: "A valid access token is required." }),
      })
      .mockResolvedValue({ ok: true, status: 200, json: async () => [] });
    vi.stubGlobal("fetch", mock);
    render(<App />);
    expect(await screen.findByRole("alert")).toBeTruthy();
    fireEvent.click(screen.getByText("Connect"));
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  });
});
