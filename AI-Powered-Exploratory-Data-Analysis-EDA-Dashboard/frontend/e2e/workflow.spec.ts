import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
test("upload, explore, filter, inspect evidence, export and reject invalid input", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Overview", exact: true }),
  ).toBeVisible();
  await page.getByText("Upload dataset", { exact: true }).click();
  await page.getByLabel("Upload dataset", { exact: true }).setInputFiles({
    name: "browser.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(
      "x,y,category\n1,2,a\n2,4,a\n3,6,b\n4,8,b\n100,200,b\n",
    ),
  });
  await expect(page.getByLabel("Active dataset")).not.toHaveValue("");
  await page.getByRole("button", { name: "Analyze dataset" }).click();
  await expect(page.getByText(/Analysis complete/)).toBeVisible();
  await page.screenshot({ path: "test-results/dashboard.png", fullPage: true });
  await page
    .getByRole("button", { name: "Data & schema", exact: true })
    .click();
  await expect(page.getByText("1–5 of 5 filtered rows")).toBeVisible();
  await expect(page.getByRole("button", { name: "Next rows" })).toBeDisabled();
  await page
    .getByLabel("Sort profiles", { exact: true })
    .selectOption("missing");
  await page.getByRole("button", { name: "Outliers", exact: true }).click();
  await page.getByLabel("Column", { exact: true }).selectOption("x");
  const outlierDownload = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export all flagged records (CSV)" })
    .click();
  expect(
    await readFile((await (await outlierDownload).path())!, "utf8"),
  ).toContain("100,200,b");
  await page.getByRole("button", { name: "Statistics", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Column statistics" }),
  ).toBeVisible();
  await page.getByLabel("Column", { exact: true }).selectOption("y");
  await expect(
    page.getByText("numeric · 5 non-null observations"),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Distributions", exact: true })
    .click();
  await expect(page.locator(".js-plotly-plot")).toHaveCount(1);
  await page.getByLabel("View", { exact: true }).selectOption("density");
  await expect(page.locator("figure")).toContainText("KDE");
  await page.getByText("Analysis settings & filters", { exact: true }).click();
  await page.getByRole("button", { name: "+ Add filter" }).click();
  await page.getByLabel("Filter column").selectOption("x");
  await page.getByLabel("Filter operator").selectOption("lt");
  await page.getByLabel("Filter value").fill("10");
  await page.getByRole("button", { name: "Analyze dataset" }).click();
  await expect(page.getByText(/Analysis complete · 4 of 5 rows/)).toBeVisible();
  await page.getByRole("button", { name: "AI insights", exact: true }).click();
  await expect(page.getByText("Evidence-backed findings")).toBeVisible();
  await expect(page.getByText("x and y: strong association")).toBeVisible();
  await page.getByRole("button", { name: "Reports", exact: true }).click();
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download JSON" }).click();
  expect((await downloaded).suggestedFilename()).toBe("prism-analysis.json");
  const exported = JSON.parse(
    await readFile((await (await downloaded).path())!, "utf8"),
  );
  expect(exported.overview.rows).toBe(4);
  await page.getByRole("checkbox", { name: "profiles", exact: true }).uncheck();
  const htmlDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download HTML" }).click();
  const report = await readFile((await (await htmlDownload).path())!, "utf8");
  expect(report).not.toContain("<h2>Column statistics and distributions</h2>");
  expect(report).toContain(exported.run_id);
  await page.getByLabel("Upload dataset", { exact: true }).setInputFiles({
    name: "bad.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("a,a\n1,2"),
  });
  await expect(page.getByRole("alert")).toContainText("Duplicate column");
});
