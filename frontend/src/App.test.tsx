import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import {
  health,
  maitriDetail,
  maitriMetrics,
  maitriSafety,
  maitriState,
  stationSummaries,
} from "./test/fixtures";

/**
 * Routes a mocked fetch to the fixture matching each real endpoint path, so
 * this exercises App -> CommandCenter -> the real api client -> real
 * component rendering, with only the network call itself replaced -- the
 * same integration path a live backend would drive, against response
 * shapes taken from one (see test/fixtures.ts).
 */
function mockApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      const ok = (body: unknown) =>
        new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });

      if (url.endsWith("/health")) return ok(health);
      if (url.endsWith("/stations")) return ok(stationSummaries);
      if (url.endsWith("/stations/maitri")) return ok(maitriDetail);
      if (url.endsWith("/stations/maitri/state")) return ok(maitriState);
      if (url.endsWith("/stations/maitri/safety")) return ok(maitriSafety);
      if (url.endsWith("/stations/maitri/metrics")) return ok(maitriMetrics);
      return new Response("not found", { status: 404 });
    }),
  );
}

describe("App", () => {
  beforeEach(() => {
    mockApi();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the API as reachable and lists real stations from the mocked backend", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText(/api up/)).toBeInTheDocument());
    expect(screen.getByText("maitri")).toBeInTheDocument();
    expect(screen.getByText("bharati")).toBeInTheDocument();
  });

  it("renders the selected station's real telemetry, not placeholder numbers", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("maitri")).toBeInTheDocument());
    fireEvent.click(screen.getByText("maitri"));

    await waitFor(() => expect(screen.getByText("Maitri")).toBeInTheDocument());
    // electrical_load_kw from the mocked /state response
    await waitFor(() => expect(screen.getByText("121.4")).toBeInTheDocument());
    expect(screen.getByText("45.0")).toBeInTheDocument(); // critical_load_kw
  });

  it("shows an error banner when the API is unreachable, never invented data", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    render(<App />);
    await waitFor(() =>
      expect(screen.getByText(/Cannot reach the Allotrope API/)).toBeInTheDocument(),
    );
    expect(screen.getByText("api unreachable")).toBeInTheDocument();
  });
});
