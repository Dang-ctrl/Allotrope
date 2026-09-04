import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SafetyPanel } from "./SafetyPanel";
import { maitriSafety } from "../test/fixtures";

describe("SafetyPanel", () => {
  it("shows 'action passed unmodified' when the last report did not intervene", () => {
    render(<SafetyPanel safety={maitriSafety} />);
    expect(screen.getByText("action passed unmodified")).toBeInTheDocument();
  });

  it("translates intervention codes into operator-readable text", () => {
    render(
      <SafetyPanel
        safety={{
          ...maitriSafety,
          last_report: {
            ...maitriSafety.last_report!,
            intervened: true,
            interventions: ["blocked_stop_that_would_breach_reserve", "raised_setpoint_to_cover_critical_load"],
          },
        }}
      />,
    );
    expect(screen.getByText("Blocked a stop that would breach reserve margin")).toBeInTheDocument();
    expect(screen.getByText("Raised a setpoint to cover critical load")).toBeInTheDocument();
    expect(screen.getByText("intervened this step")).toBeInTheDocument();
  });

  it("surfaces the deterministic fallback reason when one is active", () => {
    render(
      <SafetyPanel safety={{ ...maitriSafety, last_fallback_reason: "agent_exceeded_latency_budget" }} />,
    );
    expect(screen.getByText(/Deterministic fallback active/)).toBeInTheDocument();
    expect(screen.getByText(/agent_exceeded_latency_budget/)).toBeInTheDocument();
  });

  it("renders the real guard statistics, not placeholders", () => {
    render(<SafetyPanel safety={{ ...maitriSafety, steps: 42, max_latency_ms: 3.14159 }} />);
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("3.14 ms")).toBeInTheDocument();
  });
});
