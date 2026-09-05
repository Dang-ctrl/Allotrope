import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GensetFleet } from "./GensetFleet";
import { maitriDetail, maitriState } from "../test/fixtures";

describe("GensetFleet", () => {
  it("renders one card per genset with real online/offline state", () => {
    render(
      <GensetFleet gensets={maitriDetail.gensets} observation={maitriState.observation} />,
    );

    expect(screen.getByText("G1")).toBeInTheDocument();
    expect(screen.getByText("G2")).toBeInTheDocument();
    expect(screen.getByText("G3")).toBeInTheDocument();

    // G2 is online in the fixture; G1 and G3 are not.
    const pills = screen.getAllByText(/online|offline/);
    expect(pills.filter((p) => p.textContent === "online")).toHaveLength(1);
    expect(pills.filter((p) => p.textContent === "offline")).toHaveLength(2);
  });

  it("shows the running genset's power against its rating", () => {
    render(
      <GensetFleet gensets={maitriDetail.gensets} observation={maitriState.observation} />,
    );
    expect(screen.getByText("58")).toBeInTheDocument(); // 57.61 rounded
    expect(screen.getAllByText("/ 125 kW")).toHaveLength(3); // all three gensets rate at 125 kW
  });
});
