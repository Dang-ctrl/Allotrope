import type { Observation } from "../api/types";
import { StatCard } from "./StatCard";

export function PowerBalance({ observation }: { observation: Observation }) {
  const gensetKw = observation.genset_power_kw.reduce((a, b) => a + b, 0);
  const renewableAvailableKw = observation.pv_available_kw + observation.wind_available_kw;
  const supplyKw = gensetKw + renewableAvailableKw;
  const balance = supplyKw - observation.electrical_load_kw;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <StatCard label="Electrical load" value={observation.electrical_load_kw.toFixed(1)} unit="kW" />
      <StatCard
        label="Critical load"
        value={observation.critical_load_kw.toFixed(1)}
        unit="kW"
      />
      <StatCard label="Genset output" value={gensetKw.toFixed(1)} unit="kW" />
      <StatCard
        label="Renewables available"
        value={renewableAvailableKw.toFixed(1)}
        unit="kW"
        hint={`PV ${observation.pv_available_kw.toFixed(1)} + wind ${observation.wind_available_kw.toFixed(1)}`}
      />
      <StatCard
        label="Indoor temp"
        value={observation.indoor_temp_c.toFixed(1)}
        unit="°C"
        tone={observation.indoor_temp_c < 15 ? "warn" : "default"}
      />
      <StatCard label="Air temp" value={observation.air_temp_c.toFixed(1)} unit="°C" />
      <StatCard label="Wind speed" value={observation.wind_speed_ms.toFixed(1)} unit="m/s" />
      <StatCard
        label="Supply margin"
        value={`${balance >= 0 ? "+" : ""}${balance.toFixed(1)}`}
        unit="kW"
        tone={balance < 0 ? "crit" : "default"}
      />
    </div>
  );
}
