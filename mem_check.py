import sys
import json
from datetime import datetime, timezone


if len(sys.argv) > 1:
    LOG_FILE = sys.argv[1]
else:
    LOG_FILE = "./unified.log"

def gb(b: int) -> float:
    return b / 1_073_741_824

def get_post(data):
    ts = datetime.now(timezone.utc).isoformat()
    data = json.dumps(parse_snapshot(data))
    desc = f"\n>>> [MEM] {ts} {data} <<<\n"
    return desc

def write_to_file(data):
    post = get_post(data)
    with open(LOG_FILE, "a", encoding="UTF-8") as file:
        file.write(post)


def parse_snapshot(data):
    out = {}

    # Host RAM
    mem = data.get("system_data", {}).get("memory_info", {})
    out["host_mem_used_gb"] = round(gb(mem.get("memory_used_bytes", 0)), 2)
    out["swap_used_gb"]     = round(gb(mem.get("swap_used_bytes", 0)), 2)

    # Average CPU usage
    vcpu = data.get("system_data", {}).get("vcpu_usage", {}).get("average_usage", {})
    out["cpu_user_pct"]   = vcpu.get("user",    0)
    out["cpu_system_pct"] = vcpu.get("system",  0)

    # HBM per NeuronCore
    hbm = {}
    util = {}
    for rt in data.get("neuron_runtime_data", []):
        report = rt.get("report", {})

        core_usage = (
            report.get("memory_used", {})
            .get("neuron_runtime_used_bytes", {})
            .get("usage_breakdown", {})
            .get("neuroncore_memory_usage", {})
        )
        for core_idx, cats in core_usage.items():
            hbm[core_idx] = hbm.get(core_idx, 0) + sum(int(v) for v in cats.values())

        for core_idx, v in report.get("neuroncore_counters", {}).get("neuroncores_in_use", {}).items():
            util[core_idx] = round(v.get("neuroncore_utilization", 0), 1)

        if not out.get("exec_lat_p50_ms"):
            lat = report.get("execution_stats", {}).get("latency_stats", {}).get("total_latency") or {}
            if lat:
                out["exec_lat_p50_ms"] = round(lat.get("p50", 0) * 1000, 2)
                out["exec_lat_p99_ms"] = round(lat.get("p99", 0) * 1000, 2)

    out["hbm_gb"] = {k: round(gb(v), 2) for k, v in sorted(hbm.items())}
    out["neuroncore_util_pct"] = dict(sorted(util.items()))
    return out


def main():
    try:
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            data = json.loads(raw_line)
            write_to_file(data)
    except:
        pass


if __name__ == "__main__":
    main()