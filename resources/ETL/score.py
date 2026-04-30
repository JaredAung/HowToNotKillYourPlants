"""
Difficulty score for plants in permapeople_plants_mapped_uniform.json.

Score: 0–100, where higher = harder to care for.

Formula components:
  1. Zone flexibility (0–25): Narrow USDA zone = harder. "2-11" (9 zones) = easy, "9-11" (2) = hard.
  2. Light flexibility (0–20): Single option ("Full sun") = hard, multiple = easier.
  3. Water flexibility (0–15): Single option = hard, multiple = easier.
  4. Soil flexibility (0–15): Single type = hard, multiple = easier.
  5. Propagation complexity (0–15): Seed direct sow = easy, cuttings/division = hard.
  6. Special requirements (0–10): Cold stratification = +10, drought resistant = -10.
  7. Problems (0–10): Pests/Diseases/Warning present = harder.
  8. Data completeness (0–10): Missing core fields = harder (less guidance).

Run: python -m resources.score
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT.parent
INPUT = RESOURCES / "data_creating" / "permapeople_plants_mapped_uniform.json"
OUTPUT = RESOURCES / "data_creating" / "permapeople_plants_scored.json"


def _parse_zone_span(zone: str | None) -> int:
    """Parse '3-9' -> 6 (span). Returns 0 if invalid."""
    if not zone:
        return 0
    m = re.match(r"(\d+)\s*-\s*(\d+)", str(zone).strip())
    if m:
        return max(0, int(m.group(2)) - int(m.group(1)) + 1)
    return 0


def _count_options(s: str | None) -> int:
    """Count comma-separated options. 'Full sun, Partial' -> 2."""
    if not s or not str(s).strip():
        return 0
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return len(parts) if parts else 0


def _propagation_difficulty(method: str | None) -> int:
    """0=easy, 15=hard."""
    if not method:
        return 7  # unknown = medium
    m = str(method).lower()
    if "seed - direct sow" in m and "transplant" not in m:
        return 0
    if "seed - transplant" in m or "seed - direct sow" in m:
        return 3
    if "cuttings" in m:
        return 10
    if "division" in m:
        return 12
    return 5


def score_plant(plant: dict) -> tuple[float, dict]:
    """
    Compute difficulty score 0–100. Returns (score, breakdown).
    """
    ec = plant.get("environment_care") or {}
    prob = plant.get("problems") or {}

    # 1. Zone flexibility (0–25): wider = easier
    usda = ec.get("USDA Hardiness zone")
    if isinstance(usda, dict):
        zmin, zmax = usda.get("min"), usda.get("max")
    else:
        zmin, zmax = ec.get("usda_zone_min"), ec.get("usda_zone_max")
    if zmin is not None and zmax is not None:
        span = max(0, zmax - zmin + 1)
    else:
        span = _parse_zone_span(usda if isinstance(usda, str) else None)
    if span == 0:
        zone_pts = 25  # missing = assume hard
    else:
        zone_pts = max(0, 25 - span * 2.5)  # 10 zones -> 0, 2 zones -> 20

    # 2. Light flexibility (0–20)
    light_opts = _count_options(ec.get("Light requirement"))
    if light_opts == 0:
        light_pts = 20
    else:
        light_pts = max(0, 20 - light_opts * 5)  # 1 opt=20, 2=15, 3+=10

    # 3. Water flexibility (0–15)
    water_opts = _count_options(ec.get("Water requirement"))
    if water_opts == 0:
        water_pts = 15
    else:
        water_pts = max(0, 15 - water_opts * 4)

    # 4. Soil flexibility (0–15)
    soil_opts = _count_options(ec.get("Soil type"))
    if soil_opts == 0:
        soil_pts = 15
    else:
        soil_pts = max(0, 15 - soil_opts * 4)

    # 5. Propagation (0–15)
    prop_pts = min(15, _propagation_difficulty(ec.get("Propagation method")))

    # 6. Special requirements (0–10)
    spec_pts = 0
    if ec.get("Cold stratification temperature") or ec.get("Cold stratification time"):
        spec_pts += 8
    dr = ec.get("Drought resistant")
    if dr and str(dr).lower() in ("true", "yes"):
        spec_pts -= 5
    spec_pts = max(0, min(10, spec_pts))

    # 7. Problems (0–10)
    prob_pts = 0
    if prob.get("Pests"):
        prob_pts += 4
    if prob.get("Diseases"):
        prob_pts += 3
    if prob.get("Warning"):
        prob_pts += 3
    prob_pts = min(10, prob_pts)

    # 8. Data completeness (0–10): missing core = harder
    missing = sum(
        1
        for k in ["USDA Hardiness zone", "Light requirement", "Water requirement", "Soil type"]
        if not ec.get(k)
    )
    data_pts = min(10, missing * 3)

    total = zone_pts + light_pts + water_pts + soil_pts + prop_pts + spec_pts + prob_pts + data_pts
    total = min(100, max(0, total))

    breakdown = {
        "zone": round(zone_pts, 1),
        "light": round(light_pts, 1),
        "water": round(water_pts, 1),
        "soil": round(soil_pts, 1),
        "propagation": round(prop_pts, 1),
        "special": round(spec_pts, 1),
        "problems": round(prob_pts, 1),
        "data_completeness": round(data_pts, 1),
    }
    return round(total, 1), breakdown


def main():
    with open(INPUT) as f:
        data = json.load(f)

    def score_to_care_level(score: float) -> str:
        if score < 40:
            return "easy"
        if score < 60:
            return "medium"
        return "hard"

    plants = data.get("plants", [])
    for p in plants:
        score, breakdown = score_plant(p)
        p["difficulty_score"] = score
        p["difficulty_breakdown"] = breakdown
        p["care_level"] = score_to_care_level(score)

    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Scored {len(plants)} plants -> {OUTPUT}")

    # Also append care_level to mapped_uniform.json
    with open(INPUT) as f:
        uniform = json.load(f)
    uniform_plants = {p["id"]: p for p in uniform.get("plants", [])}
    for p in plants:
        uid = p.get("id")
        if uid in uniform_plants:
            uniform_plants[uid]["care_level"] = p["care_level"]
            uniform_plants[uid]["difficulty_score"] = p["difficulty_score"]
    with open(INPUT, "w") as f:
        json.dump(uniform, f, indent=2)
    print(f"Updated care_level in {INPUT}")

    # Show distribution
    scores = [p["difficulty_score"] for p in plants]
    print(f"\nScore distribution: min={min(scores):.1f}, max={max(scores):.1f}, mean={sum(scores)/len(scores):.1f}")
    easy = sum(1 for s in scores if s < 40)
    med = sum(1 for s in scores if 40 <= s < 60)
    hard = sum(1 for s in scores if s >= 60)
    print(f"Easy (<40): {easy}, Medium (40-59): {med}, Hard (60+): {hard}")


if __name__ == "__main__":
    main()
