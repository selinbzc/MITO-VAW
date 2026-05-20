import requests, csv
from pathlib import Path

API = "https://api.platform.opentargets.org/api/v4/graphql"

DISEASES = {
    "MELAS":          "MONDO_0010789",
    "MERRF":          "MONDO_0010790",
    "LHON":           "MONDO_0010788",
    "Leigh syndrome": "MONDO_0009723",
    "NARP":           "MONDO_0010794",
}

QUERY = """{
  disease(efoId: "%s") {
    name
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          name
          mechanismsOfAction {
            rows {
              mechanismOfAction
              targets { approvedName }
            }
          }
        }
      }
    }
  }
}"""

OUT = Path.home() / "Selin/mito_pipeline/gatk_pipeline/knowledge_base/opentargets_drugs.csv"

rows = []
for disease_name, mondo_id in DISEASES.items():
    resp = requests.post(API, json={"query": QUERY % mondo_id})
    data = resp.json()
    if "errors" in data:
        print(f"X {disease_name}: {data['errors'][0]['message'][:60]}")
        continue
    candidates = data["data"]["disease"]["drugAndClinicalCandidates"]
    print(f"OK {disease_name}: {candidates['count']} drugs")
    for row in candidates["rows"]:
        drug = row.get("drug")
        if drug is None:
            drug = {}
        drug_name = drug.get("name") or row.get("id", "?")
        stage = row.get("maxClinicalStage", "UNKNOWN")
        moa_data = drug.get("mechanismsOfAction") or {}
        moa_rows = moa_data.get("rows") or []        
        mechanisms = list(set(
            m["mechanismOfAction"] for m in moa_rows
            if m.get("mechanismOfAction")
        ))
        targets = list(set(
            t["approvedName"]
            for m in moa_rows
            for t in m.get("targets", [])
            if t.get("approvedName")
        ))
        rows.append({
            "disease":    disease_name,
            "drug":       drug_name,
            "stage":      stage,
            "mechanisms": "; ".join(mechanisms),
            "targets":    "; ".join(targets),
        })

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["disease","drug","stage","mechanisms","targets"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nTotal {len(rows)} records → {OUT}")
for r in rows:
    print(f"  {r['disease']:15s} | {r['drug']:30s} | {r['stage']}")
