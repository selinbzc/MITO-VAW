import requests, csv
from pathlib import Path

API = "https://api.platform.opentargets.org/api/v4/graphql"
OUT = Path.home() / "Selin/mito_pipeline/gatk_pipeline/knowledge_base/opentargets_drugs.csv"

# Step 1: Retrieve all diseases matching the "mitochondrial" keyword
search_query = """
{
  search(queryString: "%s", entityNames: ["disease"], page: {size: 50, index: %d}) {
    total
    hits { id name }
  }
}
"""

keywords = [
    "mitochondrial disease",
    "mitochondrial myopathy",
    "mitochondrial encephalopathy",
    "mitochondrial complex deficiency",
    "MELAS", "MERRF", "LHON", "Leigh", "NARP",
    "Kearns-Sayre", "CPEO ophthalmoplegia",
    "mitochondrial cardiomyopathy",
    "mitochondrial neuropathy",
    "mitochondrial cytopathy",
]

print(">>> Searching for mitochondrial diseases...")
all_diseases = {}

for keyword in keywords:
    for page in range(3):  # 3 pages per keyword
        resp = requests.post(API, json={"query": search_query % (keyword, page)})
        data = resp.json()
        if "errors" in data:
            break
        hits = data["data"]["search"]["hits"]
        if not hits:
            break
        for h in hits:
            if h["id"] not in all_diseases:
                all_diseases[h["id"]] = h["name"]

print(f"   ✓ {len(all_diseases)} unique diseases identified")

# Step 2: Retrieve drug data for each disease
drug_query = """{
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

print(">>> Fetching drug data...")
rows = []
seen = set()

for mondo_id, disease_name in all_diseases.items():
    resp = requests.post(API, json={"query": drug_query % mondo_id})
    data = resp.json()
    
    if "errors" in data:
        continue
    
    candidates = data["data"]["disease"]["drugAndClinicalCandidates"]
    if candidates["count"] == 0:
        continue
    
    print(f"   ✓ {disease_name}: {candidates['count']} drugs")
    
    for row in candidates["rows"]:
        drug = row.get("drug") or {}
        drug_name = drug.get("name") or row.get("id", "?")
        stage = row.get("maxClinicalStage", "UNKNOWN")
        
        key = f"{drug_name}_{mondo_id}"
        if key in seen:
            continue
        seen.add(key)
        
        moa_rows = (drug.get("mechanismsOfAction") or {}).get("rows") or []
        mechanisms = list(set(m["mechanismOfAction"] for m in moa_rows if m.get("mechanismOfAction")))
        targets = list(set(t["approvedName"] for m in moa_rows for t in m.get("targets", []) if t.get("approvedName")))
        
        rows.append({
            "disease":    disease_name,
            "mondo_id":   mondo_id,
            "drug":       drug_name,
            "stage":      stage,
            "mechanisms": "; ".join(mechanisms),
            "targets":    "; ".join(targets),
        })

# Write to CSV
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["disease","mondo_id","drug","stage","mechanisms","targets"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'='*60}")
print(f"Total {len(rows)} drug-disease records → {OUT}")
print(f"{'='*60}")

# Summary
from collections import Counter
disease_counts = Counter(r["disease"] for r in rows)
for disease, count in disease_counts.most_common():
    print(f"  {count:3d} drugs  |  {disease}")
