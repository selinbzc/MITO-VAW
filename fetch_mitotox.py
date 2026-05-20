import requests, csv, time
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://www.mitotox.org/api"
OUT  = Path.home() / "Selin/mito_pipeline/gatk_pipeline/knowledge_base/mitotox_drugs.csv"

def fetch_all(endpoint):
    results, url, page = [], f"{BASE}/{endpoint}/list", 1
    while url:
        print(f"   [{endpoint}] page {page}...", end="\r")
        resp = requests.get(url, timeout=30)
        data = resp.json()
        results.extend(data["results"])
        url = data.get("next")
        page += 1
        time.sleep(0.2)
    print(f"   [{endpoint}] ✓ {len(results)} records")
    return results

print(">>> Fetching MitoTox data...")

compounds = fetch_all("compounds")
records   = fetch_all("records")
functions = fetch_all("functions")
targets   = fetch_all("targets")

# Index mappings
comp_map = {c["compound_ID"]: c for c in compounds}
func_map = {f["func_id"]: f for f in functions}
targ_map = {t["targ_id"]: t for t in targets}

print("\n>>> Merging records...")
rows = []
for rec in records:
    cid  = rec.get("compound", "").split(":")[0].strip()
    comp = comp_map.get(cid, {})
    
    fid  = rec.get("func", "")
    func = func_map.get(fid, {})
    
    tid  = rec.get("targ") or ""
    targ = targ_map.get(tid, {}) if tid else {}
    
    drug_name = comp.get("name", "")
    if not drug_name:
        continue
    
    rows.append({
        "drug_name":        drug_name,
        "trade_name":       comp.get("trade_name", ""),
        "drug_group":       comp.get("group", ""),
        "atc_class":        comp.get("classification", ""),
        "indications":      comp.get("indications", ""),
        "side_effects":     comp.get("side_effects", ""),
        "description":      comp.get("description", ""),
        "func_id":          fid,
        "func_category":    func.get("category", ""),
        "func_name":        func.get("name", ""),
        "func_action":      rec.get("func_act", ""),
        "target_name":      targ.get("name", ""),
        "target_gene":      targ.get("gene_symbol", ""),
        "dose":             rec.get("dose", ""),
        "species":          rec.get("species", ""),
        "animal_model":     rec.get("animal_model", ""),
        "method":           rec.get("method", ""),
    })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✓ {len(rows)} records → {OUT}")

from collections import Counter
cats = Counter(r["func_category"] for r in rows if r["func_category"])
print("\n=== Mitochondrial function distribution ===")
for cat, count in cats.most_common():
    print(f"  {count:4d}x  {cat}")

groups = Counter(r["drug_group"] for r in rows)
print("\n=== Drug group distribution ===")
for g, c in groups.most_common():
    print(f"  {c:4d}x  {g}")
