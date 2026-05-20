"""
RAG Setup: MitoMap CSV + OpenTargets drug data → ChromaDB vector database.

Two separate ChromaDB collections:
  1. chroma_db/       → MitoMap variant database (332 pathogenic variants)
  2. chroma_db_drugs/ → OpenTargets drug database (LHON, Leigh, etc.)

Usage:
  python rag_setup.py          # Build both databases

Imported by main.py:
  from rag_setup import get_database, get_drug_database, search, search_drugs
"""

import csv
import re
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# === CONFIGURATION ===
PROJECT_ROOT    = Path.home() / "Selin" / "mito_pipeline" / "gatk_pipeline"
CSV_FILE        = PROJECT_ROOT / "knowledge_base" / "mitomap_veri.csv"
DRUG_CSV        = PROJECT_ROOT / "knowledge_base" / "opentargets_drugs.csv"
CHROMA_DIR      = PROJECT_ROOT / "knowledge_base" / "chroma_db"
DRUG_CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db_drugs"
EMBED_MODEL     = "all-MiniLM-L6-v2"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _safe_get(row: dict, key: str) -> str:
    """Safely retrieve a CSV column value."""
    return str(row.get(key, "") or "").strip()


def _allele_to_notation(allele: str) -> str:
    """
    Convert MitoMap allele format to standard m.POS REF>ALT VCF notation.

    Supported formats:
      T582C              → m.582T>C        (SNV)
      C960del            → m.960delC       (deletion)
      A12172AA           → m.12172A>AA     (insertion)
      G16033TCTCT...     → m.16033G>TCTCT  (complex)
    """
    allele = str(allele).strip()

    if not allele:
        return "unknown"

    # Already in m. format — return as-is
    if allele.startswith("m."):
        return allele

    # SNV: X123Y → m.123X>Y
    m = re.match(r"^([ACGT])(\d+)([ACGT])$", allele, re.I)
    if m:
        ref, pos, alt = m.group(1).upper(), m.group(2), m.group(3).upper()
        return f"m.{pos}{ref}>{alt}"

    # Deletion: X123del → m.123delX
    m = re.match(r"^([ACGT])(\d+)del$", allele, re.I)
    if m:
        ref, pos = m.group(1).upper(), m.group(2)
        return f"m.{pos}del{ref}"

    # Insertion / complex: X123XX+ → m.123X>XX+
    m = re.match(r"^([ACGT])(\d+)([ACGT]{2,})$", allele, re.I)
    if m:
        ref, pos, alt = m.group(1).upper(), m.group(2), m.group(3).upper()
        return f"m.{pos}{ref}>{alt}"

    # Unrecognized format: prepend position if present
    pos_m = re.search(r"\d+", allele)
    if pos_m:
        return f"m.{allele}"
    return allele


def _is_disease_row(row: dict) -> bool:
    """Filter out non-disease rows and entries with Disease_Binary=0."""
    disease = _safe_get(row, "Disease").lower()
    disease_binary = _safe_get(row, "Disease_Binary")

    if disease in ["", "non-disease", "non disease", "nondisease"]:
        return False
    if disease_binary == "0":
        return False
    return True


# ============================================================
# MITOMAP DATABASE
# ============================================================

def _parse_mitomap_csv() -> list[Document]:
    """Parse the MitoMap CSV into a list of LangChain Documents."""
    docs = []
    skipped = 0

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_disease_row(row):
                skipped += 1
                continue

            locus    = _safe_get(row, "Locus")
            disease  = _safe_get(row, "Disease")
            allele   = _safe_get(row, "Allele")
            mut_type = _safe_get(row, "Mutation_Type")
            domain   = _safe_get(row, "Domain")
            max_het  = _safe_get(row, "Maximum Heteroplasmy")
            hom_het  = _safe_get(row, "Homoplazmik/Heteroplazmik Durum")
            region   = _safe_get(row, "Region_Category")
            phylop   = _safe_get(row, "PhyloP")
            position = _safe_get(row, "Position")

            notation = _allele_to_notation(allele)

            text = (
                f"Variant: {notation}\n"
                f"Position: {position}\n"
                f"Gene/Locus: {locus}\n"
                f"Disease: {disease}\n"
                f"Allele change: {allele}\n"
                f"Mutation type: {mut_type}\n"
                f"Domain: {domain}\n"
                f"Maximum heteroplasmy: {max_het}%\n"
                f"Homoplasmic/Heteroplasmic: {hom_het}\n"
                f"Region: {region}\n"
                f"PhyloP conservation score: {phylop}\n"
                f"Summary: {notation} variant in {locus}, associated with {disease}. "
                f"Mutation type: {mut_type}. Domain: {domain}."
            )

            docs.append(Document(
                page_content=text,
                metadata={
                    "notation": notation,
                    "position": position,
                    "locus":    locus,
                    "disease":  disease,
                    "allele":   allele,
                    "mut_type": mut_type,
                    "domain":   domain,
                    "max_het":  max_het,
                    "hom_het":  hom_het,
                    "region":   region,
                    "phylop":   phylop,
                },
            ))

    print(f"   ✓ Non-disease rows skipped: {skipped}")
    return docs


def build_database() -> Chroma:
    """Build a ChromaDB vector store from the MitoMap CSV."""
    print(">>> Building MitoMap RAG database...")

    docs = _parse_mitomap_csv()
    print(f"   ✓ {len(docs)} pathogenic variants parsed")

    if not docs:
        raise ValueError("No documents could be created. Please verify the CSV file.")

    print(f"   ✓ Loading embeddings ({EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"   ✓ Ingesting into ChromaDB (batch=500)...")
    vectordb = None
    for i in range(0, len(docs), 500):
        batch = docs[i:i + 500]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectordb.add_documents(batch)
        print(f"   ... {min(i + 500, len(docs))}/{len(docs)}")

    print(f"   ✓ MitoMap ChromaDB ready ({vectordb._collection.count()} documents)")
    return vectordb


def load_database() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)


def get_database() -> Chroma:
    """Load existing database if available; otherwise build from scratch."""
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(">>> Loading existing MitoMap RAG database...")
        return load_database()
    return build_database()


def search(db: Chroma, query: str, k: int = 3) -> list[dict]:
    """Semantic search over the MitoMap variant database."""
    results = db.similarity_search_with_score(query, k=k)
    return [
        {
            "notation": doc.metadata.get("notation", ""),
            "position": doc.metadata.get("position", ""),
            "locus":    doc.metadata.get("locus", ""),
            "disease":  doc.metadata.get("disease", ""),
            "allele":   doc.metadata.get("allele", ""),
            "mut_type": doc.metadata.get("mut_type", ""),
            "domain":   doc.metadata.get("domain", ""),
            "max_het":  doc.metadata.get("max_het", ""),
            "content":  doc.page_content,
            "score":    score,
        }
        for doc, score in results
    ]


# ============================================================
# OPENTARGETS DRUG DATABASE
# ============================================================

def _parse_drug_csv() -> list[Document]:
    """Parse the OpenTargets drug CSV into a list of Documents."""
    docs = []

    with open(DRUG_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease  = _safe_get(row, "disease")
            drug     = _safe_get(row, "drug")
            stage    = _safe_get(row, "stage")
            mechs    = _safe_get(row, "mechanisms")
            targets  = _safe_get(row, "targets")

            if not drug:
                continue

            text = (
                f"Drug: {drug}\n"
                f"Indication (disease): {disease}\n"
                f"Clinical stage: {stage}\n"
                f"Mechanism of action: {mechs}\n"
                f"Molecular targets: {targets}\n"
                f"Summary: {drug} is investigated for {disease} treatment. "
                f"Clinical stage: {stage}. "
                f"Mechanism: {mechs}. "
                f"Targets: {targets}."
            )

            docs.append(Document(
                page_content=text,
                metadata={
                    "drug":    drug,
                    "disease": disease,
                    "stage":   stage,
                    "mechs":   mechs,
                    "targets": targets,
                },
            ))

    return docs


def build_drug_database() -> Chroma:
    """Build a ChromaDB vector store from the OpenTargets drug data."""
    print(">>> Building OpenTargets drug RAG database...")

    if not DRUG_CSV.exists():
        raise FileNotFoundError(
            f"Drug CSV not found: {DRUG_CSV}\n"
            "Run first: python opentargets.py"
        )

    docs = _parse_drug_csv()
    print(f"   ✓ {len(docs)} drug records parsed")

    print(f"   ✓ Loading embeddings ({EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if DRUG_CHROMA_DIR.exists():
        shutil.rmtree(DRUG_CHROMA_DIR)
    DRUG_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(DRUG_CHROMA_DIR),
    )

    print(f"   ✓ Drug ChromaDB ready ({vectordb._collection.count()} documents)")
    return vectordb


def load_drug_database() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=str(DRUG_CHROMA_DIR), embedding_function=embeddings)


def get_drug_database() -> Chroma:
    """Load existing database if available; otherwise build from scratch."""
    if DRUG_CHROMA_DIR.exists() and any(DRUG_CHROMA_DIR.iterdir()):
        print(">>> Loading existing drug RAG database...")
        return load_drug_database()
    return build_drug_database()


def search_drugs(db: Chroma, query: str, k: int = 3) -> list[dict]:
    """Semantic search over the drug database."""
    results = db.similarity_search_with_score(query, k=k)
    return [
        {
            "drug":    doc.metadata.get("drug", ""),
            "disease": doc.metadata.get("disease", ""),
            "stage":   doc.metadata.get("stage", ""),
            "mechs":   doc.metadata.get("mechs", ""),
            "targets": doc.metadata.get("targets", ""),
            "score":   score,
        }
        for doc, score in results
    ]

# ============================================================
# MITOTOX DRUG TOXICITY DATABASE
# ============================================================

MITOTOX_CSV        = PROJECT_ROOT / "knowledge_base" / "mitotox_drugs.csv"
MITOTOX_CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db_mitotox"


def _parse_mitotox_csv() -> list[Document]:
    """Parse the MitoTox CSV into a list of Documents."""
    docs = []
    with open(MITOTOX_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drug    = _safe_get(row, "drug_name")
            if not drug:
                continue

            trade     = _safe_get(row, "trade_name")
            group     = _safe_get(row, "drug_group")
            atc       = _safe_get(row, "atc_class")
            indicate  = _safe_get(row, "indications")
            side_fx   = _safe_get(row, "side_effects")
            desc      = _safe_get(row, "description")
            func_id   = _safe_get(row, "func_id")
            func_cat  = _safe_get(row, "func_category") if "func_category" in row else ""
            func_name = _safe_get(row, "func_name") if "func_name" in row else ""
            func_act  = _safe_get(row, "func_action") if "func_action" in row else _safe_get(row, "func_act")
            dose      = _safe_get(row, "dose")
            species   = _safe_get(row, "species")
            model     = _safe_get(row, "animal_model")
            method    = _safe_get(row, "method")

            text = (
                f"Drug: {drug}\n"
                f"Trade name: {trade}\n"
                f"Drug group: {group}\n"
                f"ATC classification: {atc}\n"
                f"Clinical indications: {indicate}\n"
                f"Side effects: {side_fx}\n"
                f"Description: {desc}\n"
                f"Mitochondrial function affected (ID): {func_id}\n"
                f"Function action: {func_act}\n"
                f"Experimental dose: {dose}\n"
                f"Species: {species}\n"
                f"Model: {model}\n"
                f"Method: {method}\n"
                f"Summary: {drug} ({trade}) affects mitochondrial function {func_id} "
                f"({func_act}). "
                f"Clinical use: {indicate}. "
                f"Side effects include: {side_fx}. "
                f"Description: {desc}."
            )

            docs.append(Document(
                page_content=text,
                metadata={
                    "drug":       drug,
                    "trade":      trade,
                    "group":      group,
                    "atc":        atc,
                    "indications": indicate,
                    "side_effects": side_fx,
                    "func_id":    func_id,
                    "func_act":   func_act,
                    "model":      model,
                }
            ))
    return docs


def build_mitotox_database() -> Chroma:
    """Build a ChromaDB vector store from the MitoTox drug toxicity data."""
    import shutil as _shutil

    print(">>> Building MitoTox drug toxicity database...")

    if not MITOTOX_CSV.exists():
        raise FileNotFoundError(f"MitoTox CSV not found: {MITOTOX_CSV}\nRun first: python fetch_mitotox.py")

    docs = _parse_mitotox_csv()
    print(f"   ✓ {len(docs)} records parsed")

    print(f"   ✓ Loading embeddings ({EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if MITOTOX_CHROMA_DIR.exists():
        _shutil.rmtree(MITOTOX_CHROMA_DIR)
    MITOTOX_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"   ✓ Ingesting into ChromaDB (batch=500)...")
    vectordb = None
    for i in range(0, len(docs), 500):
        batch = docs[i:i + 500]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(MITOTOX_CHROMA_DIR),
            )
        else:
            vectordb.add_documents(batch)
        print(f"   ... {min(i+500, len(docs))}/{len(docs)}")

    print(f"   ✓ MitoTox ChromaDB ready ({vectordb._collection.count()} documents)")
    return vectordb


def load_mitotox_database() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=str(MITOTOX_CHROMA_DIR), embedding_function=embeddings)


def get_mitotox_database() -> Chroma:
    """Load existing database if available; otherwise build from scratch."""
    if MITOTOX_CHROMA_DIR.exists() and any(MITOTOX_CHROMA_DIR.iterdir()):
        print(">>> Loading existing MitoTox database...")
        return load_mitotox_database()
    return build_mitotox_database()


def search_mitotox(db: Chroma, query: str, k: int = 3) -> list[dict]:
    """Semantic search over the MitoTox drug toxicity database."""
    results = db.similarity_search_with_score(query, k=k)
    return [
        {
            "drug":        doc.metadata.get("drug", ""),
            "trade":       doc.metadata.get("trade", ""),
            "indications": doc.metadata.get("indications", ""),
            "side_effects": doc.metadata.get("side_effects", ""),
            "func_id":     doc.metadata.get("func_id", ""),
            "func_act":    doc.metadata.get("func_act", ""),
            "model":       doc.metadata.get("model", ""),
            "content":     doc.page_content,
            "score":       score,
        }
        for doc, score in results
    ]

# ============================================================
# MAIN — Test
# ============================================================

if __name__ == "__main__":
    # MitoMap database
    db = build_database()

    print("\n>>> MitoMap test queries:")
    mitomap_tests = [
        "m.3243A>G MELAS",
        "LHON optic neuropathy",
        "tRNA myoclonic epilepsy MERRF",
        "Leigh syndrome complex I",
        "MT-ND4 substitution",
    ]
    for q in mitomap_tests:
        results = search(db, q, k=2)
        print(f"\n  '{q}'")
        for r in results:
            print(f"    → {r['notation']} | {r['locus']} | {r['disease']} (score: {r['score']:.3f})")

    print("\n" + "=" * 60)

    # OpenTargets drug database
    drug_db = build_drug_database()

    print("\n>>> Drug test queries:")
    drug_tests = [
        "LHON treatment optic neuropathy",
        "mitochondrial complex inhibitor",
        "Leigh syndrome therapy coenzyme",
        "idebenone mechanism",
    ]
    for q in drug_tests:
        results = search_drugs(drug_db, q, k=2)
        print(f"\n  '{q}'")
        for r in results:
            print(f"    → {r['drug']} | {r['disease']} | {r['stage']} (score: {r['score']:.3f})")
            
    print("\n" + "=" * 60)
    # MitoTox database
    mitotox_db = build_mitotox_database()

    print("\n>>> MitoTox test queries:")
    mitotox_tests = [
        "MELAS mitochondrial complex I toxicity",
        "mtDNA damage antiretroviral",
        "transmembrane potential uncoupler",
        "LHON optic neuropathy drug",
        "mitochondrial DNA inhibition antibiotic",
    ]
    for q in mitotox_tests:
        results = search_mitotox(mitotox_db, q, k=2)
        print(f"\n  '{q}'")
        for r in results:
           print(f"    → {r['drug']} | func: {r['func_id']} | {r['func_act']} (score: {r['score']:.3f})")
