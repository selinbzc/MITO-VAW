"""
Mito-Agent: GATK Mitochondria Pipeline Multi-Agent System

Workflow:
  Coordinator → Aligner → Commander → Geneticist → Validator

Each agent has a single responsibility:
  - Coordinator: FASTQ discovery, samplesheet construction
  - Aligner: FASTQ → BAM (BWA-MEM, hg38)
  - Commander: GATK pipeline → VCF
  - Geneticist: VCF parsing + RAG + LLM interpretation (gemma2:27b)
  - Validator: Report correction and finalization
"""
import os
import subprocess
from pathlib import Path
from typing import TypedDict, List, Optional
from datetime import datetime
 
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import ollama
 
# Modules
import align as aligner_module
import commander as commander_module
 
# RAG system
from rag_setup import (
    get_database, get_drug_database, get_mitotox_database,
    search as rag_search, search_drugs, search_mitotox
)
 
# === CONFIGURATION ===
PROJECT_ROOT = Path.home() / "Selin" / "mito_pipeline" / "gatk_pipeline"
FASTQ_DIR    = PROJECT_ROOT / "data" / "sra_samples"
BAM_DIR      = PROJECT_ROOT / "data" / "bam"
RESULTS_DIR  = PROJECT_ROOT / "results"
 
OLLAMA_MODEL = "gemma2:27b"
 
# Load RAG databases
print(">>> Loading RAG databases...")
rag_db     = get_database()
drug_db    = get_drug_database()
mitotox_db = get_mitotox_database()
print(">>> All RAG databases ready")
 
 
# ============================================================
# STATE
# ============================================================
 
class PipelineState(TypedDict):
    input_mode:  str
    fastq_dir:   Optional[str]
    bam_path:    Optional[str]
    sample_id:   Optional[str]
    aligned_bam: Optional[str]
    vcf_path:    Optional[str]
    variants:    List[dict]
    draft_report: str
    final_report: str
    error:       Optional[str]
 
 
# ============================================================
# UTILITY: VCF PARSING
# ============================================================
 
def parse_vcf(vcf_path: str) -> List[dict]:
    """Parse VCF and compute heteroplasmy levels using bcftools."""
    cmd = (
        f"bcftools query -f "
        f"'%CHROM\\t%POS\\t%REF\\t%ALT\\t%FILTER\\t[%AF]\\t[%DP]\\n' {vcf_path}"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
 
    variants = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 7:
            continue
 
        chrom, pos, ref, alt, filt = parts[0], parts[1], parts[2], parts[3], parts[4]
 
        # Multi-allelic: AF may be comma-separated
        af_raw = parts[5].split(',')[0] if parts[5] not in ['.', ''] else '0.0'
        af = float(af_raw) if af_raw not in ['.', ''] else 0.0
        dp = int(parts[6]) if parts[6] not in ['.', ''] else 0
 
        if filt != "PASS":
            continue
 
        notation = f"m.{pos}{ref}>{alt}"
        kb_info  = None
 
        # Layer 1: RAG exact match (score < 0.8 = reliable match)
        rag_results = rag_search(rag_db, notation, k=1)
        if rag_results and rag_results[0]["score"] < 0.8:
            best    = rag_results[0]
            kb_info = (
                f"{best['disease']}. Gene: {best['locus']}. "
                f"Mutation: {best['allele']}. "
                f"Max heteroplasmy: {best['max_het']}%."
            )
 
        # Layer 2: Semantic search (0.8–1.0 = probable match)
        if not kb_info:
            sem = rag_search(
                rag_db,
                f"mitochondrial position {pos} {ref} to {alt} disease",
                k=1
            )
            if sem and sem[0]["score"] < 1.0:
                best    = sem[0]
                kb_info = (
                    f"Similar known variant: {best['notation']} "
                    f"({best['disease']}). Gene: {best['locus']}."
                )
 
        # known = confirmed pathogenic match only (score < 0.8)
        is_known = kb_info is not None and "Similar known variant" not in str(kb_info)
 
        if not is_known:
            kb_info = (
                "Likely haplogroup polymorphism or variant of uncertain significance (VUS). "
                "Not found in MitoMap pathogenic database. "
                "Clinical significance unknown — further functional studies required."
            )
 
        variants.append({
            "notation":    notation,
            "chrom":       chrom,
            "pos":         int(pos),
            "ref":         ref,
            "alt":         alt,
            "heteroplasmy": af,
            "depth":       dp,
            "known":       is_known,
            "kb_info":     kb_info,
        })
 
    variants.sort(key=lambda x: x["heteroplasmy"], reverse=True)
    return variants
 
 
# ============================================================
# AGENT 1: COORDINATOR
# ============================================================
 
def coordinator_node(state: PipelineState) -> dict:
    print("\n" + "="*60)
    print("📋 AGENT 1: COORDINATOR")
    print("="*60)
 
    if state["input_mode"] == "bam":
        bam = Path(state["bam_path"])
        if not bam.exists():
            return {"error": f"BAM file not found: {bam}"}
 
        result = subprocess.run(
            ["samtools", "view", "-H", str(bam)],
            capture_output=True, text=True
        )
        sample_id = bam.name.replace(".sorted.bam", "").replace(".bam", "")
        for line in result.stdout.split('\n'):
            if line.startswith('@RG'):
                for field in line.split('\t'):
                    if field.startswith('SM:') and field[3:].strip():
                        sample_id = field[3:].strip()
                        break
 
        print(f"✓ BAM mode: {bam.name}")
        print(f"✓ Sample ID: {sample_id}")
        return {"aligned_bam": str(bam), "sample_id": sample_id, "error": None}
 
    else:
        fastq_dir = Path(state["fastq_dir"])
        if not fastq_dir.exists():
            return {"error": f"FASTQ directory not found: {fastq_dir}"}
 
        files = sorted(fastq_dir.glob("*_R1.fastq.gz"))
        if not files:
            return {"error": "No FASTQ files found"}
 
        samples = []
        for r1 in files:
            sid = r1.name.replace("_R1.fastq.gz", "")
            r2  = fastq_dir / f"{sid}_R2.fastq.gz"
            if r2.exists():
                samples.append({"id": sid, "r1": str(r1), "r2": str(r2)})
 
        print(f"✓ {len(samples)} sample(s) identified:")
        for s in samples:
            print(f"   • {s['id']}")
 
        return {
            "sample_id": samples[0]["id"] if samples else None,
            "error": None if samples else "No paired-end matches found",
        }
 
 
# ============================================================
# AGENT 2: ALIGNER
# ============================================================
 
def aligner_node(state: PipelineState) -> dict:
    print("\n" + "="*60)
    print("🧬 AGENT 2: ALIGNER (FASTQ → BAM)")
    print("="*60)
 
    if state["input_mode"] == "bam":
        print("   BAM mode — alignment step skipped")
        return {}
 
    fastq_dir = Path(state["fastq_dir"])
    r1 = fastq_dir / f"{state['sample_id']}_R1.fastq.gz"
    r2 = fastq_dir / f"{state['sample_id']}_R2.fastq.gz"
 
    result = aligner_module.align_sample(
        sample_id=state["sample_id"],
        fastq_r1=r1,
        fastq_r2=r2,
    )
 
    if not result.exists():
        return {"error": f"BAM file could not be generated: {result}"}
 
    print(f"✓ BAM ready: {result}")
    return {"aligned_bam": str(result)}
 
 
# ============================================================
# AGENT 3: COMMANDER
# ============================================================
 
def commander_node(state: PipelineState) -> dict:
    print("\n" + "="*60)
    print("⚙️  AGENT 3: COMMANDER (GATK Pipeline)")
    print("="*60)
 
    # Reuse existing VCF if available (for development purposes)
    sample_id = state.get("sample_id", "")
    existing_vcfs = sorted(
        RESULTS_DIR.rglob(f"{sample_id}.final.vcf.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if existing_vcfs:
        vcf = existing_vcfs[0]
        print(f"✓ Using existing VCF: {vcf}")
        return {"vcf_path": str(vcf), "error": None}
 
    bam_path = Path(state["aligned_bam"])
    try:
        vcf_path = commander_module.process_sample(bam_path)
        print(f"✓ VCF: {vcf_path}")
        return {"vcf_path": str(vcf_path), "error": None}
    except Exception as e:
        return {"error": f"GATK pipeline failed: {e}"}
 
 
# ============================================================
# AGENT 4: GENETICIST
# ============================================================
 
def geneticist_node(state: PipelineState) -> dict:
    print("\n" + "="*60)
    print("🩺 AGENT 4: GENETICIST")
    print("="*60)
 
    print("\n[1/3] Parsing VCF...")
    variants = parse_vcf(state["vcf_path"])
    print(f"   ✓ {len(variants)} PASS variants identified")
 
    if not variants:
        return {
            "variants": [],
            "draft_report": "No pathogenic variants detected in mitochondrial genome.",
        }
 
    known   = [v for v in variants if v["known"]]
    unknown = [v for v in variants if not v["known"]]
 
    print(f"\n[2/3] MitoMap matching: {len(known)}/{len(variants)} confirmed")
    for v in known:
        print(f"   ⚡ {v['notation']} (heteroplasmy {v['heteroplasmy']*100:.1f}%)")
 
    if unknown:
        print(f"   📋 {len(unknown)} variant(s) classified as haplogroup/VUS")
 
    # Variant list for prompt construction
    variant_lines = []
    for v in variants[:10]:
        line = (
            f"- {v['notation']}: heteroplasmy {v['heteroplasmy']*100:.1f}%, "
            f"depth {v['depth']}x | {v['kb_info']}"
        )
        variant_lines.append(line)
 
    # MitoTox: query potentially toxic drugs for confirmed diseases
    detected_diseases = list(set(
        v["kb_info"].split(".")[0] for v in variants[:5]
        if v.get("known") and v.get("kb_info")
    ))
 
    mitotox_lines = []
    drug_lines    = []
 
    for disease in detected_diseases[:3]:
        tox_results = search_mitotox(
            mitotox_db,
            f"{disease} mitochondrial toxicity contraindicated",
            k=3,
        )
        for r in tox_results:
            if r["score"] < 1.0:
                mitotox_lines.append(
                    f"- {r['drug']}: mitochondrial func {r['func_id']} "
                    f"({r['func_act']}). Side effects: {r['side_effects'][:80]}"
                )
 
        treat_results = search_drugs(drug_db, f"{disease} treatment", k=3)
        for r in treat_results:
            if r["score"] < 1.0:
                drug_lines.append(
                    f"- {r['drug']}: {r['disease']} | {r['stage']}"
                )
 
    print(f"\n[3/3] Generating Geneticist report (gemma2:27b)...")
 
    prompt = f"""You are a Senior Clinical Geneticist specializing in mitochondrial genetics.
 
TASK: Write a clinical interpretation report for the detected mitochondrial variants.
 
DETECTED VARIANTS (sorted by heteroplasmy level):
{chr(10).join(variant_lines)}
 
TOTAL VARIANTS: {len(variants)} PASS variants detected
CONFIRMED PATHOGENIC (MitoMap match): {len(known)} variants
HAPLOGROUP / VUS: {len(unknown)} variants
 
POTENTIALLY TOXIC DRUGS (MitoTox database):
{chr(10).join(mitotox_lines) if mitotox_lines else "No specific contraindications found in database."}
 
POTENTIAL THERAPEUTIC OPTIONS (OpenTargets clinical trials):
{chr(10).join(drug_lines) if drug_lines else "No specific treatments found in database."}
 
INSTRUCTIONS:
1. Write entirely in professional medical English
2. Base your interpretation ONLY on the provided data above — do not add information from outside
3. For each CONFIRMED PATHOGENIC variant, explain: gene, associated syndrome, heteroplasmy significance
4. Variants labeled as "haplogroup polymorphism or VUS" must NOT be interpreted as pathogenic — state they require further investigation only, do not speculate about disease associations
5. If no confirmed pathogenic variants are found, clearly state this is likely a normal mitochondrial haplogroup profile
6. If toxic drugs are listed, mention them as potential contraindications for patients with confirmed pathogenic variants only
7. If therapeutic options are listed, mention them as potential treatments
8. Do NOT make definitive diagnoses — provide literature-based interpretation only
9. Keep report concise: 4-5 paragraphs maximum
 
Write the clinical interpretation report:"""
 
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.1},
        )
        draft = response['message']['content']
    except Exception as e:
        draft = f"LLM error: {e}\n\nVariants found:\n" + '\n'.join(variant_lines)
 
    return {"variants": variants, "draft_report": draft}
 
 
# ============================================================
# AGENT 5: VALIDATOR
# ============================================================
 
def validator_node(state: PipelineState) -> dict:
    print("\n" + "="*60)
    print("✅ AGENT 5: VALIDATOR")
    print("="*60)
    print("   Reviewing and finalizing the report...")
 
    prompt = f"""You are a Medical Technical Editor specialized in Clinical Genomics.
 
TASK: Review and polish the following mitochondrial genetics clinical report.
 
DRAFT REPORT:
{state['draft_report']}
 
INSTRUCTIONS:
1. Correct any medical terminology errors
2. Ensure professional clinical tone throughout
3. Do NOT add information not present in the draft
4. Add this disclaimer at the end:
   "DISCLAIMER: This report is generated by an AI-assisted bioinformatics pipeline
   for research purposes only. Clinical decisions must be made by qualified medical
   professionals based on complete clinical context."
5. Return ONLY the final polished report — no meta-commentary
 
Final report:"""
 
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.1},
        )
        final = response['message']['content']
    except Exception as e:
        final = state['draft_report'] + "\n\nDISCLAIMER: Research use only."
 
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"clinical_report_{state['sample_id']}_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
 
    with open(report_path, 'w') as f:
        f.write("# Mitochondrial Variant Analysis Report\n\n")
        f.write(f"**Sample:** {state['sample_id']}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Pipeline:** GATK Mitochondria Pipeline v4.3.0.0\n\n")
        f.write("---\n\n")
        f.write(final)
 
    print(f"\n{'='*60}")
    print("📋 FINAL CLINICAL REPORT")
    print("="*60)
    print(final)
    print(f"\n✓ Report saved: {report_path}")
 
    return {"final_report": final}
 
 
# ============================================================
# ERROR HANDLER
# ============================================================
 
def error_node(state: PipelineState) -> dict:
    print("\n" + "="*60)
    print("❌ PIPELINE TERMINATED WITH ERROR")
    print("="*60)
    print(f"Error: {state.get('error')}")
    return {}
 
 
# ============================================================
# WORKFLOW
# ============================================================
 
def should_continue(state: PipelineState) -> str:
    return "error" if state.get("error") else "continue"
 
 
def skip_aligner(state: PipelineState) -> str:
    if state.get("error"):
        return "error"
    return "skip" if state["input_mode"] == "bam" else "align"
 
 
def build_workflow():
    workflow = StateGraph(PipelineState)
 
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("aligner",     aligner_node)
    workflow.add_node("commander",   commander_node)
    workflow.add_node("geneticist",  geneticist_node)
    workflow.add_node("validator",   validator_node)
    workflow.add_node("error",       error_node)
 
    workflow.set_entry_point("coordinator")
 
    workflow.add_conditional_edges(
        "coordinator", skip_aligner,
        {"align": "aligner", "skip": "commander", "error": "error"},
    )
    workflow.add_conditional_edges(
        "aligner", should_continue,
        {"continue": "commander", "error": "error"},
    )
    workflow.add_conditional_edges(
        "commander", should_continue,
        {"continue": "geneticist", "error": "error"},
    )
    workflow.add_conditional_edges(
        "geneticist", should_continue,
        {"continue": "validator", "error": "error"},
    )
 
    workflow.add_edge("validator", END)
    workflow.add_edge("error",     END)
 
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["commander"],
    )
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    print("\n🧬 Mito-Agent: GATK Mitochondria Multi-Agent Pipeline\n")
 
    app    = build_workflow()
    config = {"configurable": {"thread_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
 
    bam_files = sorted(BAM_DIR.glob("*.sorted.bam"))
 
    if bam_files:
        bam = bam_files[-1]
        print(f"📁 BAM mode: {bam.name}")
        initial: PipelineState = {
            "input_mode":  "bam",
            "fastq_dir":   None,
            "bam_path":    str(bam),
            "sample_id":   None,
            "aligned_bam": None,
            "vcf_path":    None,
            "variants":    [],
            "draft_report": "",
            "final_report": "",
            "error":       None,
        }
    else:
        print(f"📁 FASTQ mode: {FASTQ_DIR}")
        initial: PipelineState = {
            "input_mode":  "fastq",
            "fastq_dir":   str(FASTQ_DIR),
            "bam_path":    None,
            "sample_id":   None,
            "aligned_bam": None,
            "vcf_path":    None,
            "variants":    [],
            "draft_report": "",
            "final_report": "",
            "error":       None,
        }
 
    for event in app.stream(initial, config):
        pass
 
    state = app.get_state(config).values
    if state.get("error"):
        print(f"\n❌ Error during preparation stage: {state['error']}")
        return
 
    print("\n" + "="*60)
    print("🛑 CONFIRMATION REQUIRED")
    print("="*60)
    print(f"Sample: {state.get('sample_id')}")
    print(f"BAM: {state.get('aligned_bam')}")
    print("\nGATK Mitochondria Pipeline will now be executed.")
 
    confirm = input("Proceed? [Y/n]: ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("Aborted.")
        return
 
    for event in app.stream(None, config):
        pass
 
    print("\n✅ Mito-Agent completed successfully.")
 
 
if __name__ == "__main__":
    main()
