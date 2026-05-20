"""
Commander: GATK Mitochondria Pipeline (pure Python implementation).

No WDL/miniwdl/Docker required — all GATK commands are invoked via subprocess.

Workflow:
  1. SubsetBamToChrM      - Extract chrM reads using samtools
  2. BAM → FASTQ          - samtools fastq (Solexa-compatible)
  3. AlignToMt            - BWA-MEM alignment to standard chrM
  4. AlignToShiftedMt     - BWA-MEM alignment to shifted chrM
  5. Mutect2 (normal)     - Mitochondrial mode variant calling
  6. Mutect2 (shifted)    - Mitochondrial mode variant calling
  7. FilterMutectCalls    - Filter normal and shifted VCFs
  8. LiftoverVcf          - Convert shifted coordinates back to standard chrM
  9. MergeVcfs            - Generate final merged VCF
"""
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# === CONFIGURATION ===
PROJECT_ROOT    = Path.home() / "Selin" / "mito_pipeline" / "gatk_pipeline"
REF_DIR         = PROJECT_ROOT / "references"
RESULTS_DIR     = PROJECT_ROOT / "results"
BAM_DIR         = PROJECT_ROOT / "data" / "bam"

HG38_FASTA      = REF_DIR / "Homo_sapiens_assembly38.fasta"
MT_FASTA        = REF_DIR / "Homo_sapiens_assembly38.chrM.fasta"
MT_SHIFTED_FASTA= REF_DIR / "Homo_sapiens_assembly38.chrM.shifted_by_8000_bases.fasta"
SHIFT_BACK_CHAIN= REF_DIR / "ShiftBack.chain"
BLACKLIST_BED   = REF_DIR / "blacklist_sites.hg38.chrM.bed"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def run(cmd: list, description: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Execute a command and raise RuntimeError on failure."""
    print(f"\n   → {description}")
    print(f"   $ {' '.join(str(c) for c in cmd)}")

    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd)

    if result.returncode != 0:
        err = result.stderr if capture else "(stderr direct to console)"
        raise RuntimeError(f"❌ {description} FAILED (exit code {result.returncode})\n{err}")

    return result


def get_sample_name(bam_path: Path) -> str:
    """Automatically retrieve the SM field from the BAM header."""
    result = subprocess.run(
        ["samtools", "view", "-H", str(bam_path)],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if line.startswith('@RG'):
            for field in line.split('\t'):
                if field.startswith('SM:') and field[3:].strip():
                    sm = field[3:].strip()
                    print(f"   ✓ Sample name retrieved from BAM header: {sm}")
                    return sm

    name = bam_path.name.replace(".sorted.bam", "").replace(".bam", "")
    print(f"   ⚠ SM field not found; sample name derived from filename: {name}")
    return name


def _bwa_align(
    fq1: Path,
    fq2: Path,
    reference: Path,
    work_dir: Path,
    output_name: str,
    sample_name: str,
) -> Path:
    """
    FASTQ → sorted and indexed BAM (BWA-MEM + samtools).
    Does not use RevertSam / MergeBamAlignment — avoids Solexa encoding issues.
    """
    raw_bam    = work_dir / f"{output_name}_raw.bam"
    sorted_bam = work_dir / f"{output_name}.sorted.bam"

    rg = (
        f"@RG\\tID:{sample_name}_lane1"
        f"\\tSM:{sample_name}"
        f"\\tLB:{sample_name}_lib1"
        f"\\tPL:ILLUMINA"
    )

    # BWA-MEM
    bwa_cmd = (
        f"bwa mem -t 4 -R '{rg}' "
        f"{reference} {fq1} {fq2} "
        f"| samtools view -b -o {raw_bam} -"
    )
    print(f"\n   → BWA-MEM ({output_name})")
    print(f"   $ {bwa_cmd}")
    result = subprocess.run(bwa_cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"BWA-MEM failed: {output_name}")

    # Sort
    run(["samtools", "sort", "-o", str(sorted_bam), str(raw_bam)],
        f"{output_name} - sort")

    # Index
    run(["samtools", "index", str(sorted_bam)],
        f"{output_name} - index")

    raw_bam.unlink(missing_ok=True)
    print(f"   ✓ {sorted_bam.name}")
    return sorted_bam


# ============================================================
# PIPELINE STEPS
# ============================================================

def step1_subset_bam_to_chrM(input_bam: Path, work_dir: Path) -> Path:
    """Step 1. Extract chrM reads from the input BAM."""
    print("\n[STEP 1/9] Extracting chrM reads from BAM")
    output = work_dir / "chrM_reads.bam"

    run(
        ["samtools", "view", "-b", "-h",
         "-f", "2",       # properly paired
         "-F", "256",     # skip secondary alignments
         str(input_bam), "chrM",
         "-o", str(output)],
        "samtools view (chrM extraction)"
    )
    run(["samtools", "index", str(output)], "samtools index")
    return output


def step5_mutect2(
    aligned_bam: Path,
    reference_fasta: Path,
    work_dir: Path,
    output_name: str,
    interval: str = "chrM",
) -> Path:
    """Steps 5 & 6. GATK Mutect2 in mitochondrial mode."""
    print(f"\n[STEP] Mutect2 {output_name}")
    output_vcf = work_dir / f"{output_name}.vcf.gz"

    run(
        ["gatk", "Mutect2",
         "-R", str(reference_fasta),
         "-I", str(aligned_bam),
         "-L", interval,
         "--mitochondria-mode",
         "--annotation", "StrandBiasBySample",
         "--max-reads-per-alignment-start", "75",
         "--max-mnp-distance", "0",
         "-O", str(output_vcf)],
        f"Mutect2 ({output_name})"
    )
    return output_vcf


def step7_filter_mutect_calls(
    raw_vcf: Path,
    reference_fasta: Path,
    work_dir: Path,
    output_name: str,
) -> Path:
    """Step 7. FilterMutectCalls."""
    print(f"\n[STEP] FilterMutectCalls {output_name}")
    filtered_vcf = work_dir / f"{output_name}.filtered.vcf.gz"
    stats_file   = Path(str(raw_vcf) + ".stats")

    run(
        ["gatk", "FilterMutectCalls",
         "-V", str(raw_vcf),
         "-R", str(reference_fasta),
         "--stats", str(stats_file),
         "--mitochondria-mode",
         "--max-alt-allele-count", "4",
         "--min-allele-fraction", "0.01",
         "-O", str(filtered_vcf)],
        f"FilterMutectCalls ({output_name})"
    )
    return filtered_vcf


def step8_liftover_shifted(shifted_vcf: Path, work_dir: Path) -> Path:
    """Step 8. Liftover shifted VCF coordinates back to standard chrM."""
    print("\n[STEP 8/9] Lifting over shifted VCF coordinates")
    lifted_vcf = work_dir / "shifted_lifted.vcf.gz"
    rejected   = work_dir / "shifted_rejected.vcf.gz"

    run(
        ["gatk", "LiftoverVcf",
         "-I", str(shifted_vcf),
         "-O", str(lifted_vcf),
         "-R", str(MT_FASTA),
         "--CHAIN", str(SHIFT_BACK_CHAIN),
         "--REJECT", str(rejected)],
        "LiftoverVcf (shifted → standard coordinates)"
    )
    return lifted_vcf


def step9_merge_vcfs(
    normal_filtered: Path,
    shifted_lifted: Path,
    work_dir: Path,
    sample_id: str,
) -> Path:
    """Step 9. Merge normal and lifted shifted VCFs into the final output."""
    print("\n[STEP 9/9] Generating final VCF")
    final_vcf = work_dir / f"{sample_id}.final.vcf.gz"

    run(
        ["gatk", "MergeVcfs",
         "-I", str(normal_filtered),
         "-I", str(shifted_lifted),
         "-O", str(final_vcf)],
        "MergeVcfs"
    )
    return final_vcf


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_sample(input_bam: Path) -> Path:
    """Execute the complete pipeline for a single sample."""
    sample_id = get_sample_name(input_bam)

    print("\n" + "=" * 70)
    print(f"  GATK Mitochondria Pipeline (Python)")
    print(f"  Sample: {sample_id}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir  = RESULTS_DIR / sample_id / f"run_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  📁 Working directory: {work_dir}")

    start = datetime.now()

    # STEP 1: chrM extraction
    chrM_bam = step1_subset_bam_to_chrM(input_bam, work_dir)

    # STEP 2: BAM → FASTQ (samtools — Solexa-compatible, no RevertSam)
    print("\n[STEP 2/9] BAM → FASTQ (samtools)")
    fq1 = work_dir / "temp_R1.fastq"
    fq2 = work_dir / "temp_R2.fastq"

    result = subprocess.run(
        f"samtools sort -n {chrM_bam} "
        f"| samtools fastq -1 {fq1} -2 {fq2} -0 /dev/null -s /dev/null -n",
        shell=True
    )
    if result.returncode != 0:
        raise RuntimeError("samtools fastq failed")
    print(f"   ✓ FASTQ files ready ({fq1.stat().st_size // 1024} KB + {fq2.stat().st_size // 1024} KB)")

    # STEP 3: Re-alignment to standard chrM
    print("\n[STEP 3/9] Re-alignment to standard chrM")
    mt_bam = _bwa_align(fq1, fq2, MT_FASTA, work_dir, "aligned_mt", sample_id)

    # STEP 4: Re-alignment to shifted chrM
    print("\n[STEP 4/9] Re-alignment to shifted chrM")
    mt_shifted_bam = _bwa_align(fq1, fq2, MT_SHIFTED_FASTA, work_dir, "aligned_mt_shifted", sample_id)

    # Remove temporary FASTQ files
    fq1.unlink(missing_ok=True)
    fq2.unlink(missing_ok=True)

    # STEP 5: Mutect2 on standard chrM
    print("\n[STEP 5/9] Mutect2 — standard chrM")
    raw_normal_vcf = step5_mutect2(mt_bam, MT_FASTA, work_dir, "raw_normal", interval="chrM")

    # STEP 6: Mutect2 on shifted chrM
    print("\n[STEP 6/9] Mutect2 — shifted chrM")
    raw_shifted_vcf = step5_mutect2(mt_shifted_bam, MT_SHIFTED_FASTA, work_dir, "raw_shifted", interval="chrM")

    # STEP 7: Filtering
    print("\n[STEP 7/9] FilterMutectCalls")
    filtered_normal  = step7_filter_mutect_calls(raw_normal_vcf,  MT_FASTA,         work_dir, "normal")
    filtered_shifted = step7_filter_mutect_calls(raw_shifted_vcf, MT_SHIFTED_FASTA, work_dir, "shifted")

    # STEP 8: Liftover
    lifted_shifted = step8_liftover_shifted(filtered_shifted, work_dir)

    # STEP 9: Merge
    final_vcf = step9_merge_vcfs(filtered_normal, lifted_shifted, work_dir, sample_id)

    duration = (datetime.now() - start).total_seconds()

    print(f"\n{'=' * 70}")
    print(f"  ✓ Pipeline completed successfully. ({duration:.1f}s)")
    print(f"  📄 Final VCF: {final_vcf}")
    print(f"{'=' * 70}")

    print("\n  VCF contents (PASS variants):")
    subprocess.run(
        f"bcftools view -H -f PASS {final_vcf} | head -20",
        shell=True, check=False
    )

    return final_vcf


def main():
    for tool in ["gatk", "bwa", "samtools", "bcftools"]:
        if not shutil.which(tool):
            raise EnvironmentError(f"{tool} not found. Is the Conda environment active?")

    bam_files = sorted(BAM_DIR.glob("*.sorted.bam"))
    if not bam_files:
        raise FileNotFoundError(f"No BAM files found in: {BAM_DIR}")

    print(f"Samples to process: {len(bam_files)}")
    for bam in bam_files:
        print(f"  • {bam.name}")

    for bam in bam_files:
        process_sample(bam)


if __name__ == "__main__":
    main()
