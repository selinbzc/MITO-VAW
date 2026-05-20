"""
Aligner: FASTQ → BAM

Automatic mode detection:
- WGS mode:      Align to the full hg38 reference (whole genome required for NuMT filtering)
- Amplicon mode: Align to chrM only (amplicon/targeted sequencing)

Detection logic:
  - If >80% of the first 1,000 reads map to chrM → amplicon
  - Otherwise → WGS
"""
import os
import shutil
import subprocess
from pathlib import Path

# === CONFIGURATION ===
PROJECT_ROOT = Path.home() / "Selin" / "mito_pipeline" / "gatk_pipeline"
HG38_FASTA   = PROJECT_ROOT / "references" / "Homo_sapiens_assembly38.fasta"
CHRM_FASTA   = PROJECT_ROOT / "references" / "Homo_sapiens_assembly38.chrM.fasta"
BAM_DIR      = PROJECT_ROOT / "data" / "bam"

import os
THREADS = max(2, os.cpu_count() // 2)


def check_dependencies():
    missing = [t for t in ["bwa", "samtools"] if not shutil.which(t)]
    if missing:
        raise EnvironmentError(f"Missing tools: {missing}")
    print(f"✓ BWA: {shutil.which('bwa')}")
    print(f"✓ samtools: {shutil.which('samtools')}")


def detect_mode(fastq_r1: Path, threads: int = 4) -> str:
    """
    Align the first 10,000 reads to chrM and compute the mapping ratio.
    ≥70% chrM → amplicon; otherwise → wgs
    """
    print("\n>>> Detecting sequencing mode...")
    
    # Extract the first 10,000 reads (40,000 lines)
    head_cmd = f"zcat {fastq_r1} | head -40000"
    bwa_cmd = (
        f"bwa mem -t {threads} {CHRM_FASTA} - 2>/dev/null"
        f"| samtools view -F 4 -c"  # Count of mapped reads
    )
    
    result = subprocess.run(
        f"{head_cmd} | {bwa_cmd}",
        shell=True, capture_output=True, text=True
    )
    mapped = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    total = 10000
    ratio = mapped / total
    
    mode = "amplicon" if ratio >= 0.70 else "wgs"
    print(f"   chrM mapping ratio: {ratio*100:.1f}% ({mapped}/{total})")
    print(f"   → Mode: {mode.upper()}")
    return mode


def align_sample(
    sample_id: str,
    fastq_r1: Path,
    fastq_r2: Path,
    threads: int = THREADS,
    mode: str = "auto"   # "auto", "wgs", "amplicon"
) -> Path:
    """BWA-MEM alignment. Mode can be selected automatically or specified manually."""
    
    BAM_DIR.mkdir(parents=True, exist_ok=True)
    sorted_bam = BAM_DIR / f"{sample_id}.sorted.bam"
    
    print(f"\n{'='*60}")
    print(f">>> Aligning {sample_id}")
    print(f"{'='*60}")
    print(f"   R1: {fastq_r1.name}")
    print(f"   R2: {fastq_r2.name}")
    print(f"   Threads: {threads}")
    
    # Mode detection
    if mode == "auto":
        mode = detect_mode(fastq_r1, threads)
    
    # Reference selection
    if mode == "amplicon":
        reference = CHRM_FASTA
        print(f"   Reference: chrM (amplicon mode)")
    else:
        reference = HG38_FASTA
        print(f"   Reference: hg38 (WGS mode)")
    
    # BWA index verification
    bwt = Path(str(reference) + ".bwt")
    if not bwt.exists():
        raise FileNotFoundError(
            f"BWA index not found: {bwt}\n"
            f"Run: bwa index {reference}"
        )
    
    raw_bam = BAM_DIR / f"{sample_id}.raw.bam"
    
    # Read group — required by GATK
    rg = (
        f"@RG\\tID:{sample_id}_lane1"
        f"\\tSM:{sample_id}"
        f"\\tLB:{sample_id}_lib1"
        f"\\tPL:ILLUMINA"
    )
    
    # Amplicon mode-specific parameters
    if mode == "amplicon":
        # -Y: soft-clipping (for amplicon primers)
        # -M: bwa-mem compatibility flag (for Picard)
        extra_flags = "-Y -M"
    else:
        extra_flags = "-K 10000000"
    
    # BWA-MEM | samtools view
    print(f"\n[1/3] BWA-MEM alignment ({mode} mode)...")
    cmd = (
        f"bwa mem -t {threads} {extra_flags} -R '{rg}' "
        f"{reference} {fastq_r1} {fastq_r2} "
        f"| samtools view -@ {threads} -b -o {raw_bam} -"
    )
    print(f"   $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"BWA-MEM failed: {result.stderr[-500:]}")
    
    # Sort
    print(f"\n[2/3] BAM sorting...")
    sort_cmd = f"samtools sort -@ {threads} -o {sorted_bam} {raw_bam}"
    result = subprocess.run(sort_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"samtools sort failed: {result.stderr}")
    raw_bam.unlink()
    
    # Index
    print(f"\n[3/3] BAM indexing...")
    idx_cmd = f"samtools index {sorted_bam}"
    result = subprocess.run(idx_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"samtools index failed: {result.stderr}")
    
    print(f"\n✓ BAM: {sorted_bam} ({sorted_bam.stat().st_size // 1024} KB)")
    
    # Alignment statistics
    stats = subprocess.run(
        f"samtools flagstat {sorted_bam}",
        shell=True, capture_output=True, text=True
    )
    print("\n" + stats.stdout.strip().split('\n')[0])
    
    # Additional metrics for amplicon mode: coverage
    if mode == "amplicon":
        cov = subprocess.run(
            f"samtools depth {sorted_bam} | awk '{{sum+=$3}} END {{print \"Mean coverage: \" sum/NR \"x\"}}'",
            shell=True, capture_output=True, text=True
        )
        print(cov.stdout.strip())
    
    return sorted_bam


def get_sample_name(bam_path: Path) -> str:
    """Retrieve the SM (sample name) field from the BAM header."""
    result = subprocess.run(
        ["samtools", "view", "-H", str(bam_path)],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if line.startswith('@RG'):
            for field in line.split('\t'):
                if field.startswith('SM:') and field[3:].strip():
                    return field[3:].strip()
    return bam_path.name.replace(".sorted.bam", "")


def main():
    print("=" * 60)
    print("Aligner: FASTQ → BAM (Automatic mode detection)")
    print("=" * 60)
    
    check_dependencies()
    
    # Locate FASTQ files
    fastq_dir = PROJECT_ROOT / "data" / "fastq"
    sra_dir = PROJECT_ROOT / "data" / "sra_samples"
    
    # Search in both directories
    all_r1 = sorted(list(fastq_dir.glob("*_R1.fastq.gz")) +
                    list(sra_dir.glob("*_R1.fastq.gz")))
    
    if not all_r1:
        raise FileNotFoundError("No FASTQ files found")
    
    print(f"\n{len(all_r1)} sample(s) found:")
    for r1 in all_r1:
        print(f"   • {r1.name}")
    
    for r1 in all_r1:
        sample_id = r1.name.replace("_R1.fastq.gz", "")
        r2 = r1.parent / r1.name.replace("_R1", "_R2")
        
        if not r2.exists():
            print(f"⚠ R2 not found: {r2.name}, skipping")
            continue
        
        align_sample(sample_id, r1, r2)
    
    print("\n" + "=" * 60)
    print("✓ All alignments completed successfully.")
    print(f"   BAM output directory: {BAM_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
