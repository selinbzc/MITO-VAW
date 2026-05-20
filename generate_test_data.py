"""
Synthetic mtDNA test data generator (Mutect2-compatible version).

Critical properties for Mutect2 mitochondria-mode:
- Balanced strand bias (reads on both strands)
- Diverse read start positions (minimizes risk of PCR duplicate flagging)
- Variable base quality scores
- Variable insert sizes (normally distributed)
"""
import os
import gzip
import random
import subprocess
from pathlib import Path
from typing import List

random.seed(42)

# === CONFIGURATION ===
PROJECT_ROOT = Path.home() / "Selin" / "mito_pipeline" / "gatk_pipeline"
HG38_FASTA = PROJECT_ROOT / "references" / "Homo_sapiens_assembly38.fasta"
DATA_DIR = PROJECT_ROOT / "data"
FASTQ_DIR = DATA_DIR / "fastq"

FASTQ_DIR.mkdir(parents=True, exist_ok=True)

# Known pathogenic mtDNA mutations
KNOWN_MUTATIONS = [
    (3243, 'A', 'G', 'MELAS syndrome', 0.65),
    (8344, 'A', 'G', 'MERRF syndrome', 0.30),
    (11778, 'G', 'A', 'LHON', 0.95),
]


def extract_chrM_from_hg38() -> str:
    """Extract chrM from the local hg38 reference using samtools."""
    print(">>> Extracting chrM from local hg38 reference...")
    
    if not HG38_FASTA.exists():
        raise FileNotFoundError(f"hg38 reference not found: {HG38_FASTA}")
    
    result = subprocess.run(
        ["samtools", "faidx", str(HG38_FASTA), "chrM"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"samtools error: {result.stderr}")
    
    lines = result.stdout.strip().split('\n')
    sequence = ''.join(lines[1:]).upper()
    print(f"   ✓ chrM length: {len(sequence)} bp")
    return sequence


def apply_mutation(seq: str, position: int, alt_base: str) -> str:
    idx = position - 1
    return seq[:idx] + alt_base + seq[idx + 1:]


def variable_quality(length: int) -> str:
    """
    Realistic quality score profile — decreasing toward the 3' end with random variance.
    Illumina quality pattern: high at the start (Q38–40), declining at the end (Q30–35).
    """
    qualities = []
    for i in range(length):
        # Position-dependent decay with random fluctuation
        position_factor = 1.0 - (i / length) * 0.2  # decline over the last 20%
        base_q = int(38 * position_factor)
        # ±3 random variance
        q = base_q + random.randint(-3, 2)
        q = max(35, min(40, q))  # clamp to Q35–40 range
        qualities.append(chr(q + 33))
    return ''.join(qualities)


def reverse_complement(seq: str) -> str:
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


def add_sequencing_errors(seq: str, error_rate: float = 0.001) -> str:
    bases = ['A', 'C', 'G', 'T']
    return ''.join(
        random.choice([x for x in bases if x != b]) if random.random() < error_rate else b
        for b in seq
    )


def generate_paired_reads(
    reference: str,
    mutated_reference: str,
    mutation_positions: List[int],
    heteroplasmy_per_mutation: List[float],
    sample_id: str,
    coverage: int = 200,           # 100 → 200 (Mutect2 performs better at higher coverage)
    read_length: int = 150,
    insert_mean: int = 350,
    insert_std: int = 80,          # Increased standard deviation
):
    """
    Generate paired-end FASTQ files with Mutect2-compatible diversity.
    """
    ref_len = len(reference)
    n_pairs = (coverage * ref_len) // (2 * read_length)
    
    fq1 = FASTQ_DIR / f"{sample_id}_R1.fastq.gz"
    fq2 = FASTQ_DIR / f"{sample_id}_R2.fastq.gz"
    
    print(f"\n>>> Generating FASTQ for {sample_id} (Mutect2-compatible)...")
    print(f"   Read pairs: {n_pairs}, target coverage: ~{coverage}x")
    print(f"   Insert size: {insert_mean} ± {insert_std} bp (normal distribution)")
    print(f"   Read length: {read_length} bp")
    print(f"   Heteroplasmy levels:")
    for pos, het in zip(mutation_positions, heteroplasmy_per_mutation):
        print(f"     • Position {pos}: {int(het*100)}%")
    
    with gzip.open(fq1, 'wt') as f1, gzip.open(fq2, 'wt') as f2:
        for i in range(n_pairs):
            # Fresh copy per read to ensure diversity
            current_seq = list(reference)
            for pos, het in zip(mutation_positions, heteroplasmy_per_mutation):
                if random.random() < het:
                    current_seq[pos - 1] = mutated_reference[pos - 1]
            current_seq = ''.join(current_seq)
            
            # Insert size — normally distributed
            actual_insert = int(random.gauss(insert_mean, insert_std))
            actual_insert = max(read_length * 2 + 20, 
                              min(actual_insert, ref_len - 1, 600))
            
            # Random start position — simulates circular mtDNA
            # 20% probability of wrap-around (mtDNA is circular)
            if random.random() < 0.2 and ref_len > actual_insert + 100:
                # Wrap: start near the end of the reference, continue from the beginning
                start = random.randint(ref_len - actual_insert - 50, ref_len - 100)
                if start + actual_insert > ref_len:
                    insert = current_seq[start:] + current_seq[:actual_insert - (ref_len - start)]
                else:
                    insert = current_seq[start:start + actual_insert]
            else:
                max_start = ref_len - actual_insert
                start = random.randint(0, max_start) if max_start > 0 else 0
                insert = current_seq[start:start + actual_insert]
            
            # Strand selection — 50% forward, 50% reverse (for balanced strand bias)
            is_reverse = random.random() < 0.5
            
            if is_reverse:
                # Read from the reverse strand
                # R1: reverse complement of the 3' end of the insert
                # R2: forward sequence from the 5' end of the insert
                r1_raw = reverse_complement(insert[-read_length:])
                r2_raw = insert[:read_length]
            else:
                # Forward strand
                r1_raw = insert[:read_length]
                r2_raw = reverse_complement(insert[-read_length:])
            
            # Add sequencing errors
            r1 = add_sequencing_errors(r1_raw)
            r2 = add_sequencing_errors(r2_raw)
            
            # Quality scores (generated independently per read)
            q1 = variable_quality(len(r1))
            q2 = variable_quality(len(r2))
            
            read_id = f"{sample_id}_read_{i:06d}"
            f1.write(f"@{read_id}/1\n{r1}\n+\n{q1}\n")
            f2.write(f"@{read_id}/2\n{r2}\n+\n{q2}\n")
    
    print(f"   ✓ {fq1.name} ({fq1.stat().st_size // 1024} KB)")
    print(f"   ✓ {fq2.name} ({fq2.stat().st_size // 1024} KB)")


def main():
    print("=" * 60)
    print("Synthetic mtDNA test data (Mutect2-compatible)")
    print("=" * 60)
    
    reference = extract_chrM_from_hg38()
    
    print(f"\n>>> Applying mutations:")
    mutated = reference
    for pos, ref_base, alt_base, name, hetero in KNOWN_MUTATIONS:
        actual = reference[pos - 1]
        if actual != ref_base:
            print(f"   ⚠ Position {pos}: expected {ref_base}, found {actual} — skipping")
            continue
        mutated = apply_mutation(mutated, pos, alt_base)
        print(f"   ✓ m.{pos}{ref_base}>{alt_base} ({name}) — heteroplasmy {int(hetero*100)}%")
    
    generate_paired_reads(
        reference=reference,
        mutated_reference=mutated,
        mutation_positions=[m[0] for m in KNOWN_MUTATIONS],
        heteroplasmy_per_mutation=[m[4] for m in KNOWN_MUTATIONS],
        sample_id="patient01",
        coverage=200,        # ↑ Higher coverage
        read_length=150,
        insert_mean=350,
        insert_std=80,       # ↑ Wider variance
    )
    
    print("\n" + "=" * 60)
    print("✓ Test data generated successfully.")
    print(f"   Output directory: {FASTQ_DIR}")
    print("=" * 60)
    print("\nNext step: python align.py")


if __name__ == "__main__":
    main()
