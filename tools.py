import os
import csv
import subprocess

def create_samplesheet(fastq_dir, output_csv="samplesheet.csv"):
    """Sadece .gz dosyalarını kabul eden samplesheet hazırlayıcı."""
    print(f"--- [HAZIRLIK] {fastq_dir} taranıyor... ---")
    files = [f for f in os.listdir(fastq_dir) if f.endswith('.gz')]
    
    if not files:
        print("⚠️ Uyarı: data/fastq içinde .gz dosyası bulunamadı!")
        return None

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'fastq_1', 'fastq_2'])
        for file in files:
            sample_name = file.split('.')[0]
            full_path = os.path.abspath(os.path.join(fastq_dir, file))
            writer.writerow([sample_name, full_path, ''])
            
    return os.path.abspath(output_csv)

def run_mitodetect(samplesheet_path):
    """nf-core/mitodetect pipeline'ını çalıştırır."""
    print(f"🚀 [PİPELİNE] nf-core/mitodetect başlatılıyor...")
    # Ubuntu'da 'nextflow' doğrudan komut olarak çalışır
    cmd = [
        "nextflow", "run", "nf-core/mitodetect",
        "-r", "dev",
        "--input", samplesheet_path,
        "--outdir", "./results",
        "-profile", "conda"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "BAŞARILI: Sonuçlar ./results klasöründe." if result.returncode == 0 else f"HATA: {result.stderr[:200]}"