"""
Setup one-time pentru pylidc.

Rulează acest script O SINGURĂ DATĂ înainte de a folosi pipeline-ul:
    python preprocessing/setup_pylidc.py

Creează fișierul ~/.pylidcrc cu calea corectă spre datele DICOM,
pe care pylidc îl citește automat la import.
"""

import os
import sys
from configparser import ConfigParser

# Importăm calea din config-ul nostru
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LIDC_DATA_ROOT


PYLIDCRC_PATH = os.path.expanduser("~/.pylidcrc")


def write_pylidcrc(dicom_path: str) -> None:
    """Scrie fișierul ~/.pylidcrc cu calea spre datele LIDC-IDRI."""
    if os.path.exists(PYLIDCRC_PATH):
        print(f"[INFO] ~/.pylidcrc există deja. Suprascriu cu noua cale.")

    config = ConfigParser()
    config["dicom"] = {"path": dicom_path}

    with open(PYLIDCRC_PATH, "w") as f:
        config.write(f)

    print(f"[OK] ~/.pylidcrc creat cu path: {dicom_path}")


def verify_pylidc_connection() -> bool:
    """
    Verifică că pylidc poate accesa datele DICOM.
    Returnează True dacă găsește scan-urile așteptate.
    """
    try:
        import pylidc as pl

        count = pl.query(pl.Scan).count()
        print(f"[OK] pylidc găsit {count} scan-uri în baza de date.")

        if count == 0:
            print("[WARN] Niciun scan găsit. Verifică calea din ~/.pylidcrc.")
            print(f"       Calea curentă: {LIDC_DATA_ROOT}")
            return False

        if count != 1010:
            print(f"[WARN] Așteptat 1010 scan-uri, găsit {count}.")
            print("       Dataset-ul poate fi incomplet.")

        # Test că putem accesa primul pacient
        first_scan = pl.query(pl.Scan).first()
        print(f"[OK] Test accès: primul pacient = {first_scan.patient_id}")
        return True

    except Exception as e:
        print(f"[ERROR] Nu s-a putut conecta pylidc: {e}")
        return False


def main():
    print("=== Setup pylidc ===\n")

    # 1. Verifică că datele există la calea din config
    if not os.path.isdir(LIDC_DATA_ROOT):
        print(f"[ERROR] Calea LIDC_DATA_ROOT nu există: {LIDC_DATA_ROOT}")
        print("        Verifică config.py → LIDC_DATA_ROOT")
        sys.exit(1)

    print(f"[OK] Date DICOM găsite la: {LIDC_DATA_ROOT}")

    # 2. Scrie .pylidcrc
    write_pylidcrc(LIDC_DATA_ROOT)

    # 3. Verifică conexiunea
    print("\n[INFO] Verificare conexiune pylidc...")
    success = verify_pylidc_connection()

    if success:
        print("\n[OK] Setup complet. Poți rula pipeline.py acum.")
    else:
        print("\n[ERROR] Setup eșuat. Verifică mesajele de mai sus.")
        sys.exit(1)


if __name__ == "__main__":
    main()
