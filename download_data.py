"""Download the two official source files and verify SHA-256; ~1.28 GB."""
import hashlib
from pathlib import Path
import urllib.request
root = Path(__file__).resolve().parent / "Experiments/data/raw"
root.mkdir(parents=True, exist_ok=True)
files = [
 (119641, "20230316_154153_n64_l1_s50000_sr2_f2.610G_g30.0.bin", "b474340b09a52e50d23cafdc23afa747f52eb6bcad6adfdc8614a6f6d81062d0"),
 (119648, "Mar-16th-2023-04-40PM-Flight-Airdata.csv", "b94c81ef171f3e194e7bfb882baa7f39b7ce192ef55edc0593c909c5697e01b3"),
]
def sha(path):
 with path.open("rb") as stream:
  return hashlib.file_digest(stream, "sha256").hexdigest()
for file_id, name, digest in files:
 target = root / name
 if target.exists():
  if sha(target) != digest:
   raise RuntimeError(f"Existing file checksum mismatch: {target}; inspect before removing it.")
 else:
  temporary = target.with_suffix(target.suffix + ".part")
  print(f"Downloading {name}", flush=True)
  with urllib.request.urlopen(f"https://rdr.kuleuven.be/api/access/datafile/{file_id}", timeout=120) as response, temporary.open("wb") as output:
   while block := response.read(1024 * 1024):
    output.write(block)
  if sha(temporary) != digest:
   raise RuntimeError(f"Download checksum mismatch: {temporary}")
  temporary.replace(target)
 print(f"SHA-256 verified: {name}")
