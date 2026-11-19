"""Download the authors' CARGO archive and extract it without changing its split."""
from pathlib import Path
import zipfile

import gdown


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / "downloads" / "CARGO.zip"
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists() or not zipfile.is_zipfile(archive):
        result = gdown.download(
            id="1yDjyH0VtW7efxP3vgQjIqTx2oafCB67t",
            output=str(archive), resume=True,
        )
        if result is None:
            raise RuntimeError("CARGO download failed")
    destination = root / "datasets" / "CARGO"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()
        if bad:
            raise RuntimeError(f"Corrupt archive entry: {bad}")
        for info in source.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive entry: {info.filename}")
        print(f"Extracting {len(source.infolist())} entries to {destination}", flush=True)
        source.extractall(destination)
    print("CARGO extraction completed", flush=True)


if __name__ == "__main__":
    main()
