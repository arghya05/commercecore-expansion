"""Compile the canonical preprint and package its self-contained LaTeX sources.

Mirrors the build discipline used by the original CommerceCore project's
paper/build_paper.py (github.com/arghya05/commercecore/tree/main/paper):
recompute evidence from real artifacts first, compile with Tectonic, fail
the build on any undefined reference/citation or missing glyph, then
sha256-verify and package a source zip -- entirely independent of that
project's own paper (no shared LaTeX, no shared evidence, no shared repo).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ["commercecore_expansion_paper.tex", "neurips_2026.sty"] + sorted(
    str(p.relative_to(HERE))
    for folder in ("sections", "figures", "evidence")
    for p in (HERE / folder).iterdir()
    if p.suffix in (".tex", ".csv") and p.is_file()
)


def main():
    if platform.system()!='Linux' or not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('Build and test the preprint on RunPod only.')
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_*.py"], check=True)
    subprocess.run([sys.executable, str(HERE / "build_evidence.py")], check=True)
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    compiler = shutil.which("tectonic") or shutil.which("xelatex")
    if not compiler:
        raise SystemExit("Install Tectonic or XeLaTeX on RunPod.")
    if Path(compiler).name=='tectonic':
        subprocess.run([compiler,"--keep-logs","--outdir",str(build),"commercecore_expansion_paper.tex"],cwd=HERE,check=True)
    else:
        for _ in range(3):
            subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),
                'commercecore_expansion_paper.tex'],cwd=HERE,check=True)
    log = (build / "commercecore_expansion_paper.log").read_text()
    problems = re.findall(
        r"^.*(?:undefined references|undefined citations|(?:Reference|Citation) .+ undefined|Overfull \\[hv]box|Missing character).*$",
        log, re.MULTILINE | re.IGNORECASE,
    )
    if problems:
        raise SystemExit("Resolve publication build warnings before packaging:\n" + "\n".join(problems))
    source = (HERE / "commercecore_expansion_paper.tex").read_text()
    if "[preprint,nonatbib]" not in source:
        raise SystemExit("Public build must use the preprint style option.")
    for name in SOURCE_FILES:
        if not (HERE / name).is_file():
            raise SystemExit(f"Missing package dependency: {name}")
    public_pdf = HERE / "CommerceCore_Expansion_Paper.pdf"
    shutil.copyfile(build / "commercecore_expansion_paper.pdf", public_pdf)
    archive = HERE / "CommerceCore_Expansion_arXiv_Source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in SOURCE_FILES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 28, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(entry, (HERE / name).read_bytes())
    manifest = {
        "compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
        "mode": "public preprint",
        "top_level_tex": "commercecore_expansion_paper.tex",
        "source_files_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
        "public_pdf_sha256": hashlib.sha256(public_pdf.read_bytes()).hexdigest(),
        "source_zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "validation": "Compiled successfully; no undefined citations/references, missing glyphs, or overfull boxes in final log",
        "execution_location": "RunPod",
        "pod_id": os.environ['RUNPOD_POD_ID'],
        "boundary": "Remote compilation; arXiv server compilation and moderation have not been performed",
    }
    (HERE / "evidence" / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {public_pdf.name} and {archive.name}")


if __name__ == "__main__":
    main()
