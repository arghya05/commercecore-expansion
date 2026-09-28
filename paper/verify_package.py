"""Verify source hashes and compile the ZIP in isolation. Requires Tectonic + Poppler."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'evidence/build_manifest.json').read_text())
    pdf = HERE / 'CommerceCore_Expansion_Paper.pdf'
    archive = HERE / 'CommerceCore_Expansion_arXiv_Source.zip'
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == manifest['public_pdf_sha256']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['source_zip_sha256']
    with tempfile.TemporaryDirectory(prefix='commercecore-paper-verify-') as tmp:
        dst = Path(tmp)
        with zipfile.ZipFile(archive) as package:
            assert package.testzip() is None
            assert set(package.namelist()) == set(manifest['source_files_sha256'])
            for name, digest in manifest['source_files_sha256'].items():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                assert hashlib.sha256(package.read(name)).hexdigest() == digest
                assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
            package.extractall(dst)
        process = subprocess.run(['tectonic', '--keep-logs', 'commercecore_expansion_paper.tex'],
                                 cwd=dst, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.returncode:
            raise SystemExit(process.stdout)
        log = (dst / 'commercecore_expansion_paper.log').read_text()
        assert not re.search(r'undefined (?:references|citations)|(?:Reference|Citation) .+ undefined|Overfull \\[hv]box|Missing character', log, re.I)
        expected = subprocess.check_output(['pdftotext', str(pdf), '-'])
        actual = subprocess.check_output(['pdftotext', str(dst / 'commercecore_expansion_paper.pdf'), '-'])
        assert expected == actual, 'Isolated package PDF text differs from public PDF'
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        verification = dict(source_zip_sha256=manifest['source_zip_sha256'],
                            public_pdf_sha256=manifest['public_pdf_sha256'], zip_crc='passed',
                            source_entries_hash_verified=len(manifest['source_files_sha256']),
                            isolated_zip_compilation='passed', isolated_pdf_text_matches_public_pdf=True,
                            page_count=int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1)),
                            undefined_references_or_citations=False, overfull_boxes_or_missing_characters=False)
        (HERE / 'evidence/package_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
        print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
