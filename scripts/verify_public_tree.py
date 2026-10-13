"""Bounded public-tree checks; print locations/categories, never secret matches."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', '.venv', '.pytest_cache', 'output', '.playwright-cli', 'node_modules', 'dist', 'build', 'data', 'models'}
FORBIDDEN_SUFFIXES = {'.pem', '.key', '.pfx', '.p12', '.sqlite', '.sqlite3', '.db', '.log', '.zip', '.whl', '.pdf', '.docx', '.csv', '.jsonl'}
RULES = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b'),
    'personal-path': re.compile(r'(?i)(?:[A-Z]:[\\/](?:Users|BaiduNetdiskDownload)[\\/])'),
    'private-remote': re.compile(r'private' + r'-snapshot|i-[a-z0-9]{20,}'),
}

def inspect(paths, *, staged=False):
    findings=[]
    for path in paths:
        relative=path.relative_to(ROOT).as_posix()
        if path.is_symlink() or (staged and subprocess.check_output(['git', 'ls-files', '--stage', '--', relative], cwd=ROOT).startswith(b'120000 ')):
            findings.append({'path':relative,'issue':'symbolic-link'}); continue
        if not staged and not path.is_file():
            findings.append({'path':relative,'issue':'missing-file'}); continue
        if path.name.startswith('.env') and path.name != '.env.example':
            findings.append({'path':relative,'issue':'local-environment'})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or relative.startswith(('data/','models/','archive/','exports/')):
            findings.append({'path':relative,'issue':'excluded-material'})
        data = subprocess.check_output(['git', 'show', ':' + relative], cwd=ROOT) if staged else path.read_bytes()
        if len(data) > 1_000_000:
            findings.append({'path':relative,'issue':'oversized-file'})
        if path.suffix.lower() == '.png': continue
        try: body=data.decode('utf-8-sig')
        except UnicodeDecodeError:
            findings.append({'path':relative,'issue':'unreviewed-binary'}); continue
        for name,pattern in RULES.items():
            for match in pattern.finditer(body):
                findings.append({'path':relative,'line':body.count('\n',0,match.start())+1,'issue':name})
    return findings

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tracked',action='store_true'); args=parser.parse_args()
    if args.tracked:
        data=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT)
        paths=[ROOT/name.decode('utf-8') for name in data.split(b'\0') if name]
        if not paths: raise SystemExit('No tracked files; stage reviewed files first.')
    else:
        paths=[]
        for base, dirs, names in os.walk(ROOT, followlinks=False):
            selected=[d for d in dirs if d not in SKIP_DIRS and not d.endswith('.egg-info') and not d.startswith('.venv')]
            paths.extend(Path(base)/d for d in selected if (Path(base)/d).is_symlink())
            dirs[:]=[d for d in selected if not (Path(base)/d).is_symlink()]
            paths.extend(Path(base)/name for name in names)
    findings=inspect(paths, staged=args.tracked)
    print(json.dumps({'scope':'tracked' if args.tracked else 'tree','files_checked':len(paths),'findings':findings},ensure_ascii=False,indent=2))
    return bool(findings)

if __name__ == '__main__': raise SystemExit(main())
