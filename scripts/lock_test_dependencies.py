"""Snapshot the installed dependency closure; no resolver, network or installation."""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import platform
import re

from packaging.requirements import Requirement

TEST_ROOTS=['pytest','fastapi','pydantic','openai','httpx','python-dotenv','rank-bm25','mcp','langgraph','ijson','tzdata','PyYAML']


def installed_constraints(roots=TEST_ROOTS):
    queue=[(name,set()) for name in roots]
    seen, extras_seen={}, {}
    while queue:
        name, requested_extras=queue.pop()
        canonical=re.sub(r'[-_.]+','-',name).lower()
        if canonical in seen and requested_extras.issubset(extras_seen[canonical]):
            continue
        distribution=metadata.distribution(name)
        seen[canonical]=distribution.version
        extras_seen.setdefault(canonical,set()).update(requested_extras)
        for text in distribution.requires or []:
            dependency=Requirement(text)
            if dependency.marker is None or any(dependency.marker.evaluate({'extra':extra})
                                                for extra in {''}|extras_seen[canonical]):
                installed=metadata.version(dependency.name)
                if dependency.specifier and installed not in dependency.specifier:
                    raise ValueError('installed dependency does not satisfy '+str(dependency))
                queue.append((dependency.name,set(dependency.extras)))
    lines=['# Installed dependency closure, not an untested production/model lock.',
           '# Captured on '+platform.system()+' / Python '+platform.python_version()+'.',
           '# Linux/other Python combinations require their own CI validation.',
           '# Roots: '+', '.join(roots)]
    for name,version in sorted(seen.items()):
        suffix=' ; sys_platform == "win32"' if name in {'pywin32'} else ''
        lines.append(name+'=='+version+suffix)
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile',choices=['test','storage','application'],default='test')
    args=parser.parse_args(argv)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    roots=TEST_ROOTS+(['chromadb'] if args.profile in {'storage','application'} else [])
    if args.profile=='application':
        roots+=['streamlit','sseclient-py','google-api-python-client','google-auth-oauthlib','google-auth-httplib2']
    output.write_text(installed_constraints(roots),encoding='utf-8')


if __name__=='__main__':
    main()
