"""Bounded, byte-verified local artifact bundles; no network or application imports."""
from pathlib import Path,PurePosixPath
from contextlib import closing,nullcontext
import hashlib
import os
import stat

KINDS={'corpus':'file','raw':'directory','index':'directory'}


def reject_links(path):
    path=Path(path).absolute()
    for part in [path,*path.parents]:
        if part.exists() or part.is_symlink():
            info=part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                raise ValueError('Bundle paths must not contain symlinks, junctions or reparse points')
    return path.resolve(strict=True)


def member_path(value):
    if not isinstance(value,str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid bundle member path')
    parsed=PurePosixPath(value)
    if parsed.is_absolute() or any(part in {'','..','.'} or part.endswith((' ','.')) for part in value.split('/')):
        raise ValueError('Invalid bundle member path')
    reserved={'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}
    if any(part.split('.')[0].upper() in reserved for part in parsed.parts):
        raise ValueError('Invalid Windows bundle member name')
    return parsed


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1_048_576),b''):
            value.update(block)
    return value.hexdigest()


def inventory(artifacts,*,max_files=100000,max_bytes=10*1024**3):
    if type(max_files) is not int or max_files<1 or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('Bundle budgets must be positive integers')
    if not isinstance(artifacts,dict) or not set(artifacts).issubset(KINDS):
        raise ValueError('Unknown artifact kind')
    plans={};count=total=0
    for kind,value in artifacts.items():
        root=reject_links(value)
        if (KINDS[kind]=='file') != root.is_file() or (KINDS[kind]=='directory' and not root.is_dir()):
            raise ValueError('Artifact kind does not match its path')
        files=[];seen=set()
        if root.is_file():
            paths=[root]
        else:
            def walk():
                pending=[root]
                directories=0
                while pending:
                    parent=pending.pop()
                    with os.scandir(parent) as entries:
                        for entry in entries:
                            member=reject_links(parent/entry.name)
                            if entry.is_dir(follow_symlinks=False):
                                directories+=1
                                if directories>10000 or len(member.relative_to(root).parts)>64:
                                    raise ValueError('Bundle directory/depth budget exceeded')
                                pending.append(member)
                            else:
                                yield member
            paths=walk()
        with (closing(paths) if root.is_dir() else nullcontext(paths)):
            for source in paths:
                source=reject_links(source)
                if not source.is_file() or (root.is_dir() and not source.is_relative_to(root)):
                    raise ValueError('Artifact member escapes its root or is not a regular file')
                relative='data.json' if root.is_file() else source.relative_to(root).as_posix()
                member_path(relative)
                if relative.casefold() in seen:
                    raise ValueError('Bundle members collide on a case-insensitive filesystem')
                seen.add(relative.casefold())
                info=source.stat();count+=1;total+=info.st_size
                if count>max_files or total>max_bytes:
                    raise ValueError('Artifact bundle exceeds its file or byte budget')
                checksum=digest(source)
                after=source.stat()
                if (info.st_size,info.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise ValueError('Source changed while preparing the backup; stop all writers')
                files.append({'path':relative,'bytes':info.st_size,'sha256':checksum})
        plans[kind]={'type':KINDS[kind],'files':sorted(files,key=lambda row:row['path']),
                     'total_bytes':sum(row['bytes'] for row in files)}
    return plans


def _target(root,relative):
    path=root.joinpath(*member_path(relative).parts)
    path.parent.mkdir(parents=True,exist_ok=True)
    parent=reject_links(path.parent)
    if not parent.is_relative_to(root.resolve()):
        raise ValueError('Bundle destination escapes its root')
    return path


def copy_artifacts(artifacts,plans,destination):
    destination=Path(destination)
    for kind,plan in plans.items():
        source_root=reject_links(artifacts[kind])
        target_root=destination/'artifacts'/kind
        target_root.mkdir(parents=True,exist_ok=False)
        for item in plan['files']:
            source=source_root if source_root.is_file() else source_root.joinpath(*member_path(item['path']).parts)
            source=reject_links(source)
            target=_target(target_root,item['path'])
            _copy_verified(source,target,item)


def _copy_verified(source,target,item):
    checksum=hashlib.sha256();size=0
    with source.open('rb') as reader,target.open('xb') as writer:
        for block in iter(lambda:reader.read(1_048_576),b''):
            size+=len(block)
            if size>item['bytes']:
                raise ValueError('Source grew during bundle copy')
            checksum.update(block);writer.write(block)
        writer.flush();os.fsync(writer.fileno())
    if size!=item['bytes'] or checksum.hexdigest()!=item['sha256']:
        raise ValueError('Source changed during bundle copy')


def validate_artifacts(directory,plans,*,max_files=100000,max_bytes=10*1024**3):
    if type(max_files) is not int or max_files<1 or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('Bundle budgets must be positive integers')
    if not isinstance(plans,dict) or not set(plans).issubset(KINDS):
        raise ValueError('Invalid artifact manifest')
    directory=reject_links(directory)
    count=total=0
    for kind,plan in plans.items():
        if not isinstance(plan,dict) or plan.get('type')!=KINDS[kind] or not isinstance(plan.get('files'),list):
            raise ValueError('Invalid artifact manifest')
        root=reject_links(directory/'artifacts'/kind)
        if not root.is_relative_to(directory):
            raise ValueError('Artifact root escapes backup directory')
        seen=set()
        for item in plan['files']:
            relative=member_path(item.get('path'))
            if str(relative).casefold() in seen:
                raise ValueError('Duplicate artifact member')
            seen.add(str(relative).casefold())
            if type(item.get('bytes')) is not int or item['bytes']<0:
                raise ValueError('Invalid artifact size')
            count+=1;total+=item['bytes']
            if count>max_files or total>max_bytes:
                raise ValueError('Artifact restore exceeds its file or byte budget')
            source=reject_links(root.joinpath(*relative.parts))
            if not source.is_file() or not source.is_relative_to(root):
                raise ValueError('Artifact member escapes backup root')
            if source.stat().st_size!=item['bytes'] or digest(source)!=item.get('sha256'):
                raise ValueError('Artifact checksum mismatch')
        if KINDS[kind]=='file' and (len(plan['files'])!=1 or plan['files'][0]['path']!='data.json'):
            raise ValueError('Invalid corpus artifact member')
    return plans


def restore_artifacts(directory,plans,destination):
    for kind,plan in plans.items():
        target_root=Path(destination)/'artifacts'/kind
        target_root.mkdir(parents=True,exist_ok=False)
        for item in plan['files']:
            source=reject_links(Path(directory)/'artifacts'/kind/item['path'])
            _copy_verified(source,_target(target_root,item['path']),item)
