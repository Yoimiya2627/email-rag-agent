"""Typed backend-owned reference material. Text can never promote its authority."""
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class TrustedScope:
    owner_id: str
    session_id: str | None
    run_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.owner_id, str) or not self.owner_id:
            raise ValueError('trusted owner is required')
        if self.session_id is not None and (not isinstance(self.session_id, str) or not self.session_id):
            raise ValueError('invalid trusted session')

    def accepts(self, row):
        return all(row.get(key, expected) == expected for key, expected in
                   (('owner_id', self.owner_id), ('session_id', self.session_id)))


@dataclass(frozen=True)
class ContextMaterial:
    material_id: str
    kind: str
    scope: TrustedScope | None
    source_id: str
    version: str
    text: str
    start: int = 0
    coverage: str = 'selected_subset'
    required: bool = False
    offset_basis: str = 'material_text'

    def manifest(self, *, position, method):
        if type(self.start) is not int or self.start < 0:
            raise ValueError('invalid material source offset')
        return {'material_id': self.material_id, 'kind': self.kind, 'source_id': self.source_id,
                'version': self.version, 'visible_start': self.start,
                'visible_end': self.start + len(self.text),
                'visible_hash': sha256(self.text.encode('utf-8')).hexdigest(),
                'offset_basis': self.offset_basis, 'coverage': self.coverage,
                'position': position, 'token_estimation_method': method,
                'scope_validated': self.scope is not None, 'account_scope': 'single_account'}


@dataclass(frozen=True)
class ContextRequest:
    scope: TrustedScope | None
    stage: str | None
    current_user_text: str

    def __post_init__(self):
        if not isinstance(self.current_user_text, str):
            raise ValueError('current request must be text')
        if self.scope is not None and not isinstance(self.scope, TrustedScope):
            raise ValueError('context scope must come from the trusted backend')


@dataclass
class ContextSelection:
    request: ContextRequest
    messages: list
    material_manifest: list
    omissions: list

    def metrics(self):
        from collections import Counter
        return {'material_manifest':self.material_manifest, 'omissions':self.omissions,
                'material_counts':dict(Counter(row['kind'] for row in self.material_manifest)),
                'omission_counts':dict(Counter(row['reason'] for row in self.omissions)),
                'selection_policy':'purpose_v1', 'stage':self.request.stage}
