"""Single-owner API boundary. This is not a multi-tenant authentication system."""
import hmac
import ipaddress
from dataclasses import dataclass

from fastapi import HTTPException, Request
import config.settings as cfg


@dataclass(frozen=True)
class Identity:
    owner_id: str


def require_identity(request: Request) -> Identity:
    configured = cfg.API_AUTH_TOKEN
    if configured:
        scheme, _, token = request.headers.get('authorization', '').partition(' ')
        if scheme.lower() != 'bearer' or not hmac.compare_digest(token.encode(), configured.encode()):
            raise HTTPException(401, 'API authentication required', headers={'WWW-Authenticate': 'Bearer'})
    else:
        hostname = request.url.hostname or ''
        try:
            allowed_host = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            allowed_host = hostname.lower() == 'localhost'
        test_peer = request.client and request.client.host == 'testclient'
        if not allowed_host and not (test_peer and hostname == 'testserver'):
            raise HTTPException(403, 'Local access requires a loopback Host or API_AUTH_TOKEN')
        origin = request.headers.get('origin')
        if origin and origin not in cfg.API_CORS_ORIGINS:
            raise HTTPException(403, 'Origin not allowed for unauthenticated local access')
        host = request.client.host if request.client else ''
        # TestClient is an in-process ASGI peer; real TCP peers have IP addresses.
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == 'testclient'
        if not local:
            raise HTTPException(403, 'Remote access requires API_AUTH_TOKEN')
    return Identity(cfg.API_OWNER_ID)
