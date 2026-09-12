"""Local-only RFC822 parser process. No model clients or network operations."""
import json
import sys
from pathlib import Path


def main():
    # Parsing has no legitimate network dependency, including remote HTML images.
    def local_only(event, arguments):
        if event in {'socket.connect', 'socket.getaddrinfo'}:
            raise RuntimeError('parser_network_disabled')
    sys.addaudithook(local_only)
    from core.imap_mime import MailParseError, parse_imap_message
    from core.mail_sync import atomic_write
    source, destination, locator_path = map(Path, sys.argv[1:4])
    locator = json.loads(locator_path.read_text(encoding='utf-8'))
    try:
        result = {'email': parse_imap_message(source.read_bytes(), **locator).model_dump()}
    except MailParseError as exc:
        result = {'error_code': exc.code}
    except Exception:
        result = {'error_code': 'parse_failed'}
    atomic_write(destination, json.dumps(result, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    main()
