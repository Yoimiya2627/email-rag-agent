"""Process-local warm-up state; liveness never implies model/network readiness."""
import threading
import time


class Readiness:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {'status':'cold','components':{},'error_code':None}

    def snapshot(self):
        with self._lock:
            return {**self._state,'components':dict(self._state['components']),
                    'provider_network_verified':False,'scope':'this_api_process'}

    def start(self, initializer):
        with self._lock:
            if self._state['status'] in {'warming','ready'}:
                return False
            self._state = {'status':'warming','components':{},'error_code':None}
        def run():
            started = time.monotonic()
            try:
                components = initializer()
                with self._lock:
                    self._state.update(status='ready',components=components,
                        elapsed_seconds=round(time.monotonic()-started,3))
            except Exception as exc:
                with self._lock:
                    self._state.update(status='failed',error_code=type(exc).__name__,
                        elapsed_seconds=round(time.monotonic()-started,3))
        try:
            threading.Thread(target=run,daemon=True,name='email-agent-warmup').start()
        except Exception as exc:
            with self._lock:
                self._state.update(status='failed',error_code=type(exc).__name__)
            raise
        return True
