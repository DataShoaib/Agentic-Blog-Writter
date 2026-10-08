
def test_redis_unavailable_fails_open_for_rate_limit(monkeypatch):
    """Redis-only limiter: a store outage must not block legitimate requests."""
    from app.services.cache import allow_request, get_redis_store

    monkeypatch.setattr(get_redis_store(), "incr_with_expiry", lambda key, ttl: None)

    assert allow_request("some-user") is True


class _FixedWindowScript:
    """Records how the Redis rate limiter invokes the fixed-window Lua script."""

    def __init__(self):
        self.calls = []

    def __call__(self, keys=None, args=None):
        self.calls.append((list(keys or []), list(args or [])))
        return 3


def test_redis_rate_limiter_uses_fixed_window_script(monkeypatch):
    """Regression: the registered Lua script must be executed, not a sliding pipeline."""
    from app.services.cache import RedisStore

    store = RedisStore()
    store._client = object()
    script = _FixedWindowScript()
    store._incr_script = script
    monkeypatch.setattr(store, "_ensure_client", lambda: None)

    assert store.incr_with_expiry("rate:u", 60) == 3
    assert script.calls == [(["rate:u"], ["60"])]
