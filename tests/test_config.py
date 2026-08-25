from app.config import APP_CONFIG


def test_core_config_defaults_are_sane():
    assert APP_CONFIG.max_workers >= 1
    assert APP_CONFIG.max_revision_attempts >= 0
    assert APP_CONFIG.rate_limit_per_minute >= 1
