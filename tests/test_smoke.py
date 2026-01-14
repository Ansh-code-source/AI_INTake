def test_package_import():
    import ai_intake_bot

    assert hasattr(ai_intake_bot, "__version__")


def test_credentials_model_basic():
    from ai_intake_bot.security.credentials import CredentialsConfig

    c = CredentialsConfig()
    assert c.openai_api_key is None
    assert isinstance(c.redacted(), dict)
