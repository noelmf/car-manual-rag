"""The .env rules are easy to get subtly wrong and impossible to notice."""
import pytest

from car_manual_rag import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with no settings and an unread .env."""
    for name in config.SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_loaded", False)


def env_file(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadEnv:
    def test_reads_a_plain_assignment(self, tmp_path, monkeypatch):
        config.load_env(env_file(tmp_path, "GEMINI_API_KEY=abc123\n"))
        assert config.required(config.API_KEY) == "abc123"

    def test_an_exported_variable_wins_over_the_file(self, tmp_path, monkeypatch):
        # Otherwise overriding a setting for a single run would be impossible.
        monkeypatch.setenv("GEMINI_API_KEY", "del-entorno")
        config.load_env(env_file(tmp_path, "GEMINI_API_KEY=del-fichero\n"))
        assert config.required(config.API_KEY) == "del-entorno"

    def test_tolerates_export_quotes_comments_and_junk(self, tmp_path):
        config.load_env(env_file(tmp_path, "# comentario\n\n"
                                           'export GEMINI_MODEL="un-modelo"\n'
                                           "GEMINI_EMBEDDING='otro'\n"
                                           "linea sin igual\n"))
        assert config.required(config.MODEL) == "un-modelo"
        assert config.required(config.EMBEDDING) == "otro"

    def test_reads_the_file_only_once(self, tmp_path):
        path = env_file(tmp_path, "GEMINI_MODEL=primero\n")
        config.load_env(path)
        path.write_text("GEMINI_MODEL=segundo\n", encoding="utf-8")
        config.load_env(path)
        assert config.required(config.MODEL) == "primero"

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        config.load_env(tmp_path / "no-existe")


class TestRequired:
    def test_a_missing_setting_names_itself_and_where_to_get_it(self, monkeypatch):
        monkeypatch.setattr(config, "_loaded", True)     # do not read the real .env
        with pytest.raises(LookupError, match="GEMINI_API_KEY is not set"):
            config.required(config.API_KEY)

    def test_every_setting_has_a_hint(self):
        assert set(config.SETTINGS) == {config.API_KEY, config.EMBEDDING, config.MODEL}
        assert all(config.SETTINGS.values())

    def test_an_empty_value_counts_as_missing(self, monkeypatch):
        monkeypatch.setattr(config, "_loaded", True)
        monkeypatch.setenv("GEMINI_MODEL", "")
        with pytest.raises(LookupError):
            config.required(config.MODEL)
