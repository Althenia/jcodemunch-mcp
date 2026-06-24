"""Behavior spec for the structured secret-file classifier (#351 redesign).

The case table mirrors the oracle attached to issue #351 (contributed by
@mmashwani): every pattern group, the precedence ordering, the template / public
-key / public-cert exemptions, path normalization, and the group-aware and
per-pattern override knobs.
"""

import pytest

from jcodemunch_mcp.secret_classifier import (
    classify_secret_file,
    GROUP_PATH_SPECIFIC,
    GROUP_EXACT_NAME,
    GROUP_KEY_MATERIAL,
    GROUP_CREDENTIAL_EXT,
    GROUP_SECRET_STORE,
    GROUP_BROAD_BASENAME,
)


# (path, is_secret, group_when_secret)
ORACLE = [
    # exact credential names
    (".env", True, GROUP_EXACT_NAME),
    (".env.local", True, GROUP_EXACT_NAME),
    (".env.example", False, None),
    (".env.template", False, None),
    (".npmrc", True, GROUP_EXACT_NAME),
    (".pypirc", True, GROUP_EXACT_NAME),
    (".netrc", True, GROUP_EXACT_NAME),
    ("id_rsa", True, GROUP_EXACT_NAME),
    ("id_rsa.pub", False, None),
    ("id_ed25519.pub", False, None),
    ("id_ed25519.old", True, GROUP_EXACT_NAME),
    ("id_ecdsa.backup", True, GROUP_EXACT_NAME),
    ("service-account-prod.json", True, GROUP_EXACT_NAME),
    ("service-account.sample.json", False, None),
    ("client_secret_123.json", True, GROUP_EXACT_NAME),
    ("client_secret.template.json", False, None),
    ("token.json", True, GROUP_EXACT_NAME),
    ("tokenizer_config.json", False, None),
    ("my-firebase-adminsdk-prod.json", True, GROUP_EXACT_NAME),
    ("account.agekey", True, GROUP_EXACT_NAME),
    # credential extensions
    ("server.key", True, GROUP_CREDENTIAL_EXT),
    ("server.key.example", False, None),
    ("truststore.jks", True, GROUP_CREDENTIAL_EXT),
    ("release.keystore", True, GROUP_CREDENTIAL_EXT),
    ("certificate.p12", True, GROUP_CREDENTIAL_EXT),
    ("certificate.pfx", True, GROUP_CREDENTIAL_EXT),
    ("putty.ppk", True, GROUP_CREDENTIAL_EXT),
    ("private.p8", True, GROUP_CREDENTIAL_EXT),
    ("bundle.pem", True, GROUP_CREDENTIAL_EXT),
    # key-material directories (precedence over the generic extension group)
    ("certs/server.key", True, GROUP_KEY_MATERIAL),
    ("keys/server.pem", True, GROUP_KEY_MATERIAL),
    ("private-keys/identity.ppk", True, GROUP_KEY_MATERIAL),
    ("certs/public.crt", False, None),
    ("certs/ca.cer", False, None),
    # path-specific credentials
    (".aws/credentials", True, GROUP_PATH_SPECIFIC),
    ("workspace/.aws/credentials", True, GROUP_PATH_SPECIFIC),
    (".aws/config", False, None),
    (".docker/config.json", True, GROUP_PATH_SPECIFIC),
    ("config/docker/config.json", False, None),
    (".kube/config", True, GROUP_PATH_SPECIFIC),
    (".config/gcloud/application_default_credentials.json", True, GROUP_PATH_SPECIFIC),
    ("gcloud/application_default_credentials.json", True, GROUP_PATH_SPECIFIC),
    ("composer/auth.json", True, GROUP_PATH_SPECIFIC),
    ("auth.json", False, None),
    (".cargo/credentials.toml", True, GROUP_PATH_SPECIFIC),
    (".cargo/credentials", True, GROUP_PATH_SPECIFIC),
    (".azure/accessTokens.json", True, GROUP_PATH_SPECIFIC),
    (".gem/credentials", True, GROUP_PATH_SPECIFIC),
    # secret-store data directories (expanded segments + compound suffixes)
    ("config/secrets/database.yaml", True, GROUP_SECRET_STORE),
    ("config/credentials/database.toml", True, GROUP_SECRET_STORE),
    ("credential/database.ini", True, GROUP_SECRET_STORE),
    ("creds/app.properties", True, GROUP_SECRET_STORE),
    ("secret/config.xml", True, GROUP_SECRET_STORE),
    ("secrets/terraform.tfstate.backup", True, GROUP_SECRET_STORE),
    ("vault/prod.auto.tfvars", True, GROUP_SECRET_STORE),
    ("vault/values.tfvars.json", True, GROUP_SECRET_STORE),
    ("config/secrets/README.md", False, None),
    ("internal/secrets/router.go", False, None),
    ("credentials/handler.py", False, None),
    ("secrets-manager/router.py", False, None),
    ("secret-store/database.yaml", False, None),
    ("state/terraform.tfstate", False, None),
    # broad token-boundary basename
    ("data/prod-secrets.yaml", True, GROUP_BROAD_BASENAME),
    ("data/secret.json", True, GROUP_BROAD_BASENAME),
    ("data/prod.secret.json", True, GROUP_BROAD_BASENAME),
    ("data/secretariat.csv", False, None),
    ("data/prodsecret.yaml", False, None),
    ("notes/secret_notes.txt", False, None),
    ("src/secret_redaction.py", False, None),
    ("src/secret_scanner.ts", False, None),
    ("docs/secrets-handling.md", False, None),
    ("prod-secrets.sample.yaml", False, None),
    # path normalization (windows separators + casing)
    (".Docker\\Config.JSON", True, GROUP_PATH_SPECIFIC),
    (".ENV.LOCAL", True, GROUP_EXACT_NAME),
]


@pytest.mark.parametrize("path,is_secret,group", ORACLE)
def test_oracle(path, is_secret, group):
    d = classify_secret_file(path)
    assert d.is_secret is is_secret, f"{path}: {d}"
    if is_secret:
        assert d.group == group, f"{path}: {d}"
    else:
        assert d.group == "none"


class TestConfidence:
    def test_exact_and_path_are_high(self):
        assert classify_secret_file(".env").confidence == "high"
        assert classify_secret_file(".aws/credentials").confidence == "high"
        assert classify_secret_file("certs/server.key").confidence == "high"

    def test_heuristics_are_medium(self):
        assert classify_secret_file("config/secrets/db.yaml").confidence == "medium"
        assert classify_secret_file("data/prod-secrets.yaml").confidence == "medium"

    def test_not_secret_confidence_none(self):
        assert classify_secret_file("main.py").confidence == "none"


class TestOverrides:
    def test_disable_single_group(self):
        assert classify_secret_file("data/prod-secrets.yaml",
                                    disabled_groups=[GROUP_BROAD_BASENAME]).is_secret is False
        # other groups unaffected
        assert classify_secret_file(".env",
                                    disabled_groups=[GROUP_BROAD_BASENAME]).is_secret is True

    def test_key_material_falls_through_to_extension_when_disabled(self):
        d = classify_secret_file("keys/server.pem", disabled_groups=[GROUP_KEY_MATERIAL])
        assert d.is_secret is True
        assert d.group == GROUP_CREDENTIAL_EXT

    def test_disable_both_removes_protection(self):
        assert classify_secret_file(
            "keys/server.pem",
            disabled_groups=[GROUP_KEY_MATERIAL, GROUP_CREDENTIAL_EXT],
        ).is_secret is False

    def test_extra_path_pattern_added(self):
        assert classify_secret_file(
            "project/custom/auth.db",
            extra_secret_path_patterns=["*/custom/auth.db"],
        ).is_secret is True
        assert classify_secret_file("project/custom/auth.db").is_secret is False

    def test_allow_pattern_overrides_everything(self):
        assert classify_secret_file(".env", allow_patterns=[".env"]).is_secret is False
        assert classify_secret_file("server.pem", allow_patterns=["*.pem"]).is_secret is False
        # non-matching allow pattern leaves the verdict intact
        assert classify_secret_file(".env", allow_patterns=["*.pem"]).is_secret is True
