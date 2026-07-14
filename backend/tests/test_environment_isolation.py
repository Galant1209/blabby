"""Unit coverage for deployment-environment isolation."""

import pytest

import main


@pytest.mark.parametrize("app_env", ["", "test", "prod", "STAGINGG"])
def test_unknown_app_environment_fails_closed(app_env):
    with pytest.raises(RuntimeError, match="APP_ENV"):
        main._validate_environment_isolation(app_env, "", "")


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_non_development_requires_expected_project_ref(app_env):
    with pytest.raises(RuntimeError, match="EXPECTED_SUPABASE_PROJECT_REF"):
        main._validate_environment_isolation(
            app_env, "", "https://actual-project.supabase.co"
        )


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_staging_or_production_project_mismatch_fails_closed(app_env):
    with pytest.raises(RuntimeError) as exc:
        main._validate_environment_isolation(
            app_env,
            "expected-project",
            "https://different-project.supabase.co",
        )
    message = str(exc.value)
    assert "different-project" not in message
    assert "expected-project" not in message


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_expected_project_is_accepted(app_env):
    main._validate_environment_isolation(
        app_env,
        "expected-project",
        "https://expected-project.supabase.co",
    )


def test_development_can_run_without_supabase_configuration():
    main._validate_environment_isolation("development", "", "")


def test_custom_or_malformed_url_fails_when_project_ref_is_expected():
    with pytest.raises(RuntimeError, match="does not match"):
        main._validate_environment_isolation(
            "staging", "expected-project", "not-a-supabase-project-url"
        )
