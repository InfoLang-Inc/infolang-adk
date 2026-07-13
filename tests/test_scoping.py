from __future__ import annotations

from infolang_adk import DEFAULT_NAMESPACE_PREFIX, default_namespace_for


def test_default_namespace_joins_prefix_app_and_user() -> None:
    assert default_namespace_for("myapp", "u1") == "adk-myapp-u1"


def test_default_namespace_prefix_override() -> None:
    assert default_namespace_for("myapp", "u1", prefix="prod") == "prod-myapp-u1"


def test_default_namespace_prefix_constant() -> None:
    assert DEFAULT_NAMESPACE_PREFIX == "adk"


def test_default_namespace_sanitizes_unsafe_chars() -> None:
    ns = default_namespace_for("team/reporting", "user@example.com")
    assert ns == "adk-team-reporting-user-example.com"


def test_default_namespace_sanitized_collisions_are_documented_behavior() -> None:
    # "team/a" and "team a" both sanitize to "team-a" -- this is the documented
    # collision tradeoff of the default scheme (see README "Namespace scoping").
    assert default_namespace_for("team/a", "u1") == default_namespace_for("team a", "u1")


def test_default_namespace_blank_segment_falls_back_to_unknown() -> None:
    assert default_namespace_for("   ", "u1") == "adk-unknown-u1"
