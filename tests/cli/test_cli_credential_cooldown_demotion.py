"""Interactive CLI: route around a primary whose whole pool is serving a 429.

``resolve_runtime_provider`` succeeds for a drained provider — it hands back a
key — so without a check the CLI spends one doomed request before the agent's
own fallback engages.  For a ``hermes chat -q`` worker (how a kanban card is
dispatched) that is one doomed request per card.

The demotion is turn-scoped: the snapshot taken here is put back at the top of
the next turn once the window the provider reported has elapsed, so the switch
lands between turns and the session returns on its own.
"""

from __future__ import annotations

import sys
import time
import types

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY


@pytest.fixture(autouse=True)
def _stub_cli_module(monkeypatch, tmp_path):
    """``_ensure_runtime_credentials`` lazily imports three names from ``cli``.

    Importing the real ``cli`` module drags in the whole interactive stack for
    a test about credential routing, so stand in for exactly those names.
    Pointing HERMES_HOME at a temp dir keeps the fallback chain's own pool
    probe off the developer's real credential store.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    stub = types.ModuleType("cli")

    class _Console:
        def print(self, *args, **kwargs):
            pass

    stub.ChatConsole = _Console
    stub._cprint = lambda *args, **kwargs: None
    stub.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "cli", stub)
    yield


class _Shell(CLIAgentSetupMixin):
    """The attributes ``_ensure_runtime_credentials`` reads and writes."""

    def __init__(self, chain):
        self.requested_provider = "gemini"
        self.model = "gemini-3.7-flash"
        self.provider = None
        self.api_key = None
        self.base_url = None
        self.api_mode = None
        self.acp_command = None
        self.acp_args = []
        self.agent = None
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._credential_pool = None
        self._active_agent_route_signature = None
        self._fallback_model = chain

    def _normalize_model_for_provider(self, provider):
        """The real normalizer, not a stub.

        HermesCLI rewrites ``self.model`` here for most providers
        ("anthropic/claude-sonnet-4.6" -> "claude-sonnet-4-6"). A stub that
        returned False hid a bug where the demotion snapshot recorded the
        PRE-normalization name, so every later turn read as "the user switched
        models" and the primary never came back. Calling the real function
        keeps these tests honest about that.
        """
        from hermes_cli.model_normalize import normalize_model_for_provider

        normalized = normalize_model_for_provider(self.model, provider)
        if normalized and normalized != self.model:
            self.model = normalized
            return True
        return False


_CHAIN = [{"provider": "openai-codex", "model": "gpt-5.6-luna"}]


def _cooling_primary(until):
    return {
        "provider": "gemini",
        "api_mode": "chat_completions",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "sk-test-000-not-a-real-key",
        "source": "env/config",
        CREDENTIALS_COOLING_DOWN_KEY: until,
    }


def _healthy_fallback():
    return {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "fake-codex-token",
        "source": "env/config",
    }


def _resolver(monkeypatch, primary, calls):
    """Serve the primary for ``requested=gemini``, the fallback otherwise."""

    def _resolve(**kwargs):
        requested = (kwargs.get("requested") or "gemini").strip().lower()
        calls.append(requested)
        if requested == "gemini":
            return dict(primary())
        return _healthy_fallback()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
    )


def test_a_rate_limited_primary_runs_the_turn_on_a_fallback(monkeypatch):
    """The doomed request is never sent: the chain is consulted first."""
    until = time.time() + 1800
    shell = _Shell(_CHAIN)
    _resolver(monkeypatch, lambda: _cooling_primary(until), [])

    assert shell._ensure_runtime_credentials() is True

    assert shell.provider == "openai-codex"
    assert shell.model == "gpt-5.6-luna"
    assert shell.api_key == "fake-codex-token"
    # The primary we owe a return to is remembered, with the window that
    # decides when to take it.
    assert shell._cooldown_demotion["requested_provider"] == "gemini"
    assert shell._cooldown_demotion["model"] == "gemini-3.7-flash"
    assert shell._cooldown_demotion["until"] == until


def test_an_explicit_endpoint_pin_does_not_follow_the_provider(monkeypatch):
    """``--api-key`` / ``--base-url`` name the PRIMARY's endpoint.

    Carrying them past the switch would resolve the fallback's provider while
    still holding the primary's bearer token — a credential crossing an origin
    boundary. They belong to the snapshot until the return.
    """
    shell = _Shell(_CHAIN)
    shell._explicit_api_key = "primary-pin"
    shell._explicit_base_url = "https://primary.example.test/v1"
    calls: list[str] = []

    _resolver(monkeypatch, lambda: _cooling_primary(time.time() + 5), calls)
    assert shell._ensure_runtime_credentials() is True

    assert shell.provider == "openai-codex"
    assert shell._explicit_api_key is None
    assert shell._explicit_base_url is None

    healthy = dict(_cooling_primary(0))
    healthy.pop(CREDENTIALS_COOLING_DOWN_KEY)
    _resolver(monkeypatch, lambda: healthy, calls)
    _advance_clock_past(monkeypatch, shell._cooldown_demotion["until"])
    assert shell._ensure_runtime_credentials() is True

    assert shell._explicit_api_key == "primary-pin"
    assert shell._explicit_base_url == "https://primary.example.test/v1"


def test_the_primary_returns_on_the_first_turn_after_the_window_lifts(monkeypatch):
    """Return is automatic and free — the primary is not re-probed early."""
    shell = _Shell(_CHAIN)
    calls: list[str] = []

    # Turn 1: benched, so the session demotes.
    _resolver(monkeypatch, lambda: _cooling_primary(time.time() + 5), calls)
    assert shell._ensure_runtime_credentials() is True
    assert shell.provider == "openai-codex"

    # Turn 2: still inside the window. The configured primary is not asked
    # again — that is the wasted call this whole path exists to avoid.
    calls.clear()
    assert shell._ensure_runtime_credentials() is True
    assert shell.provider == "openai-codex"
    assert "gemini" not in calls

    # Turn 3: the clock passes the window the provider itself reported, and
    # the pool is healthy again. The window is what drives the return, so let
    # a controlled clock cross it rather than editing the snapshot by hand.
    healthy = dict(_cooling_primary(0))
    healthy.pop(CREDENTIALS_COOLING_DOWN_KEY)
    calls.clear()
    _resolver(monkeypatch, lambda: healthy, calls)
    _advance_clock_past(monkeypatch, shell._cooldown_demotion["until"])

    assert shell._ensure_runtime_credentials() is True
    assert shell.requested_provider == "gemini"
    assert shell.provider == "gemini"
    assert shell.model == "gemini-3.7-flash"
    assert shell._cooldown_demotion is None
    assert "gemini" in calls


def test_a_cooling_primary_is_kept_when_no_fallback_is_usable(monkeypatch):
    """A cooldown DEMOTES a provider; it does not disqualify it.

    With nothing to demote to, the real upstream 429 beats refusing to run.
    """
    shell = _Shell([])
    _resolver(monkeypatch, lambda: _cooling_primary(time.time() + 1800), [])

    assert shell._ensure_runtime_credentials() is True

    assert shell.provider == "gemini"
    assert shell.model == "gemini-3.7-flash"
    assert getattr(shell, "_cooldown_demotion", None) is None


def _advance_clock_past(monkeypatch, deadline):
    """Move the mixin's clock just past *deadline*.

    Patches the module's `time` reference rather than the stdlib module, so
    nothing else in the process sees a doctored clock.
    """
    import hermes_cli.cli_agent_setup_mixin as mixin

    monkeypatch.setattr(
        mixin, "time", types.SimpleNamespace(time=lambda: deadline + 1)
    )


def test_a_fallback_defined_by_its_own_url_keeps_that_url_next_turn(monkeypatch):
    """A chain entry can name an endpoint by URL instead of a registered name.

    Clearing the explicit pins on demotion would leave the next turn resolving
    that entry by provider name alone — landing somewhere else, or failing.
    """
    chain = [{
        "provider": "custom",
        "model": "local-model",
        "base_url": "https://fallback.example.test/v1",
        "api_key": "sk-test-000-not-a-real-fallback-key",
    }]
    shell = _Shell(chain)
    seen: list[dict] = []

    def _resolve(**kwargs):
        seen.append(kwargs)
        if (kwargs.get("requested") or "gemini") == "gemini" and not kwargs.get(
            "explicit_base_url"
        ):
            return _cooling_primary(time.time() + 1800)
        return {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": kwargs.get("explicit_base_url") or "https://wrong.example.test",
            "api_key": kwargs.get("explicit_api_key") or "",
            "source": "direct-alias",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
    )
    assert shell._ensure_runtime_credentials() is True
    assert shell.base_url == "https://fallback.example.test/v1"

    # Turn 2 must resolve the SAME endpoint, not just the provider name.
    seen.clear()
    assert shell._ensure_runtime_credentials() is True
    assert shell.base_url == "https://fallback.example.test/v1"
    assert seen[0]["explicit_base_url"] == "https://fallback.example.test/v1"


def test_an_explicit_model_switch_survives_the_windows_expiry(monkeypatch):
    """`/model` during a demotion is a deliberate choice made with the
    fallback in view. The automatic return is owed to a session that never
    moved — reverting one the user has since steered would undo that choice
    silently, and hand back the previous provider's credentials with it.
    """
    shell = _Shell(_CHAIN)
    calls: list[str] = []
    _resolver(monkeypatch, lambda: _cooling_primary(time.time() + 5), calls)
    assert shell._ensure_runtime_credentials() is True
    assert shell.provider == "openai-codex"

    # The user takes over mid-cooldown, the way cmd_model does.
    shell.requested_provider = "anthropic"
    shell.model = "claude-fable-5-1"
    shell._explicit_api_key = None
    shell._explicit_base_url = None

    def _resolve(**kwargs):
        calls.append(kwargs.get("requested") or "")
        return {
            "provider": kwargs.get("requested"),
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-test-000-not-a-real-key",
            "source": "env/config",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
    )
    _advance_clock_past(monkeypatch, shell._cooldown_demotion["until"])

    assert shell._ensure_runtime_credentials() is True

    assert shell.requested_provider == "anthropic"
    assert shell.model == "claude-fable-5-1"
    assert shell.provider == "anthropic"
    # The snapshot is spent either way, so a later window cannot resurrect it.
    assert getattr(shell, "_cooldown_demotion", None) is None


def test_a_fallback_whose_model_gets_normalized_still_returns_the_primary(monkeypatch):
    """Regression: the session must not be stranded on the fallback.

    ``_normalize_model_for_provider`` rewrites ``self.model`` after the
    demotion records where it put the session. Stamping the pre-normalization
    name made the next turn look like a user switch, so the pending return was
    dropped and the primary never came back until restart -- the exact failure
    the return exists to prevent. ``anthropic`` normalizes; the chain used by
    the other tests happens not to, which is why this needs its own.
    """
    chain = [{"provider": "anthropic", "model": "anthropic/claude-sonnet-4.6"}]
    shell = _Shell(chain)
    calls: list[str] = []

    def _resolve(**kwargs):
        requested = (kwargs.get("requested") or "gemini").strip().lower()
        calls.append(requested)
        if requested == "gemini":
            return dict(_cooling_primary(time.time() + 5))
        return {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-test-000-not-a-real-key",
            "source": "env/config",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
    )
    assert shell._ensure_runtime_credentials() is True
    assert shell.provider == "anthropic"
    # The normalizer really did rewrite it, which is the precondition.
    assert shell.model == "claude-sonnet-4-6"

    # A quiet turn inside the window must not look like a user switch.
    assert shell._ensure_runtime_credentials() is True
    assert shell._cooldown_demotion is not None

    # And once the window lifts, the primary comes back.
    healthy = dict(_cooling_primary(0))
    healthy.pop(CREDENTIALS_COOLING_DOWN_KEY)
    _resolver(monkeypatch, lambda: healthy, calls)
    _advance_clock_past(monkeypatch, shell._cooldown_demotion["until"])
    assert shell._ensure_runtime_credentials() is True

    assert shell.requested_provider == "gemini"
    assert shell.model == "gemini-3.7-flash"
    assert shell._cooldown_demotion is None
