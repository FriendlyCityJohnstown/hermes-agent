"""The owner policy: a blocked model may never run a child.

A configuration value is not a guarantee. On 2026-11-13 a route leaked into the
shared `delegation.model` and children ran on a metered model nobody chose for
that lane, so the blocklist is enforced where the child's credentials are
resolved — not where they are configured.

These tests drive the real resolver. No network, no credentials, no provider.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import delegate_tool_config as dtc


class BlockedModelMatchingTests(unittest.TestCase):
    def test_normalization_strips_a_provider_prefix(self):
        self.assertEqual(dtc._normalize_model_id("deepseek/deepseek-v4-pro"), "deepseek-v4-pro")
        self.assertEqual(dtc._normalize_model_id("  DEEPSEEK-V4-PRO  "), "deepseek-v4-pro")
        self.assertEqual(dtc._normalize_model_id(None), "")
        self.assertEqual(dtc._normalize_model_id(42), "")

    def test_a_pattern_covers_the_whole_family(self):
        blocked = dtc.blocked_models_from({"blocked_models": ["deepseek-v4-pro"]})
        self.assertIsNotNone(dtc.blocked_model_reason("deepseek-v4-pro", blocked))
        self.assertIsNotNone(dtc.blocked_model_reason("deepseek-v4-pro-20261101", blocked))
        self.assertIsNotNone(dtc.blocked_model_reason("deepseek/deepseek-v4-pro", blocked))

    def test_another_model_is_not_blocked(self):
        blocked = dtc.blocked_models_from({"blocked_models": ["deepseek-v4-pro"]})
        for model in ("deepseek-flash", "claude-opus-5-5", "gpt-6.1-sol", "MiniMax-M3.1-Flash-Preview"):
            with self.subTest(model=model):
                self.assertIsNone(dtc.blocked_model_reason(model, blocked))

    def test_an_empty_or_missing_list_blocks_nothing(self):
        for cfg in ({}, {"blocked_models": []}, {"blocked_models": None}, {"blocked_models": "   "}):
            with self.subTest(cfg=cfg):
                self.assertEqual(dtc.blocked_models_from(cfg), [])
                self.assertIsNone(dtc.blocked_model_reason("deepseek-v4-pro", dtc.blocked_models_from(cfg)))

    def test_the_list_is_deduplicated_and_normalized(self):
        blocked = dtc.blocked_models_from(
            {"blocked_models": ["deepseek-v4-pro", "DEEPSEEK-V4-PRO", " deepseek/v4-pro "]})
        self.assertEqual(blocked, ["deepseek-v4-pro", "v4-pro"])


class BlockedModelRefusalTests(unittest.TestCase):
    """The resolver must REFUSE, not silently substitute a different model."""

    def _route(self, **over):
        # No provider: the resolver would then need that provider's real
        # credentials, which a unit test must not require. The inherit branch
        # resolves the model without contacting anything.
        cfg = {"model": "deepseek-flash", "blocked_models": ["deepseek-v4-pro"]}
        cfg.update(over)
        return cfg

    def test_an_explicit_blocked_model_is_refused(self):
        route = self._route(model="deepseek-v4-pro")
        with self.assertRaises(ValueError) as error:
            dtc._resolve_delegation_credentials(route, _Parent())
        self.assertIn("blocked_models", str(error.exception))

    def test_a_provider_prefixed_blocked_model_is_refused(self):
        route = self._route(model="deepseek/deepseek-v4-pro")
        with self.assertRaises(ValueError):
            dtc._resolve_delegation_credentials(route, _Parent())

    def test_a_dated_variant_of_the_blocked_family_is_refused(self):
        route = self._route(model="deepseek-v4-pro-20261101")
        with self.assertRaises(ValueError):
            dtc._resolve_delegation_credentials(route, _Parent())

    def test_the_allowed_model_still_resolves(self):
        route = self._route()
        creds = dtc._resolve_delegation_credentials(route, _Parent())
        self.assertEqual(creds["model"], "deepseek-flash")

    def test_an_inherited_blocked_parent_model_is_refused(self):
        # delegation.model blank + no provider = the child runs on the parent's
        # model, so the parent's model is what the policy must be asked about.
        route = {"model": "", "blocked_models": ["deepseek-v4-pro"]}
        with self.assertRaises(ValueError) as error:
            dtc._resolve_delegation_credentials(route, _Parent(model="deepseek-v4-pro"))
        self.assertIn("blocked_models", str(error.exception))

    def test_an_inherited_allowed_parent_model_is_fine(self):
        route = {"model": "", "blocked_models": ["deepseek-v4-pro"]}
        creds = dtc._resolve_delegation_credentials(route, _Parent(model="deepseek-flash"))
        self.assertIsNone(creds.get("model"))


class AuxiliaryPathTests(unittest.TestCase):
    """An auxiliary task resolves its OWN model, so it needs the same policy.

    Delegation is not the only subsystem that can put a call on a model: an
    auxiliary task reads `auxiliary.<task>.model` directly and never touches the
    delegation credential resolver. A config-only route is exactly how the
    earlier leak happened, so it is refused at the same kind of chokepoint.
    """

    def setUp(self):
        from agent import auxiliary_client as ac
        self.ac = ac
        self._original = ac._aux_policy_config

    def tearDown(self):
        self.ac._aux_policy_config = self._original

    def test_a_blocked_model_is_refused(self):
        self.ac._aux_policy_config = lambda: {"blocked_models": ["deepseek-v4-pro"]}
        with self.assertRaises(ValueError) as error:
            self.ac._resolve_task_provider_model(task="compression", model="deepseek-v4-pro")
        self.assertIn("blocked_models", str(error.exception))

    def test_the_same_model_is_fine_when_nothing_is_blocked(self):
        self.ac._aux_policy_config = dict
        route = self.ac._resolve_task_provider_model(task="compression", model="deepseek-v4-pro")
        self.assertEqual(route[1], "deepseek-v4-pro", "an empty policy must change nothing")

    def test_an_allowed_model_is_unaffected(self):
        self.ac._aux_policy_config = lambda: {"blocked_models": ["deepseek-v4-pro"]}
        route = self.ac._resolve_task_provider_model(task="compression", model="deepseek-flash")
        self.assertEqual(route[1], "deepseek-flash")

    def test_an_unreadable_config_never_fails_the_call(self):
        # A policy read must not be the reason a side task dies.
        self.ac._aux_policy_config = dict
        route = self.ac._resolve_task_provider_model(task="compression")
        self.assertIsInstance(route, tuple)
        self.assertEqual(len(route), 5)


class _Parent:
    """A stand-in for the parent agent, with only what the resolver reads."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.request_overrides = None


if __name__ == "__main__":
    unittest.main()
