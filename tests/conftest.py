"""Pytest session setup (loaded before any test module is imported).

The model/persona layer validates ``NVIDIA_API_KEY`` at *import* time — a
deliberate fail-fast for production (``src/personas/base.py``). The unit suite
mocks the LLM and never makes a real API call, so we inject a dummy key here,
before collection imports those modules. This lets the whole suite run with **no
real credentials** — in CI, for new contributors, and offline — without
weakening the production check.

Set ``NVIDIA_API_KEY`` in your environment to run any opt-in integration checks.
"""
import os

# Cover both the unset and empty-string cases (CI has it unset; a blank .env
# value would also otherwise trip the import-time validation).
if not os.environ.get("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = "test-key-not-used"
