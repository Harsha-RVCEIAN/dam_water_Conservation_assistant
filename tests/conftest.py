"""Pytest root configuration and test session fixtures.

Ensures that testing environments (including clean CI runners without local mlruns/)
automatically have access to the registered model Version 1 provisioned from tracked
repository artifacts.
"""

from __future__ import annotations

import logging
import pytest

from src.models.provision_ci_registry import ensure_ci_registry_provisioned

logger = logging.getLogger("krs_dam.tests.conftest")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Session-level fixture ensuring MLflow Model Registry is provisioned if absent."""
    try:
        provisioned = ensure_ci_registry_provisioned()
        if provisioned:
            logger.info("Auto-provisioned MLflow Model Registry for test session.")
    except Exception as exc:
        logger.warning("Could not auto-provision MLflow Model Registry: %s", exc)

