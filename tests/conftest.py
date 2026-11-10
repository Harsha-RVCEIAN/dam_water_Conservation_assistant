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
    """Session-level fixture ensuring MLflow Model Registry is provisioned if absent.

    Fails explicitly with RuntimeError if provisioning cannot be completed,
    preventing silent downstream test failure cascades.
    """
    try:
        provisioned = ensure_ci_registry_provisioned()
        if provisioned:
            logger.info("Auto-provisioned MLflow Model Registry for test session.")
    except Exception as exc:
        logger.error("Failed to provision MLflow Model Registry: %s", exc)
        raise RuntimeError(
            f"Test session environment setup failed: could not provision MLflow Model Registry. "
            f"Error details: {exc}"
        ) from exc

