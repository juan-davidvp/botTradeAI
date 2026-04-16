"""
tests/conftest.py
Fixtures compartidas entre todos los tests de Fase 9.
"""
import os
import pytest

# Inyectar variables de entorno ficticias para tests que instancian EToroClient
os.environ.setdefault("ETORO_API_KEY",  "TEST_API_KEY")
os.environ.setdefault("ETORO_USER_KEY", "TEST_USER_KEY")
