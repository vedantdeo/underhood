"""Tools a model calls, and the guardrails around them, shared by aigent's agents and underhood's
fine-tunes so a model is trained against exactly what it later runs.

Imports nothing else from underhood, only the standard library, so it can be lifted unchanged; its
constants live in `tools.config`, which travels beside it. tests/gpu/test_boundary.py holds that it
loads no torch.
"""
