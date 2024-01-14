"""Implementation of parallel tempering plugin for Omnisolver."""
from omnisolver.common.plugin import Plugin, plugin_from_specification
from pkg_resources import resource_stream
from yaml import safe_load

from .sampler import BruteforceGPUSampler


def get_plugin() -> Plugin:
    """Get package name and resource path."""
    specification = safe_load(resource_stream("omnisolver.bruteforce.gpu", "gpu.yml"))
    return plugin_from_specification(specification)


__all__ = ["BruteforceGPUSampler", "get_plugin"]
