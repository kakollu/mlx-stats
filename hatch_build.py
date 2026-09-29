"""Build hook: mlx-stats is for Apple Silicon Macs only.

The package is Python, but it runs custom Metal kernels on the Mac's GPU. So a wheel is tagged for macOS on arm64 only
(pip will not install it anywhere else), and building one elsewhere - e.g. pip installing from source on Linux, Windows or
an Intel Mac - stops with a plain message instead of installing something that cannot run.
"""
import platform
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class MacOnlyBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            raise RuntimeError(
                "mlx-stats runs only on Apple Silicon Macs (M1 or later, native arm64 Python): it needs the Mac's GPU "
                f"through Metal. This Python is {sys.platform}/{platform.machine()}."
            )
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-macosx_11_0_arm64"
