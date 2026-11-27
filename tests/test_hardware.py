import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from workbench import hardware


@unittest.skipUnless(os.name == "nt", "Windows hardware discovery")
class WindowsHardwareTests(unittest.TestCase):
    def test_inventory_reports_the_current_cpu_memory_and_multiple_video_adapters(self):
        memory = 32 * 1024**3
        inventory = {
            "cpu": "Example Intel processor",
            "memory": memory,
            "gpus": ["Intel Graphics", "NVIDIA Example"],
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(hardware, "_read_command", side_effect=[json.dumps(inventory), "NVIDIA Example, 8192\n"]),
            patch.object(os, "cpu_count", return_value=16),
        ):
            self.assertEqual(hardware.hardware_inventory(), {
                "cpu": {"name": "Example Intel processor", "logical_cores": 16},
                "memory": {"total_bytes": memory},
                "gpus": [
                    {"name": "Intel Graphics", "memory_bytes": None},
                    {"name": "NVIDIA Example", "memory_bytes": 8 * 1024**3},
                ],
            })

    def test_inventory_keeps_amd_and_intel_adapters_without_an_nvidia_driver(self):
        memory = 64 * 1024**3
        inventory = {
            "cpu": "Example AMD processor",
            "memory": memory,
            "gpus": ["AMD Graphics", None, "Intel Graphics", ""],
        }
        with (
            patch.object(Path, "is_file", return_value=False),
            patch.object(hardware, "_read_command", return_value=json.dumps(inventory)) as query,
            patch.object(os, "cpu_count", return_value=24),
        ):
            self.assertEqual(hardware.hardware_inventory(), {
                "cpu": {"name": "Example AMD processor", "logical_cores": 24},
                "memory": {"total_bytes": memory},
                "gpus": [
                    {"name": "AMD Graphics", "memory_bytes": None},
                    {"name": "Intel Graphics", "memory_bytes": None},
                ],
            })
        query.assert_called_once()

    def test_timed_out_queries_leave_unavailable_fields_empty(self):
        with (
            patch.object(Path, "is_file", return_value=False),
            patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("powershell", 6)),
            patch.dict(os.environ, {"PROCESSOR_IDENTIFIER": ""}),
            patch.object(os, "cpu_count", return_value=4),
        ):
            self.assertEqual(hardware.hardware_inventory(), {
                "cpu": {"name": None, "logical_cores": 4},
                "memory": {"total_bytes": None},
                "gpus": [],
            })
