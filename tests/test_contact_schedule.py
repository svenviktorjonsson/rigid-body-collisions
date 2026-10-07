import subprocess
import tempfile
import unittest
from pathlib import Path


class ContactScheduleTests(unittest.TestCase):
    def test_native_portable_and_parallel_full_block_controls(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'research/modern-contact-optimization/schedule_controls.cpp'
        with tempfile.TemporaryDirectory() as directory:
            for flags in [[],['-fopenmp']]:
                binary=Path(directory)/('controls-openmp' if flags else 'controls-portable')
                subprocess.run(['g++','-std=c++17','-O3','-Wall','-Wextra','-Werror',*flags,str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
                result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
                self.assertIn('PASS:',result.stdout)
