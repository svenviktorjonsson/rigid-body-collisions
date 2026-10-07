"""Zero-copy native independent supported-contact updates, field-major Float64.

The existing planar law and energy/branch checks are unchanged. A PreparedBatch
owns immutable validated inputs; material/input validation is preparation cost,
not silently omitted from an end-to-end timing. Parallelism is explicit.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parent
INPUT_FIELDS=('mass_kg','inertia_kg_m2','radius_m','normal_load_N','drive_force_N',
              'velocity_m_s','omega_rad_s','spin_rad_s','duration_s','mu_s','mu_d',
              'mu_r','rolling_length_m','mu_n','spin_length_m')
OUTPUT_FIELDS=('velocity_m_s','omega_rad_s','spin_rad_s','distance_m',
               'tangent_impulse_Ns','independent_rolling_impulse_Nms',
               'independent_spin_impulse_Nms','sliding_loss_J','rolling_loss_J',
               'spin_loss_J','energy_residual_J')
ABI_VERSION=1
_POINTER=ctypes.POINTER(ctypes.c_double)
_LIBRARY=None


def _library():
    global _LIBRARY
    if _LIBRARY is not None:return _LIBRARY
    sources=[ROOT/'supported_backend/batch.cpp',ROOT/'supported_backend/kernel.h',ROOT/'supported_backend/batch.h']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:20]
    folder=ROOT/'build/supported-batch'/digest;folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'supported.so'
    if not binary.exists():
        temporary=folder/f'supported-{os.getpid()}.so'
        command=['g++','-std=c++17','-O3','-Wall','-Wextra','-Werror','-fopenmp','-fPIC','-shared',str(sources[0]),'-o',str(temporary)]
        result=subprocess.run(command,capture_output=True,text=True)
        (folder/f'compile-{os.getpid()}.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
        result.check_returncode();temporary.replace(binary)
    library=ctypes.CDLL(str(binary))
    library.supported_batch.argtypes=[_POINTER,_POINTER,ctypes.c_size_t,ctypes.c_int]
    library.supported_batch.restype=ctypes.c_size_t
    library.supported_max_threads.restype=ctypes.c_int
    for name in ['supported_abi_version','supported_input_fields','supported_output_fields']:
        getattr(library,name).restype=ctypes.c_uint32
    library.supported_validate.argtypes=[_POINTER,ctypes.c_size_t]
    library.supported_validate.restype=ctypes.c_size_t
    if (library.supported_abi_version(),library.supported_input_fields(),library.supported_output_fields())!=(ABI_VERSION,len(INPUT_FIELDS),len(OUTPUT_FIELDS)):
        raise RuntimeError('Incompatible supported-contact ABI')
    _LIBRARY=library;return library


class PreparedBatch:
    """Owned immutable inputs validated once; each run writes every output field.

    Each column is independent, not a contact group. Runs may reuse the same
    initial states (e.g. a parameter sweep); this API does not integrate a world.
    Input ownership prevents later mutation from invalidating validation.
    """
    def __init__(self,inputs):
        values=np.array(inputs,dtype=np.float64,order='C',copy=True)
        if values.ndim!=2 or values.shape[0]!=len(INPUT_FIELDS) or not np.isfinite(values).all():
            raise ValueError('Finite field-major (15, count) Float64 inputs required')
        if np.any(values[[0,1,2,8]]<=0) or np.any(values[[3,9,10,11,12,13,14]]<0):
            raise ValueError('Positive mass/inertia/radius/duration and nonnegative capacities required')
        if np.any(values[10]>values[9]):raise ValueError('mu_d <= mu_s required')
        if np.any((values[11]>0)&(values[12]==0)) or np.any((values[13]>0)&(values[14]==0)):
            raise ValueError('Nonzero angular resistance requires a physical moment length')
        values.flags.writeable=False
        self._inputs=values;self.count=values.shape[1];self._native=_library()

    def run(self,*,threads=1,out=None):
        """Checked zero-copy call; allocation excluded only when out is supplied."""
        if type(threads)!=int or not 1<=threads<=self._native.supported_max_threads():
            raise ValueError('Explicit positive thread count within host CPU count required')
        if out is None:out=np.empty((len(OUTPUT_FIELDS),self.count),dtype=np.float64)
        if not isinstance(out,np.ndarray) or out.dtype!=np.float64 or out.shape!=(len(OUTPUT_FIELDS),self.count) or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError('Writable contiguous Float64 (11, count) output required')
        if np.shares_memory(out,self._inputs):raise ValueError('Input/output storage must not alias')
        failure=self._native.supported_batch(self._inputs.ctypes.data_as(_POINTER),out.ctypes.data_as(_POINTER),self.count,threads)
        if failure==ctypes.c_size_t(-2).value:raise RuntimeError('Invalid native supported-batch call arguments')
        if failure!=ctypes.c_size_t(-1).value:
            raise RuntimeError(f'Supported-contact branch/energy gate failed at body {failure}; discard all outputs')
        return out
