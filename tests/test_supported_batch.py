import unittest
import ctypes
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from supported_batch import PreparedBatch,OUTPUT_FIELDS
from supported_contact import Resistance,advance_planar


class SupportedBatchTests(unittest.TestCase):
    def test_c_compatible_header_version_validation_and_global_errors(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'probe.c';target=Path(directory)/'probe.o'
            source.write_text('#include "batch.h"\nsize_t probe(const double* p) { return supported_validate(p,0); }\n')
            subprocess.run(['gcc','-std=c11','-Wall','-Wextra','-Werror','-I',str(root/'supported_backend'),'-c',str(source),'-o',str(target)],check=True,capture_output=True)
        inputs=np.ascontiguousarray(self.inputs());batch=PreparedBatch(inputs);lib=batch._native;ptr=ctypes.POINTER(ctypes.c_double)
        self.assertEqual(lib.supported_abi_version(),1)
        self.assertEqual(lib.supported_validate(inputs.ctypes.data_as(ptr),400),ctypes.c_size_t(-1).value)
        inputs[0,17]=-1
        self.assertEqual(lib.supported_validate(inputs.ctypes.data_as(ptr),400),17)
        self.assertEqual(lib.supported_validate(None,1),ctypes.c_size_t(-2).value)
        self.assertEqual(lib.supported_batch(None,None,0,0),ctypes.c_size_t(-2).value)

    def inputs(self):
        return np.loadtxt(Path(__file__).resolve().parents[1]/'research/contact-gap-fix/audit-v2/controls.txt',skiprows=1)[:,:15].T

    def test_all_400_reference_cases_and_thread_count_determinism(self):
        inputs=self.inputs();batch=PreparedBatch(inputs);single=batch.run(threads=1);parallel=batch.run(threads=4)
        np.testing.assert_array_equal(single,parallel)
        for k in range(inputs.shape[1]):
            m,I,R,N,F,v,w,s,h,mus,mud,mur,ar,mun,an=inputs[:,k]
            reference=advance_planar(mass_kg=m,inertia_kg_m2=I,radius_m=R,normal_load_N=N,drive_force_N=F,velocity_m_s=v,omega_rad_s=w,spin_rad_s=s,duration_s=h,material=Resistance(mus,mud,mur,ar,mun,an))
            speed=max(abs(v),R*abs(w),R*abs(s),abs(F/m)*h,1e-30)
            scales=np.array([speed,speed/R,speed/R,speed*h,m*speed,m*R*speed,m*R*speed,m*speed*speed,m*speed*speed,m*speed*speed,m*speed*speed])
            np.testing.assert_allclose(single[:,k]/scales,np.array([reference[f] for f in OUTPUT_FIELDS])/scales,atol=1e-10,rtol=1e-10)

    def test_owned_inputs_and_zero_copy_output(self):
        inputs=self.inputs();batch=PreparedBatch(inputs);expected=batch.run();inputs[:]=0
        out=np.empty_like(expected);self.assertIs(batch.run(out=out),out)
        np.testing.assert_array_equal(out,expected)

    def test_size_scaling_and_small_real_frictionless_drive(self):
        inputs=[]
        for R in [1e-12,1e-9,1e-6,1e-3,1.,1e3,1e6]:
            inputs.append([1.,.4*R*R,R,9.81,1e-8,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.])
        result=PreparedBatch(np.array(inputs).T).run(threads=4)
        np.testing.assert_allclose(result[0],1e-8,rtol=1e-15,atol=0)
        np.testing.assert_array_equal(result[1],0.)

    def test_axial_arrest_is_exact_and_does_not_reverse_roundoff(self):
        inputs=self.inputs();inputs[5:7]=0;inputs[4]=0;inputs[8]=1e6
        out=PreparedBatch(inputs).run();np.testing.assert_array_equal(out[2],0.)

    def test_invalid_inputs_threads_outputs_and_nonfinite_response(self):
        inputs=self.inputs()
        with self.assertRaises(ValueError):PreparedBatch(inputs[:14])
        with self.assertRaises(ValueError):PreparedBatch(inputs*np.nan)
        inputs[10]=inputs[9]+1
        with self.assertRaises(ValueError):PreparedBatch(inputs)
        batch=PreparedBatch(self.inputs())
        with self.assertRaises(ValueError):batch.run(threads=0)
        with self.assertRaises(ValueError):batch.run(out=np.empty((11,400),dtype=np.float32))
        extreme=self.inputs()[:,:1].copy();extreme[5]=1e308
        with self.assertRaises(RuntimeError):PreparedBatch(extreme).run()
        # A static force/couple can overflow its impulse without doing work.
        held=np.array([1e308,1e308,1.,1e308,1e308,0.,0.,0.,1e308,1.,0.,1.,1.,0.,0.])[:,None]
        with self.assertRaises(RuntimeError):PreparedBatch(held).run()

    def test_empty_batch_is_well_defined(self):
        self.assertEqual(PreparedBatch(np.empty((15,0))).run().shape,(11,0))
