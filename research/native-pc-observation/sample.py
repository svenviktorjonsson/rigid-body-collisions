"""Three brief read-only x86_64 native-PC observations, never register writes."""
import ctypes,datetime,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
plan=json.loads((D/'plan.json').read_text());pid=int(sys.argv[1]);destination=D/'observation.json'
assert not destination.exists() and platform.machine()=='x86_64'
exe=Path(os.readlink(f'/proc/{pid}/exe'));binary_hash=hashlib.sha256(exe.read_bytes()).hexdigest()
assert binary_hash==plan['expected_binary_sha256']
lib=ctypes.CDLL(None,use_errno=True);lib.ptrace.restype=ctypes.c_long
lib.ptrace.argtypes=[ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p,ctypes.c_void_p]
report={'plan_sha256':hashlib.sha256((D/'plan.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pid':pid,'binary_sha256':binary_hash,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observations':[],'register_or_memory_writes':False,'timing_ranking_qualified':False}
for index in range(plan['samples']):
 row={'sample':index,'attempt_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};attached=False;start=time.perf_counter()
 try:
  ctypes.set_errno(0)
  if lib.ptrace(16,pid,None,None)!=0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
  attached=True;_,status=os.waitpid(pid,0)
  if not os.WIFSTOPPED(status):raise RuntimeError('Tracee did not report a stopped state')
  registers=(ctypes.c_ulonglong*27)()
  if lib.ptrace(12,pid,None,ctypes.cast(registers,ctypes.c_void_p))!=0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
  row['instruction_pointer']=int(registers[16]);row['stop_signal']=os.WSTOPSIG(status)
  row['maps']=Path(f'/proc/{pid}/maps').read_text()
 except Exception as error:row['error']=str(error)
 finally:
  if attached:
   ctypes.set_errno(0);rc=lib.ptrace(17,pid,None,None)
   row['detach_succeeded']=rc==0
   if rc!=0:row['detach_errno']=ctypes.get_errno()
  row['attach_read_detach_elapsed_s']=time.perf_counter()-start
 report['observations'].append(row)
 destination.write_text(json.dumps(report,indent=2)+'\n')
 if 'error' in row or row.get('detach_succeeded')is False:break
for row in report['observations']:
 if 'instruction_pointer' not in row:continue
 pc=row['instruction_pointer']
 for line in row['maps'].splitlines():
  parts=line.split(maxsplit=5);left,right=(int(v,16)for v in parts[0].split('-'))
  if left<=pc<right and len(parts)==6 and parts[5].startswith('/'):
   module=Path(parts[5]);relative=pc-left+int(parts[2],16);row['module']=str(module);row['relative_pc']=relative
   row['module_sha256']=hashlib.sha256(module.read_bytes()).hexdigest()
   command=['nm','-n','--defined-only',str(module)]
   if module!=exe:command.insert(1,'-D')
   result=subprocess.run(command,text=True,capture_output=True)
   row['symbol_lookup_exit_code']=result.returncode;nearest=None
   for text in result.stdout.splitlines():
    values=text.split(maxsplit=2)
    if len(values)!=3 or values[1] not in ('T','t','W','w'):continue
    try:address=int(values[0],16)
    except ValueError:continue
    if address<=relative and(nearest is None or address>nearest[0]):nearest=(address,values[2])
   if nearest:row['nearest_symbol']={'name':nearest[1],'offset_bytes':relative-nearest[0]}
   break
 row.pop('maps',None)
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
