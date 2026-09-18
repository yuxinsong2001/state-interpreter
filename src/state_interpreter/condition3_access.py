"""Fail-closed access policy for staged XJTU Condition 3 experiments."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

MODES=("development","validation","blind")
def sha256(path: Path)->str:
 d=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
 return d.hexdigest()

class Condition3AccessPolicy:
 def __init__(self, config: dict, root: Path):
  self.config=config; self.root=root.resolve()
  split=config.get('split',{}); groups=[tuple(split.get(m,())) for m in MODES]
  flat=sum(groups,())
  if groups != [("Bearing3_1","Bearing3_2","Bearing3_3"),("Bearing3_4",),("Bearing3_5",)] or len(flat)!=len(set(flat)):
   raise ValueError('Condition 3 split is not the preregistered 3/1/1 split')
  if config.get('status')!='preregistered_no_bearing3_data_read': raise ValueError('invalid preregistration status')
 @classmethod
 def load(cls,path,root): return cls(json.loads(Path(path).read_text(encoding='utf-8')),Path(root))
 def verify_artifacts(self):
  result={}
  for item in self.config['frozen_artifacts']:
   p=(self.root/item['path']).resolve()
   try: p.relative_to(self.root)
   except ValueError as e: raise ValueError('artifact escapes repository') from e
   if not p.is_file() or sha256(p)!=item['sha256']: raise ValueError(f"frozen artifact mismatch: {item['path']}")
   result[item['path']]=item['sha256']
  return result
 def authorize(self, mode, bearings, token, *, development_record=False, validation_locked=False, blind_record=False):
  if mode not in MODES: raise ValueError('invalid mode')
  expected=tuple(self.config['split'][mode])
  if tuple(bearings)!=expected: raise ValueError(f'{mode} may read exactly {expected}')
  if token!=self.config['tokens'][mode]: raise PermissionError('exact confirmation token required')
  if mode=='validation' and not development_record: raise PermissionError('development record required')
  if mode=='blind' and (not validation_locked or blind_record): raise PermissionError('locked validation required and blind must be unused')
  return expected
