import base64,concurrent.futures,hashlib,json,shutil,threading,time
from pathlib import Path
import google_crc32c,requests
ROOT=Path('/root/gpufree-data/pi05-repro')
CHUNK=1048576
LOCAL=threading.local()
def valid(path,item):
 if not path.exists() or path.stat().st_size!=int(item['size']):return False
 with path.open('rb') as f:
  if 'md5Hash' in item:digest=hashlib.file_digest(f,'md5').digest();expected=item['md5Hash']
  else:
   crc=google_crc32c.Checksum()
   while block:=f.read(8*1048576):crc.update(block)
   digest=crc.digest();expected=item['crc32c']
 return base64.b64encode(digest).decode()==expected

def fetch(task):
 item,start,end,dest=task
 if dest.exists() and dest.stat().st_size==end-start+1:return dest.stat().st_size
 if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
 url='https://storage.googleapis.com/openpi-assets/'+item['name']
 for attempt in range(6):
  try:
   with LOCAL.session.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,45)) as r:
    assert r.status_code==206,(r.status_code,item['name'])
    assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
    data=r.content
   assert len(data)==end-start+1
   temp=dest.with_suffix('.incomplete');temp.write_bytes(data);temp.replace(dest)
   return len(data)
  except Exception as e:
   print('RETRY',item['name'].split('/')[-1],start,attempt+1,type(e).__name__,flush=True)
   if attempt==5:raise
   time.sleep(min(attempt+1,5))

for model in ['pi05_base','pi05_libero']:
 manifest=json.loads((ROOT/'logs'/f'{model}-asset-list.json').read_text())
 tasks=[];plans=[];present=0
 for item in manifest['items']:
  target=ROOT/'checkpoints'/model/item['name'].removeprefix('checkpoints/'+model+'/')
  target.parent.mkdir(parents=True,exist_ok=True)
  if valid(target,item):present+=int(item['size']);continue
  part=target.with_name(target.name+'.part')
  if target.exists() and (not part.exists() or target.stat().st_size>part.stat().st_size):target.replace(part)
  part.touch(exist_ok=True)
  start=part.stat().st_size
  assert start<=int(item['size']),target
  present+=start
  chunkdir=ROOT/'cache/checkpoint-chunks'/model/target.name
  chunkdir.mkdir(parents=True,exist_ok=True)
  chunks=[]
  for offset in range(start,int(item['size']),CHUNK):
   end=min(offset+CHUNK,int(item['size']))-1
   chunk=chunkdir/f'{offset:012d}-{end:012d}.chunk'
   tasks.append((item,offset,end,chunk));chunks.append(chunk)
  plans.append((item,target,part,chunks))
 print('START',model,'existing_GB',round(present/1e9,3),'ranges',len(tasks),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
  pending={pool.submit(fetch,t) for t in tasks}
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=20,return_when=concurrent.futures.FIRST_EXCEPTION)
   for f in done:present+=f.result()
   print('PROGRESS',model,round(present/1e9,3),'GB','remaining',len(pending),flush=True)
 for item,target,part,chunks in plans:
  with part.open('ab') as out:
   for chunk in chunks:
    with chunk.open('rb') as source:shutil.copyfileobj(source,out,1048576)
    out.flush();chunk.unlink()
  assert valid(part,item),('official checksum mismatch',target)
  part.replace(target)
  print('VERIFIED',model,target.name,flush=True)
 (ROOT/'checkpoints'/model/'server-verified.txt').write_text('All official object sizes and MD5/CRC32C checksums verified.\n')
 print('MODEL_READY',model,flush=True)
