import base64,hashlib,json,sys
import google_crc32c
from pathlib import Path
root=Path('/root/gpufree-data/pi05-repro')
model=sys.argv[1]
manifest=json.loads((root/'logs'/f'{model}-asset-list.json').read_text())
for x in manifest['items']:
    p=root/'checkpoints'/model/x['name'].removeprefix('checkpoints/'+model+'/')
    assert p.stat().st_size==int(x['size']),p
    with p.open('rb') as f:
        if 'md5Hash' in x:
            digest=hashlib.file_digest(f,'md5').digest(); expected=x['md5Hash']
        else:
            crc=google_crc32c.Checksum()
            while block:=f.read(8*1024*1024): crc.update(block)
            digest=crc.digest(); expected=x['crc32c']
    assert base64.b64encode(digest).decode()==expected,p
    print('VERIFIED',p.name,flush=True)
(root/'checkpoints'/model/'server-verified.txt').write_text('All official object sizes and MD5/CRC32C checksums verified.\n')
print('SERVER_CHECKPOINT_VERIFIED',model,flush=True)
