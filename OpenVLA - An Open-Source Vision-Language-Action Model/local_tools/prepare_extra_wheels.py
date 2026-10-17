import json,urllib.request,concurrent.futures
from pathlib import Path
root=Path('/root/gpufree-data/openvla-repro')
names=['robosuite','opencv-python','llvmlite','scipy','imageio-ffmpeg','libclang']
def one(name):
 p=next((root/'cache/uv/wheels-v7/index/3ba65c4f41aac3a1'/name).glob('*.msgpack'))
 version=p.stem.split('-')[0]
 filename=name.replace('-','_')+'-'+p.stem+'.whl'
 with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as f:d=json.load(f)
 x=next(x for x in d['urls'] if x['filename']==filename)
 return {'filename':filename,'url':x['url'].replace('https://files.pythonhosted.org/packages/','https://mirrors.aliyun.com/pypi/packages/'),'size':x['size'],'sha256':x['digests']['sha256']}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p: records=list(p.map(one,names))
(root/'extra_wheel_manifest.json').write_text(json.dumps(records,indent=2))
s=(root/'scripts/download_wheels.py').read_text().replace('wheel_manifest.json','extra_wheel_manifest.json').replace('wheels_complete.txt','extra_wheels_complete.txt')
(root/'scripts/download_extra_wheels.py').write_text(s)
print([(r['filename'],r['size']) for r in records],flush=True)
exec(compile(s,str(root/'scripts/download_extra_wheels.py'),'exec'))
