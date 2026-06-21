import os, sys, zipfile, tarfile, shutil
root = os.path.dirname(os.path.dirname(__file__))
data = os.path.join(root,'data')
print('Scanning', data)
for fn in os.listdir(data):
    path = os.path.join(data, fn)
    if fn.lower().endswith('.zip'):
        print('Trying zipfile for', fn)
        try:
            with zipfile.ZipFile(path,'r') as zf:
                zf.extractall(os.path.join(data, fn + '_zip'))
                print('Extracted with zipfile')
                continue
        except Exception as e:
            print('zipfile failed:', e)
        try:
            import py7zr
            print('Trying py7zr for', fn)
            with py7zr.SevenZipFile(path, mode='r') as z:
                z.extractall(path=os.path.join(data, fn + '_7z'))
                print('Extracted with py7zr')
                continue
        except Exception as e:
            print('py7zr failed or not available:', e)
        # last resort: try system 7z
        try:
            import subprocess
            outdir = os.path.join(data, fn + '_7zcli')
            os.makedirs(outdir, exist_ok=True)
            subprocess.check_call(['7z','x',path,'-o'+outdir])
            print('Extracted with 7z CLI')
            continue
        except Exception as e:
            print('7z CLI failed:', e)
    elif fn.lower().endswith(('.tar.gz','.tgz')):
        try:
            with tarfile.open(os.path.join(data,fn),'r:gz') as tf:
                tf.extractall(os.path.join(data, fn + '_tar'))
                print('Extracted tar.gz', fn)
        except Exception as e:
            print('tar extraction failed:', e)
print('Done')
