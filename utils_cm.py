import os 
import numpy as np 
import shutil
import glob 

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copyfile(src, dst):
    path = os.path.dirname(dst)
    mkdir_p(path)
    shutil.copyfile(src, dst)

def chdir_p(path='/content/drive/My Drive/Workspace/OT/myOT/'): 
    os.chdir(path)
    WP = os.path.dirname(os.path.realpath('__file__')) +'/'
    print('CHANGING WORKING PATH: ', WP)

def writelog(data=None, logfile=None, printlog=True):
    fid = open(logfile,'a')
    fid.write('%s\n'%(data))
    fid.flush()
    fid.close()
    if printlog: 
        print(data)

def dict2str(d): 
    # assert(type(d)==dict)
    res = ''
    for k in d.keys(): 
        v = d[k]
        res = res + '{}:{},'.format(k,v)
    return res 

def list2str(l): 
    # assert(type(l)==list)
    res = ''
    for i in l: 
        res = res + ' {}'.format(i)
    return res 

def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def delete_existing(path, overwrite=True):
    """Delete directory if it exists

    Used for automatically rewrites existing log directories
    """
    if not overwrite:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)
    else:
        if os.path.exists(path):
            shutil.rmtree(path)

def backup(source_dir, dest_dir):
    files = glob.iglob(os.path.join(source_dir, "*.py"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)

# List all dir with specific name 
def list_dir(folder_dir, filetype='.png'):
    if '.' in filetype:
        all_dir = sorted(glob.glob(folder_dir+"*"+filetype), key=os.path.getmtime)
    else:
        all_dir = sorted(glob.glob(folder_dir+"*."+filetype), key=os.path.getmtime)
    return all_dir

def split_dict(d): 
    assert(type(d) is dict)
    all_dicts = []
    nb_d = 1 
    for k in d.keys(): 
        if type(d[k]) is list or type(d[k]) is tuple: 
            nb_d = np.maximum(nb_d, len(d[k]))

    def _get(d, k, i): 
        v = d[k]
        if type(v) is list or type(v) is tuple: 
            if i <= len(v):
                return v[i]
            else: 
                return v[-1]
        else: 
            return v

    for i in range(nb_d):
        new_dict = dict() 
        for k in d.keys(): 
            new_dict[k] = _get(d, k, i)
        all_dicts.append(new_dict)
    return all_dicts