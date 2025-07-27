
import os
import numpy as np
import shutil


DS_path= r'C:\Users\fsalm\Desktop\DynEO4SLUMS\nairobi\planetscope\nairobi_psb_sd_20230129_surfrefl_8bands_psscene_analytic_8b_sr_udm2\ds'
package = 'valid'

flst = os.listdir(os.path.join(DS_path, package))

counter = 0
for file in flst:
    img_np = np.load(os.path.join(DS_path, package, file))
    if img_np.max() == 0:
        counter += 1
        shutil.move(os.path.join(DS_path, package, file),os.path.join(DS_path, 'vd_zero', file))

print(len(flst), counter)