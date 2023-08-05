
import sys
import pathlib


scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent))
import cirultils as cu
import matplotlib.pyplot as plt
import logging



file_names = ['healthy_3d.out', 
             'treetrunk_0.05_0.05_merged.out','treetrunk_0.07_0.03_3d.out']
Ez = cu.remove_coupling_3d(file_names, src_filename="cir_src_only_3d.out" , bscan_num=36)

# file_names =['treetrunk_0.07_0.03_merged.out', 'treetrunk_0.04_0.03_merged.out', 'treetrunk_0.01_0.03_merged.out']
# Ez = cu.remove_coupling(file_names, src_filename="cir_src_only.out" , bscan_num=36)

logging.debug(file_names)
plt.imshow(Ez,cmap= 'seismic',aspect= 'auto',)
plt.set_cmap('gray')  
output_image = './output/B_Scan_com_3d.png'
plt.savefig(output_image, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)