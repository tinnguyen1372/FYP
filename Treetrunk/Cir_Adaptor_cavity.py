from gprMax.gprMax import api
from gprMax.receivers import Rx


import h5py
import numpy as np
import matplotlib.pyplot as plt

import math

import sys
import pathlib

import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(module)s %(levelname)s]: %(message)s,'
        )
import cirultils as cu

#######################################################################
def generate_config(input,txtfile,cavity,geometry):
        config = '''
#title: Healthy TreeTrunk

Configuration
#domain: 0.8 0.8 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 20e-9

#pml_cells: 20 20 0 20 20 0

Material
#material: 5.22 0.005 1 0 Heartwood
#material: 5.9 0.02 1 0 Inner_Sapwood
#material: 6.1 0.033 1 0 Outer_Sapwood
#material: 6 1 1 0 Cabdium
#material: 5 0 1 0 Bark

Source - Receiver - Waveform
#waveform: ricker 1 1e9 my_wave
#hertzian_dipole: z 0.15 0.3 0 my_wave 
#rx: 0.15 0.5 0 

Centric Layer
#cylinder: 0.4 0.4 0 0.4 0.4 0.002 0.15 Bark
#cylinder: 0.4 0.4 0 0.4 0.4 0.002 0.14 Cabdium
#cylinder: 0.4 0.4 0 0.4 0.4 0.002 0.135 Outer_Sapwood
#cylinder: 0.4 0.4 0 0.4 0.4 0.002 0.11 Inner_Sapwood
#cylinder: 0.4 0.4 0 0.4 0.4 0.002 0.08 Heartwood

Cavity
#material: 1 0 1 0 cavity

#python:
cavity_coord = []
with open('{}', 'r') as file:
    for line in file:
        x, y, z = map(float,line.strip().split())
        cavity_coord.append((x, y, z))
file.close()
cav_x, cav_y, cav_z = cavity_coord[current_model_run-1]
print(f"#geometry_objects_read: {} {} {} {} ./cavity/cavity.txt")

#end_python:
Geometry objects write

#geometry_objects_write: 0 0 0 0.8 0.8 0.002 {}
        '''.format(txtfile, str('{cav_x}'), str('{cav_y}'), str('{cav_z}'), cavity,geometry)

        with open(input, 'w') as file:
            file.write(config)
        file.close()


#########################################################################
class Cir_Adaptor():
    
    def __init__(self,args,**kw) -> None:
        self.bscan_num = 36

        # self.cx , self.cy, self.cz, self.cradius = args.cx, args.cy, args.cz, args.cradius
        self.cx , self.cy, self.cz = args.center
        self.cradius = args.cradius
        self.cavity = args.cavity
        self.files = []
        # Mode 0: Manual    # Mode 1: Auto
        self.mode = args.auto
        self.cavity_sample = ['./cavity/geometry_generated.h5']

        if self.mode == 0:
            self.boundary_limit = 0.08
            self.save_geo = None
            self.run()

        else:
            self.run_boundary()
            # self.run_cavity(0.05,0.03)
            Ez = cu.remove_coupling(self.files,bscan_num=self.bscan_num)
            logging.debug(self.files)
            plt.imshow(Ez,cmap= 'seismic',aspect= 'auto',)
            plt.set_cmap('gray')  
            output_image = './output/B_Scan_comparison_radius.png'
            plt.savefig(output_image, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    def init_cavity(self):
        self.distance = self.cradius - (self.cavity + self.boundary_limit)
        
        self.input = './treetrunk_{:.2f}_{:.2f}.in'.format(round(self.distance,2),self.cavity)
        self.filename = './cavcoord/cavcoord_{:.2f}_{:.2f}.txt'.format(round(self.distance,2),self.cavity)

        x,y,z= cu.generate_point_on_circle_2d(self.distance,self.cx, self.cy, self.cz, angle_degrees= 0)
        radius1 = math.sqrt((x - self.cx)**2 + (y - self.cy)**2 + (z - self.cz)**2)
        cavity_coord = cu.generate_points_in_circular_view(x,y,z,self.cx,self.cy, self.cz, radius1, self.bscan_num, self.filename)
        
    def save_geometry(self,filename):
        f = h5py.File(filename + '.h5', 'r')
        dset = f['data']
        # Generate the image
        plt.imshow(dset, cmap='viridis')
        # Display the image
        plt.savefig('./geometry/{}.png'.format(filename.replace('./','')),dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def run_Ascan(self):
        from tools.plot_Ascan import mpl_plot
        api(self.input, n=1, geometry_only=True)
        # outputs = Rx.defaultoutputs
        # outputs = ['Ez']
        filename_g = self.input.replace('.in','')
        # plt = mpl_plot(filename_g + '.out', outputs, fft=True)
        # # Save the plot as an image file
        # output_file = './output/A_Scan_{}.png'.format(filename_g.replace('./',''))
        # plt.savefig(output_file, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        # # plt.show()
        # # Close the plot (if needed)
        # plt.close() 
        self.save_geometry(filename_g)
        logging.debug("Finish running A_scan for {}".format(self.input))

    def run_Bscan(self,continue_from = 1,deletefiles = False):
        from tools.outputfiles_merge import merge_files
        from tools.plot_Bscan import get_output_data, mpl_plot  

        filename_g = self.input.replace('.in','')
        output_image = './output/B_Scan_{}.png'.format(filename_g.replace('./',''))
        rxnumber = 1
        rxcomponent = 'Ez'

        api(self.input, restart= continue_from ,n=self.bscan_num - continue_from +1 , geometry_only=False)

        merge_files(str(filename_g.replace('./','')), deletefiles)
        output_file =str(filename_g.replace('./',''))+ '_merged.out'
        self.files.append(output_file)
        if self.save_geo is None:
            self.save_geometry(filename_g)
            self.save_geo = 1
        outputdata, dt = get_output_data(output_file, rxnumber, rxcomponent)
        plt_raw = mpl_plot(output_file, outputdata, dt, rxnumber, rxcomponent)  
        plt_raw.savefig(output_image, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        plt_raw.close()
        
        logging.debug("Finish running B_scan for {}".format(filename_g.replace('./','')))

    def run_boundary(self):
        self.boundary_limit = 0.05
        while True:
            if self.boundary_limit + self.cavity > self.cradius:
                break
            self.save_geo = None
            self.run()
            self.boundary_limit = self.boundary_limit + self.cradius/5

    def run_cavity(self,boundary,cavity):
        self.boundary_limit = boundary
        self.cavity = cavity
        while True:
            if self.cavity > float(self.cradius/2):
                break
            self.save_geo = None
            self.run()
            self.cavity = self.cavity + 0.02
        
    def run(self):
        self.init_cavity()
        logging.info("Cavity with radius {} m and distance {} from center treetrunk initialized".format(self.cavity, self.distance))
        generate_config(self.input,self.filename,self.cavity_sample[0],f'treetrunk_{round(self.distance,2):.2f}_{self.cavity:.2f}')
        logging.info("Input config initialized")
        if not self.mode:
            B_scan = input("Running Bscan (y/yc/n):")
            if 'y' in B_scan.lower():
                if 'c' in B_scan.lower():
                    continue_from = input("Continue from:")
                    self.run_Bscan(continue_from)
                else:
                    self.run_Bscan()
            else:
                A_scan = input("Running Ascan (y/n):")
                if A_scan.lower() == 'y':
                    self.run_Ascan()
                else:
                    pass
        else:
            logging.debug("Automated running bscan")
            self.run_Bscan()
            
        

    
if __name__ == "__main__":
    import argsutils as au
    # handle input arguments configuration
    import argparse
    parser = argparse.ArgumentParser(description="Circular Scanning for FYP")  
    au.add_arg(parser, '-d', '--debug', h='Turn on debugging output on console', a=True)

    g = parser.add_argument_group("adaptor configuration parameters")
    au.add_arg(g, "--center", h="Center of treetrunk", d=(0.4,0.4,0), m='COORD')
    au.add_arg(g, "--cradius",t=float, h="Radius of treetrunk", m='m', d=0.15)
    au.add_arg(g, "--cavity",t=float, h="Radius of cavity", m='m', d=0.03)
    au.add_arg(g, "--auto",t=float, h="Automate of running", m='int', d=0)
    args = au.parse_args(parser)

    # start  adaptor
    logging.debug("{}".format(args))
    cir = Cir_Adaptor(args=args)
    try:
        while not cir.is_quit(10.):
            pass
    except:
        pass