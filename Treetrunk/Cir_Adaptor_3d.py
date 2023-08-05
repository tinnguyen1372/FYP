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
def generate_config(input,txtfile,cavity_radius,geometry,cavity_height):
        config = '''
#title: Healthy Tree

Configuration
#domain: 0.6 0.6 0.6
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 2e-8
pml_cells: 20 20 20 20 20 20

Material
#material: 5.22 0.005 1 0 Heartwood
#material: 5.9 0.02 1 0 Inner_Sapwood
#material: 6.1 0.033 1 0 Outer_Sapwood
#material: 6 1 1 0 Cabdium
#material: 5 0 1 0 Bark


#cylinder: 0.3 0 0.3 0.3 0.6 0.3 0.15 Bark
cylinder: 0.3 0 0.3 0.3 0.6 0.3 0.14 Cabdium
#cylinder: 0.3 0 0.3 0.3 0.6 0.3 0.135 Outer_Sapwood
cylinder: 0.3 0 0.3 0.3 0.6 0.3 0.11 Inner_Sapwood
#cylinder: 0.3 0 0.3 0.3 0.6 0.3 0.08 Heartwood

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
print(f"#cylinder: {} {} {} {} {} {} {} cavity")

from user_libs.antennas.GSSI import antenna_like_GSSI_1500
antenna_like_GSSI_1500(0.3, 0.15 ,0.5, 0.002)


#end_python:

#geometry_objects_write: 0 0 0 0.6 0.6 0.6 {}
        '''.format(txtfile, str('{cav_x}'), str('{cav_y}'), str('{cav_z}'), str('{cav_x}'), cavity_height ,str('{cav_z}'),cavity_radius,geometry)


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
        self.cavity_height = args.cavity_height
        self.files = []
        # Mode 0: Manual    # Mode 1: Auto
        self.mode = args.auto

        if self.mode == 0:
            self.boundary_limit = 0.05
            self.save_geo = None
            self.run()

        else:
            self.run_boundary()
            # self.run_cavity(0.10,0.05)
            Ey = cu.remove_coupling(self.files,bscan_num=self.bscan_num)
            logging.debug(self.files)
            plt.imshow(Ey,cmap= 'seismic',aspect= 'auto',)
            plt.set_cmap('gray')  
            output_image = './output/B_Scan_comparison_radius.png'
            plt.savefig(output_image, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    def init_cavity(self):
        self.distance = self.cradius - (self.cavity + self.boundary_limit)
        
        self.input = './treetrunk_{}_{}.in'.format(round(self.distance,2),self.cavity)
        self.filename = './cavcoord/cavcoord_{}_{}.txt'.format(round(self.distance,2),self.cavity)

        x,y,z= cu.generate_point_on_circle_3d(self.distance,self.cx, self.cy, self.cz, angle_degrees= 0)
        radius1 = math.sqrt((x - self.cx)**2 + (y - self.cy)**2 + (z - self.cz)**2)
        cavity_coord = cu.generate_points_in_circular_view_3d(x,y,z,self.cx,self.cy, self.cz, radius1, self.bscan_num, self.filename)
        
    def save_geometry(self,filename):
        f = h5py.File(filename + '.h5', 'r')
        dset = f['data'][()]
        # Generate the image
        fig, ax = plt.subplots(1, 3,figsize=(10, 5))
        ax[0].imshow(np.transpose(dset[:,73,:], axes=(1, 0)), cmap='viridis')
        ax[0].invert_yaxis()
        ax[0].set_title('Domain View')

        # # Plot dset2 on the right subplot
        ax[1].imshow(np.transpose(dset[:,:,150], axes=(1, 0)), cmap='viridis')
        ax[1].invert_yaxis()
        ax[1].set_title('Trunk View')

        ax[2].imshow(np.transpose(dset[:,:,250], axes=(1, 0)), cmap='viridis')
        ax[2].invert_yaxis()
        ax[2].set_title('Antenna View')

        # Display the image
        fig.savefig('./geometry/{}.png'.format(filename.replace('./','')),dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()



    def run_Ascan(self):
        from tools.plot_Ascan import mpl_plot
        api(self.input, n=1, geometry_only=True)
        outputs = Rx.defaultoutputs
        outputs = ['Ey']
        filename_g = self.input.replace('.in','')
        # plt = mpl_plot(filename_g + '.out', outputs, fft=True)
        # # Save the plot as an image file
        # output_file = './output/A_Scan_{}.png'.format(filename_g.replace('./',''))
        # plt.savefig(output_file, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        # plt.show()
        # # Close the plot (if needed)
        # plt.close() 
        self.save_geometry(filename_g)
        logging.debug("Finish running A_scan for {}".format(self.input))

    def run_Bscan(self,continue_from = 1,deletefiles = True):
        from tools.outputfiles_merge import merge_files
        from tools.plot_Bscan import get_output_data, mpl_plot  

        filename_g = self.input.replace('.in','')
        output_image = './output/B_Scan_{}.png'.format(filename_g.replace('./',''))
        rxnumber = 1
        rxcomponent = 'Ey'

        api(self.input, gpu= [0],restart= continue_from ,n= int(self.bscan_num - int(continue_from) +1), geometry_only=False)

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
        self.boundary_limit = 0.02
        while True:
            if self.boundary_limit + self.cavity > self.cradius:
                break
            self.save_geo = None
            self.run()
            self.boundary_limit = self.boundary_limit + self.cradius/3

    def run_cavity(self,boundary,cavity):
        self.boundary_limit = boundary
        self.cavity = cavity
        while True:
            if self.boundary_limit + self.cavity > self.cradius:
                break
            self.save_geo = None
            self.run()
            self.cavity = self.cavity + self.cradius/5
        
    def run(self):
        self.init_cavity()
        logging.info("Cavity with radius {} m and distance {} from center treetrunk initialized".format(self.cavity, self.distance))
        generate_config(self.input,self.filename,self.cavity,f'treetrunk_{round(self.distance,2)}_{self.cavity}',self.cavity_height)
        logging.info("Input config initialized")
        if not self.mode:
            B_scan = input("Running Bscan (y/yc/n):")
            if 'y' in B_scan.lower():
                if 'c' in B_scan.lower():
                    continue_from = input("Continue from:")
                    self.run_Bscan(int(continue_from))
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
    au.add_arg(g, "--center", h="Center of treetrunk", d=(0.3,0.3,0.3), m='COORD')
    au.add_arg(g, "--cradius",t=float, h="Radius of treetrunk", m='m', d=0.15)
    au.add_arg(g, "--cavity",t=float, h="Radius of cavity", m='m', d=0.05)
    au.add_arg(g, "--cavity_height",t=float, h="Radius of cavity", m='m', d=0.08)
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