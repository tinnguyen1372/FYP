from gprMax.gprMax import api
from gprMax.receivers import Rx
from tools.outputfiles_merge import merge_files
from tools.plot_Bscan import get_output_data, mpl_plot  


import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import logging
import os
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(module)s %(levelname)s]: %(message)s,'
        )

import cirultils as cu
from Generate import generate_and_save_data

#######################################################################


#########################################################################
class Cir_Adaptor():
    def __init__(self,args,**kw) -> None:
        self.bscan_num = 1
        self.boundary_limit = 0.01
        self.start = args.start
        self.prefix = args.prefix
        self.count = args.count
        self.generated_healthy_base = os.getcwd() + '/Data/Healthy/healthy'
        self.generated_cavity_base = os.getcwd() + '/Data/Defect/defect'
        self.files = []
        # Mode 0: Manual    # Mode 1: Auto
        self.mode = args.auto
        self.data = np.load('SL_trunk.npz')
        
        # Init and pass data
        # self.data = generate_and_save_data(trunk_count= trunk_count ,filename=data_file)
        self.run()

    

    def run_healthy(self,iteration):
            logging.debug(self.data['r_trunk_L1'][iteration][0])
            self.distance = self.data['r_Bcavity'][iteration][0]
            self.cradius = self.data['r_trunk_L1'][iteration][0]
            self.rcavity = self.data['R_center_Bcavity'][iteration] 
            self.cx, self.cy, self.cz = (self.data['CenTrunk_X'],self.data['CenTrunk_Y'],0)
            self.input = './treetrunk_{}_{}.in'.format(np.round(self.distance,2),self.rcavity)
            src_to_rx = 0.1
            diameter =self.cradius * 2

            resolution = 0.002
            res = diameter / 0.002
            time_window = 3e-8

            pml_cells = 20
            pml = resolution * pml_cells

            x_gap = 0.02
            y_gap = 0.02

            src_to_pml = 0.02
            src_to_trunk = 0.2

            sharp_domain = src_to_rx, diameter + src_to_trunk

            z = 0

            domain = [
                sharp_domain[0] + pml * 2 + x_gap * 2,
                sharp_domain[1] + pml * 2 + y_gap + src_to_pml, resolution
            ]
            trunk_base = [
                0.25 + pml + x_gap - self.cradius , src_to_trunk + pml + src_to_pml, z
            ]
            src_position = [pml + x_gap, pml + src_to_pml, z]
            rx_position = [src_position[0] + src_to_rx, src_position[1], z]
            


            print('_ iteration {} _'.format(iteration))

            # Convert the generated image to h5
            cu.preprocess(self.generated_healthy_base + str(iteration) + ".png", res,prefix= self.prefix, i = iteration)
            # cu.preprocess(self.generated_cavity_base + str(iteration) + ".png", res)
            # Write to materials.txt
            config = f'''
#title: Healthy TreeTrunk

Configuration
#domain: {domain[0]:.3f} {domain[1]:.3f} {domain[2]:.3f}
#dx_dy_dz: 0.002 0.002 0.002
#time_window: {time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0


Source - Receiver - Waveform
#waveform: ricker 1 1e9 my_wave
#hertzian_dipole: z {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}

Geometry objects read

#geometry_objects_read: {trunk_base[0]:.3f} {trunk_base[1]:.3f} {trunk_base[2]:.3f} {self.prefix}healthy.h5 {self.prefix}materials.txt
        '''
            with open(self.input, 'w') as file:
                file.write(config)
            file.close()
            filename_g = self.input
            api(filename_g, 
                # gpu= [0],
                n= self.bscan_num,
                geometry_only=False,
                geometry_fixed=False)
            merge_files(str(filename_g.replace('./','').replace('.in','')), True)
            output_file =str(filename_g.replace('./','').replace('.in',''))+ '_merged.out'

            output_filename = self.prefix + 'bscan_healthy.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ez'][()]
                dt = f1.attrs['dt']

            with h5py.File('cir_src_only.out', 'r') as f1:
                data_source = f1['rxs']['rx1']['Ez'][()]

            src = data_source
            src = src[:, np.newaxis]
            Ez0 = np.repeat(src, self.bscan_num, axis=1)

            self.merged_data_healthy = np.subtract(data1, Ez0)

            # Create a new output file and write the merged data
            with h5py.File(output_filename, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ez', data=self.merged_data_healthy)

            # Draw data with normal plot
            rxnumber = 1
            rxcomponent = 'Ez'
            plt = mpl_plot("merged_output_data", self.merged_data_healthy, dt, rxnumber,rxcomponent)

            # Draw data with gray plot
            file_names = "healthy-" + str(iteration)

            # fig_width = merged_data_healthy.shape[1] / 100
            # # Adjust the size based on the B-scan height
            # fig_height = merged_data_healthy.shape[0] / 100
            fig_width = 15
            fig_height = 15

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            plt.imshow(self.merged_data_healthy, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding

            save_path = "./Output/" + "Healthy/"
            save_filename = file_names
            plt.savefig(save_path + save_filename + ".png")
            logging.debug("Finish running B_scan for Healthy")

    def init_cavity(self):
        self.input = './treetrunk_{}_{}.in'.format(np.round(self.distance,2),self.rcavity)
        self.filename = './cavcoord/cavcoord_{}_{:.3f}.txt'.format(np.round(self.distance,2),self.rcavity)

        x,y,z= cu.generate_point_on_circle_2d(self.distance,self.cx, self.cy, self.cz, angle_degrees= self.theta_cavity)
        radius1 = math.sqrt((x - self.cx)**2 + (y - self.cy)**2 + (z - self.cz)**2)
        cavity_coord = cu.generate_points_in_circular_view(x,y,z,self.cx,self.cy, self.cz, radius1, self.bscan_num, self.filename)
        

    


            
        

    
if __name__ == "__main__":
    import argsutils as au
    # handle input arguments configuration
    import argparse
    parser = argparse.ArgumentParser(description="Circular Scanning for FYP")  
    au.add_arg(parser, '-d', '--debug', h='Turn on debugging output on console', a=True)
    
    g = parser.add_argument_group("adaptor configuration parameters")
    au.add_arg(g, "--auto",t=float, h="Automate of running", m='int', d=0)
    au.add_arg(g, "--start",t=int, h="Starting index value", m='int', d=1)
    au.add_arg(g, "--count",t=int, h="Iteration count value", m='int', d=10)
    au.add_arg(g, "--trunk",t=int, h="Trunks count value", m='int', d= None)
    au.add_arg(g, "--prefix",t=str, h="Prefix", m='string', d= "FYP_")
    args = au.parse_args(parser)

    # start  adaptor
    logging.debug("{}".format(args))
    cir = Cir_Adaptor(args=args)
    try:
        while not cir.is_quit(10):
            pass
    except:
        pass