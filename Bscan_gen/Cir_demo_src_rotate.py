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


#################################################################
class Cir_Adaptor():
    def __init__(self,args,**kw) -> None:
        self.bscan_num = 36
        self.boundary_limit = 0.01
        self.start = args.start
        self.prefix = args.prefix
        self.count = args.count
        
        self.generated_healthy_base = os.getcwd() + '/Data/Healthy/healthy'
        self.generated_cavity_base = os.getcwd() + '/Data/Defect/defect'


        # Mode 0: Manual    # Mode 1: Auto
        self.data = np.load('SL_trunk.npz')
        
        # Init and pass data
        # self.data = generate_and_save_data(trunk_count= trunk_count ,filename=data_file)
        # self.run()
    
    def run(self):
        for iteration in range(self.start, self.start + self.count):
            self.run_healthy(iteration)
            self.run_cavity(iteration)
            source_file_healthy = self.prefix + "bscan_healthy.out"
            source_file_cavity = self.prefix + "bscan_cavity.out"
            destination_file_healthy = './Output/bscan/bscan_healthy_' + str(iteration) + '.out'
            destination_file_cavity ='./Output/bscan/bscan_cavity_' + str(iteration) + '.out'
            while os.path.isfile(destination_file_healthy):
                destination_file_healthy = cu.increment_file_index(destination_file_healthy)

            while os.path.isfile(destination_file_cavity):
                destination_file_cavity = cu.increment_file_index(destination_file_cavity)

            with open(source_file_healthy,"rb") as source, open(destination_file_healthy,"wb") as destination:
                destination.write(source.read())

            with open(source_file_cavity, "rb") as source, open(destination_file_cavity,"wb") as destination:
                destination.write(source.read())

            print('B_scan_healthy exported to file: {}'.format(destination_file_healthy))
            print('B_scan_cavity exported to file: {}'.format(destination_file_cavity))
            # Plot the cavity_only signal
            import matplotlib.pyplot as plt
            bscan_data = np.abs(self.merged_data_cavity - self.merged_data_healthy)
            fig_width = 15
            fig_height = 15

            # Create the figure and axes objects with the desired size and aspect ratio
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Generate the image plot with the logarithmic-scaled and normalized data
            im = ax.imshow(bscan_data, cmap='gray', aspect='auto')
            ax.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding

            save_path = "./Output/Cav_only/"
            save_filename = "cavity-only_" + str(iteration)
            plt.savefig(save_path + save_filename + ".png")

    def run_healthy(self,iteration):
        # logging.debug(self.data['r_trunk_L1'][iteration][0])
        # self.distance = self.data['r_Bcavity'][iteration][0]
        self.cradius = self.data['r_trunk_L1'][iteration][0]
        eps_trunk = self.data['eps_trunk'][iteration]
        # logging.info(self.cradius)
        # self.rcavity = self.data['R_center_Bcavity'][iteration] 
        # self.cx, self.cy, self.cz = (self.data['CenTrunk_X'],self.data['CenTrunk_Y'],0)
        self.input = './{}{}.in'.format(self.prefix, iteration)
        src_to_rx = 0.1
        diameter =self.cradius * 2
        resolution = 0.001
        res = diameter / resolution
        time_window = 3e-8
        pml_cells = 10
        pml = resolution * pml_cells *2
        x_gap = 0.02
        y_gap = 0.02
        src_to_pml = 0.01
        src_to_trunk = 0.03
        sharp_domain = diameter + src_to_trunk * 2, diameter + src_to_trunk * 2
        z = 0
        domain = [
            sharp_domain[0] + pml * 2 + x_gap ,
            sharp_domain[1] + pml * 2 + y_gap + src_to_pml, 
            0.001
        ]
        trunk_base = [
            pml + x_gap,
            src_to_trunk + pml + src_to_pml, z
        ]
        src_position = [self.cradius + pml + x_gap - src_to_rx/2, pml + src_to_pml, z]
        rx_position = [src_position[0] + src_to_rx, src_position[1], z]
        with open('{}materials.txt'.format(self.prefix), "w") as file:
            file.write('#material: {} 0 1 0 trunk\n'.format(eps_trunk))
            file.write('#material: 1 0 1 0 cavity')
        logging.debug('_ iteration {} _'.format(iteration))
        cu.preprocess(self.generated_healthy_base + str(iteration) + ".png", res,self.prefix, iteration, angle = 0)
        cu.generate_points_in_circular_view(src_position[0],src_position[1],0, 
                                    self.cradius + pml + x_gap,
                                    pml + src_to_pml + self.cradius + src_to_trunk,0,
                                    self.cradius+src_to_trunk,36, filename= 'src_coord.txt')
        cu.generate_points_in_circular_view(rx_position[0],rx_position[1],0, 
                            self.cradius + pml + x_gap,
                            pml + src_to_pml + self.cradius + src_to_trunk
                            ,0, self.cradius+src_to_trunk,36, filename= 'rx_coord.txt')
        config = f'''
#title: Healthy TreeTrunk

Configuration
#domain: {domain[0]:.3f} {domain[1]:.3f} {domain[2]:.3f}
#dx_dy_dz: 0.001 0.001 0.001
#time_window: {time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0

#python:
src_coord = []
rx_coord = []
with open('src_coord.txt', 'r') as file:
    for line in file:
        x, y, z = map(float,line.strip().split())
        src_coord.append((x, y, z))
file.close()
src_x, src_y, src_z = src_coord[current_model_run-1]

with open('rx_coord.txt', 'r') as file:
    for line in file:
        x, y, z = map(float,line.strip().split())
        rx_coord.append((x, y, z))
file.close()
rx_x, rx_y, rx_z = rx_coord[current_model_run-1]

print(f'#hertzian_dipole: z {str('{src_x}')} {str('{src_y}')} {str('{src_z}')} my_wave')
print(f'#rx: {str('{rx_x}')} {str('{rx_y}')} {str('{rx_z}')}')

#end_python:

Source - Receiver - Waveform
#waveform: ricker 1 1e9 my_wave

Geometry objects read

#geometry_objects_read: {trunk_base[0]:.3f} {trunk_base[1]:.3f} {trunk_base[2]:.3f} {self.prefix}healthy.h5 {self.prefix}materials.txt

    '''
        with open(self.input, 'w') as file:
            file.write(config)
            file.close()
        filename_g = self.input.replace('./','')
        #filename = FYP_1.in
        api(filename_g, 
            # gpu= [0],
            n= 36,
            geometry_only=False,
            geometry_fixed=False)
            # os.rename(f"{filename_g.replace('.in','')}" + ".out", f"{filename_g.replace('.in','')}" + f"{i+1}.out")

        merge_files(str(filename_g.replace('.in','')), False)
        output_file =str(filename_g.replace('.in',''))+ '_merged.out'

        output_filename = self.prefix + 'bscan_healthy.out'
        dt = 0

        with h5py.File(output_file, 'r') as f1:
            data1 = f1['rxs']['rx1']['Ez'][()]
            dt = f1.attrs['dt']

        with h5py.File('cir_src_rotate_only.out', 'r') as f1:
            data_source = f1['rxs']['rx1']['Ez'][()]

        # src = data_source
        # src = src[:, np.newaxis]
        # Ez0 = np.repeat(src, self.bscan_num, axis=1)

        # self.merged_data_healthy = np.subtract(data1, Ez0)
        self.merged_data_healthy = np.subtract(data1, data_source)

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
        plt.xlabel('Trace number')
        plt.ylabel('Time [s]')
        plt.imshow(self.merged_data_healthy, cmap='gray', aspect='auto')
        # plt.axis('off')
        ax.margins(0, 0)  # Remove any extra margins or padding
        fig.tight_layout(pad=0)  # Remove any extra padding

        save_path = "./Output/" + "Healthy/"
        save_filename = file_names
        plt.savefig(save_path + save_filename + ".png")
        logging.debug("Finish running B_scan for Healthy")


    def run_cavity(self,iteration):   
        # logging.debug(self.data['r_trunk_L1'][iteration][0])
        # self.distance = self.data['r_Bcavity'][iteration][0]
        self.cradius = self.data['r_trunk_L1'][iteration][0]
        eps_trunk = self.data['eps_trunk'][iteration]
        # logging.info(self.cradius)
        # self.rcavity = self.data['R_center_Bcavity'][iteration] 
        # self.cx, self.cy, self.cz = (self.data['CenTrunk_X'],self.data['CenTrunk_Y'],0)
        self.input = './{}{}.in'.format(self.prefix, iteration)
        src_to_rx = 0.1
        diameter =self.cradius * 2
        resolution = 0.001 #0.002
        res = diameter / resolution
        time_window = 3e-8
        pml_cells = 10 #20
        pml = resolution * pml_cells *2
        x_gap = 0.02
        y_gap = 0.02
        src_to_pml = 0.01
        src_to_trunk = 0.03
        sharp_domain = diameter + src_to_trunk * 2, diameter + src_to_trunk * 2
        z = 0
        domain = [
            sharp_domain[0] + pml * 2 + x_gap ,
            sharp_domain[1] + pml * 2 + y_gap + src_to_pml, 
            0.001 #resolution
        ]
        trunk_base = [
            pml + x_gap,
            src_to_trunk + pml + src_to_pml, z
        ]
        src_position = [self.cradius + pml + x_gap - src_to_rx/2, pml + src_to_pml, z]
        rx_position = [src_position[0] + src_to_rx, src_position[1], z]
        with open('{}materials.txt'.format(self.prefix), "w") as file:
            file.write('#material: {} 0 1 0 trunk\n'.format(eps_trunk))
            file.write('#material: 1 0 1 0 cavity')
        logging.debug('_ iteration {} _'.format(iteration))
        with open('{}materials.txt'.format(self.prefix), "w") as file:
            file.write('#material: {} 0 1 0 trunk\n'.format(eps_trunk))
            file.write('#material: 1 0 1 0 cavity')
        logging.debug('_ iteration {} _'.format(iteration))
        cu.preprocess(self.generated_cavity_base + str(iteration) + ".png", res,self.prefix, iteration, angle = 0)
        cu.generate_points_in_circular_view(src_position[0],src_position[1],0, 
                                    self.cradius + pml + x_gap,
                                    pml + src_to_pml + self.cradius+src_to_trunk,0, 
                                    self.cradius+src_to_trunk,36, filename= 'src_coord.txt')
        cu.generate_points_in_circular_view(rx_position[0],rx_position[1],0, 
                            self.cradius + pml + x_gap,
                            pml + src_to_pml + self.cradius +src_to_trunk,0, 
                            self.cradius+src_to_trunk,36, filename= 'rx_coord.txt')
        # Convert the generated image to h5
        # cu.preprocess(self.generated_cavity_base + str(iteration) + ".png", res)
        # Write to materials.txt
        config = f'''
#title: Defect TreeTrunk

Configuration
#domain: {domain[0]:.3f} {domain[1]:.3f} {domain[2]:.3f}
#dx_dy_dz: 0.001 0.001 0.001
#time_window: {time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0


Source - Receiver - Waveform
#waveform: ricker 1 1e9 my_wave
#python:
src_coord = []
rx_coord = []
with open('src_coord.txt', 'r') as file:
    for line in file:
        x, y, z = map(float,line.strip().split())
        src_coord.append((x, y, z))
file.close()
src_x, src_y, src_z = src_coord[current_model_run-1]

with open('rx_coord.txt', 'r') as file:
    for line in file:
        x, y, z = map(float,line.strip().split())
        rx_coord.append((x, y, z))
file.close()
rx_x, rx_y, rx_z = rx_coord[current_model_run-1]

print(f'#hertzian_dipole: z {str('{src_x}')} {str('{src_y}')} {str('{src_z}')} my_wave')
print(f'#rx: {str('{rx_x}')} {str('{rx_y}')} {str('{rx_z}')}')

#end_python:
Geometry objects read

#geometry_objects_read: {trunk_base[0]:.3f} {trunk_base[1]:.3f} {trunk_base[2]:.3f} {self.prefix}cavity.h5 {self.prefix}materials.txt

    '''
        with open(self.input, 'w') as file:
            file.write(config)
            file.close()
        filename_g = self.input.replace('./','')
        api(filename_g, 
            # gpu= [0],
            n= 36,
            geometry_only=False,
            geometry_fixed=False)
        # os.rename(f"{filename_g.replace('.in','')}" + ".out", f"{filename_g.replace('.in','')}" + f"{i+1}.out")
        merge_files(str(filename_g.replace('.in','')), False)
        output_file =str(filename_g.replace('./','').replace('.in',''))+ '_merged.out'

        output_filename = self.prefix + 'bscan_cavity.out'
        dt = 0

        with h5py.File(output_file, 'r') as f1:
            data1 = f1['rxs']['rx1']['Ez'][()]
            dt = f1.attrs['dt']

        with h5py.File('cir_src_rotate_only.out', 'r') as f1:
            data_source = f1['rxs']['rx1']['Ez'][()]

        # src = data_source
        # src = src[:, np.newaxis]
        # Ez0 = np.repeat(src, self.bscan_num, axis=1)
        # self.merged_data_cavity = np.subtract(data1, Ez0)

        self.merged_data_cavity = np.subtract(data1, data_source)
        # Create a new output file and write the merged data
        with h5py.File(output_filename, 'w') as f_out:
            f_out.attrs['dt'] = dt  # Set the time step attribute
            f_out.create_dataset('rxs/rx1/Ez', data=self.merged_data_cavity)

        # Draw data with normal plot
        rxnumber = 1
        rxcomponent = 'Ez'
        plt = mpl_plot("merged_output_data", self.merged_data_cavity, dt, rxnumber,rxcomponent)

        # Draw data with gray plot
        file_names = "cavity-" + str(iteration)
        fig_width = 15
        fig_height = 15

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plt.xlabel('Trace number')
        plt.ylabel('Time [s]')
        plt.imshow(self.merged_data_cavity, cmap='gray', aspect='auto')
        # plt.axis('off')
        ax.margins(0, 0)  # Remove any extra margins or padding
        fig.tight_layout(pad=0)  # Remove any extra padding

        save_path = "./Output/" + "Defect/"
        save_filename = file_names
        plt.savefig(save_path + save_filename + ".png")
        logging.debug("Finish running B_scan for Cavity")

        # Export to data.csv
        csv_file = './Output/csv/data.csv'
        # Prepare the data to be written to the CSV file
        data = [
            iteration,
            self.cradius,
            # self.rcavity,
            # self.theta_cavity,
            # self.distance , 
            src_to_trunk, src_to_rx,
            time_window, pml_cells, x_gap, y_gap, src_to_pml, res, 
            domain[0],domain[1], domain[2], 
            trunk_base[0], trunk_base[1], trunk_base[2]
        ]

        # Write the data to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

        logging.info("Data appended to the CSV file.")

if __name__ == "__main__":
    import argsutils as au
    # handle input arguments configuration
    import argparse
    parser = argparse.ArgumentParser(description="Circular Scanning for FYP")  
    au.add_arg(parser, '-d', '--debug', h='Turn on debugging output on console', a=True)
    
    g = parser.add_argument_group("adaptor configuration parameters")
    au.add_arg(g, "--start",t=int, h="Starting index value", m='int', d=1)
    au.add_arg(g, "--count",t=int, h="Iteration count value", m='int', d=10)
    au.add_arg(g, "--trunk",t=int, h="Trunks count value", m='int', d= None)
    au.add_arg(g, "--prefix",t=str, h="Prefix", m='string', d= "FYP_")
    args = au.parse_args(parser)

    # start  adaptor
    logging.debug("{}".format(args))
    cir = Cir_Adaptor(args=args)
    cir.run()
    try:
        while not cir.is_quit(10):
            pass
    except:
        pass