import h5py
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
prefix_1 = 'imgrot_test_'
prefix_2 = 'srcrot_test_'

# Select the Ascan index for the comparison
ascan_index = 1


for ascan_index in range(1,37):
    with h5py.File('{}0{}.out'.format(prefix_1, ascan_index), 'r') as f1:
        data1 = f1['rxs']['rx1']['Ez'][()]

        dt = f1.attrs['dt']
        iterations = f1.attrs['Iterations']
        time = np.linspace(0, (iterations - 1) * dt, num=iterations)


    with h5py.File('{}0{}.out'.format(prefix_2, ascan_index), 'r') as f2:
        data2 = f2['rxs']['rx1']['Ez'][()]

    # Calculate the Structural Similarity Index (SSI)
    ssi_value = ssim(data1, data2, data_range=data1.max() - data1.min())

    # Convert the SSI value to a percentage similarity
    percentage_similarity = (ssi_value + 1) * 50

    # Print the similarity value
    print("Ascan {} similarity = {:.2f}%".format(ascan_index, percentage_similarity))

    from gprMax.receivers import Rx
    outputs_1 = Rx.defaultoutputs
    outputs_1 = ['Ez']
    from tools.plot_Ascan import mpl_plot
    plt_1 = mpl_plot('{}0{}.out'.format(prefix_2, ascan_index), outputs_1, fft=False)
    plt_1.savefig('Receiver_{}_{}.png'.format(prefix_2, ascan_index))
    plt_1.close()  # Close the plot to free up memory

    outputs_2 = Rx.defaultoutputs
    outputs_2 = ['Ez']
    plt_2 = mpl_plot('{}0{}.out'.format(prefix_1, ascan_index), outputs_2, fft=False)
    plt_2.savefig('Receiver_{}_{}.png'.format(prefix_1, ascan_index))
    plt_2.close()

    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [s]', ylabel='Substracted Ez' + ' field strength [V/m]'), num='rx', figsize=(20, 10), facecolor='w', edgecolor='w')
    ax.plot(time, data1, label='Src rotation')
    ax.plot(time, data2, label='Img rotation')
    ax.set_xlim([0, np.amax(time)])
    # ax.set_ylim([-15, 20])
    ax.grid(which='both', axis='both', linestyle='-.')
    # Add labeled descriptions
    plt.text(0.5, 0.9, 'Data 1: {}'.format(prefix_1), transform=plt.gca().transAxes, fontsize=12, color='blue')
    plt.text(0.5, 0.85, 'Data 2: {}'.format(prefix_2), transform=plt.gca().transAxes, fontsize=12, color='orange')

    fig.savefig('comparison_Ascan_{}.png'.format(ascan_index))
    
    plt.close()
    # fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [s]', ylabel=outputtext + ' field strength [V/m]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
  # Subtract the data
    subtracted_data = data1 - data2

    # Create the plot for subtracted data
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [s]', ylabel='Substracted Ez' + ' field strength [V/m]'), num='rx', figsize=(20, 10), facecolor='w', edgecolor='w')
    line = ax.plot(time, data1 - data2, 'r', lw=2, label='Substracted')
    ax.set_xlim([0, np.amax(time)])
    # ax.set_ylim([-15, 20])
    ax.grid(which='both', axis='both', linestyle='-.')
    fig.savefig('Subtracted_Ascan_{}.png'.format(ascan_index))
    plt.close()  # Close the plot to free up memory

    # #Calculate the data ranges
    # range_subtracted_data = np.ptp(subtracted_data)
    # range_data1 = np.ptp(data1)
    # range_data2 = np.ptp(data2)

    # # Calculate the percentage difference between data ranges
    # percentage_difference_range1 = np.abs(range_subtracted_data) / range_data1 * 100
    # percentage_difference_range2 = np.abs(range_subtracted_data) / range_data2 * 100

    # # Compare the percentage differences with a threshold
    # threshold = 5  # You can adjust this threshold as needed
    # if percentage_difference_range1 < threshold and percentage_difference_range2 < threshold:
    #     comparison_result = "Good"
    # else:
    #     comparison_result = "Poor"

    # print("Ascan {}: Percentage Difference Range from data1: {:.2f}%".format(ascan_index, percentage_difference_range1))
    # print("Ascan {}: Percentage Difference Range from data2: {:.2f}%".format(ascan_index, percentage_difference_range2))
    # print("Ascan {}: Comparison Result: {}".format(ascan_index, comparison_result))

    # Calculate the Mean Squared Error (MSE)
    mse_value = np.mean((data1 - data2)**2)

    # Calculate the value range of the data
    data_range = np.max(data1) - np.min(data1)
    data_range_2 = np.max(data2) - np.min(data2)

    print("Ascan {} MSE {:.2f}".format(ascan_index, mse_value))

with h5py.File('imgrot_test_bscan_cavity.out', 'r') as f1:
    data1 = f1['rxs']['rx1']['Ez'][()]
    dt = f1.attrs['dt']

with h5py.File('srcrot_test_bscan_cavity.out', 'r') as f2:
    data2 = f2['rxs']['rx1']['Ez'][()]

# Calculate the Structural Similarity Index (SSI)
ssi_value = ssim(data1, data2, data_range=data1.max() - data1.min())

# Convert the SSI value to a percentage similarity
percentage_similarity = (ssi_value + 1) * 50

# Print the similarity value
print("Bscan similarity = {:.2f}%".format(percentage_similarity))

# Calculate the range

print("Bscan MSE: {}".format(np.square(np.subtract(data1,data2)).mean()))

subtracted_data = np.subtract(data1, data2)
# Create the plot
plt.figure(figsize=(10, 6))
plt.imshow(subtracted_data, extent=[0, data1.shape[1], 0, data1.shape[0]], aspect='auto')
plt.colorbar(label='Subtracted Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Subtracted Data (data1 - data2)')
plt.savefig('Bscan.png')
