import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
os.chdir("./")

if __name__ == "__main__":
    # Read the names of the files in the current directory as the neuron name
    neuron_name= os.path.basename(os.getcwd())
    # Read in the data
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv(f"{(neuron_name)}_BP_csv.csv", index=False, encoding='utf-8-sig')
    # Read the data from the CSV file
    df = pd.read_csv(f"{(neuron_name)}_BP_csv.csv")

    # Group the data by SGD
    groups = df.groupby('SNN arch')

    # Set the size of the chart
    plt.figure(figsize=(10, 8))

    # Set a color map
    colors = plt.get_cmap('Set1')

    # Create a line chart for each group
    for i, (name, group) in enumerate(groups):
        # Use the color map to get a unique color for each line
        color = colors(i)
        plt.plot(group['Epoch'], group['Test Acc'], label=f'Test Acc ({name})', linestyle='dotted', color=color)
        plt.plot(group['Epoch'], group['Train Acc'], label=f'Train Acc ({name})', color=color)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{(neuron_name)} Neuron Test & Train Accuracy vs. Backpropagation Algorithms")

    # Move the legend to the right of the chart
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f"{(neuron_name)}_chart.png", dpi=400, bbox_inches='tight')
    plt.show()
    print(f"{(neuron_name)}_chart.png saved")