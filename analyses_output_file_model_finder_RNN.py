'''
Author: Thomas Ross
Analyses output file of model_finder_RNN
'''
import re
import os

def find_lowest_maes_and_layers(filename, n_lowest=10):
    with open(filename, 'r') as file:
        lines = file.readlines()

    trial_pattern = re.compile(r'Trial (\d+):')
    mae_pattern = re.compile(r'- val_mae: ([\d.]+)')  # Change this line to capture val_mae

    current_trial = None
    last_mae = None
    lowest_maes = []
    layers_info = {}
    current_layers = []
    current_batch_size = None
    current_learning_rate = None
    current_output_activation = None

    for line in lines:
        trial_match = trial_pattern.search(line)
        mae_match = mae_pattern.search(line)

        if trial_match:
            if current_trial is not None and last_mae is not None:
                # Insert in a sorted manner
                index = 0
                while index < len(lowest_maes) and lowest_maes[index][0] < last_mae:
                    index += 1
                lowest_maes.insert(index, (last_mae, current_trial))

                layers_info[current_trial] = {
                    "layers": current_layers,
                    "batch_size": current_batch_size,
                    "learning_rate": current_learning_rate,
                    "output_activation": current_output_activation
                }

                if len(lowest_maes) > n_lowest:
                    removed_mae, removed_trial = lowest_maes.pop()
                    del layers_info[removed_trial]

            current_trial = int(trial_match.group(1))
            last_mae = None
            current_layers = []
            current_batch_size = None
            current_learning_rate = None
            current_output_activation = None

        if mae_match:
            last_mae = float(mae_match.group(1))

        if current_trial:
            if "Layer" in line:
                current_layers.append(line.strip())
            elif "Batch size:" in line:
                current_batch_size = line.strip()
            elif "Learning rate:" in line:
                current_learning_rate = line.strip()
            elif "Output activation:" in line:
                current_output_activation = line.strip()

    # Check the last trial
    if last_mae is not None:
        index = 0
        while index < len(lowest_maes) and lowest_maes[index][0] < last_mae:
            index += 1
        lowest_maes.insert(index, (last_mae, current_trial))

        layers_info[current_trial] = {
            "layers": current_layers,
            "batch_size": current_batch_size,
            "learning_rate": current_learning_rate,
            "output_activation": current_output_activation
        }

        if len(lowest_maes) > n_lowest:
            removed_mae, removed_trial = lowest_maes.pop()
            del layers_info[removed_trial]

    return lowest_maes[:n_lowest], layers_info

filename = 'output_model_finder.txt'  # Replace this with your output file name
file_base, _ = os.path.splitext(filename)
output_file = f"{file_base}_lowest_val_maes_info.txt"  # Output file name
lowest_maes, layers_info = find_lowest_maes_and_layers(filename)

for i, (mae, trial) in enumerate(lowest_maes, 1):
    print(f'Lowest Val MAE {i}: {mae} (Trial {trial})')
    print("Layers:")
    print(layers_info[trial]["batch_size"])
    for layer in layers_info[trial]["layers"]:
        print(layer)
    print(layers_info[trial]["learning_rate"])
    print(layers_info[trial]["output_activation"])
    print()

# save in .txt file
with open(output_file, 'w') as file:
    for i, (mae, trial) in enumerate(lowest_maes, 1):
        file.write(f'Lowest Val MAE {i}: {mae} (Trial {trial})\n')
        file.write("Layers:\n")
        file.write(layers_info[trial]["batch_size"] + '\n')
        for layer in layers_info[trial]["layers"]:
            file.write(layer + '\n')
        file.write(layers_info[trial]["learning_rate"] + '\n')
        file.write(layers_info[trial]["output_activation"] + '\n')
        file.write('\n')

