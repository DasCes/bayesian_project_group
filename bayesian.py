import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def psycometrique(data_tuple):
    result, mean_used = data_tuple
    true_array_mean_used = []
    true_array_result = []
    n = len(result)
    block_size = 50
    big_blocs = 11
    size_one_block = block_size * big_blocs
    
    val = list(range(0, n, block_size))
    
    for j in range(4):
        array_mean_result = []
        array_mean_used = []
        for i in range(0, size_one_block, block_size):
            # bloc de result
            block_r = result[size_one_block*j + i : size_one_block*j + i+block_size]
            mean_r = sum(block_r) / len(block_r)
            array_mean_result.append(mean_r)

            # bloc de mean_used
            block_m = mean_used[size_one_block*j + i : size_one_block*j + i+block_size]
            mean_m = sum(block_m) / len(block_m)
            array_mean_used.append(mean_m)

        true_array_result.append(array_mean_result)
        true_array_mean_used.append(array_mean_used)
    return true_array_result, true_array_mean_used


def ploter(true_array_mean_used, true_array_result):
    plt.figure(figsize=(7,5))

    for idx, (mean_used, mean_res) in enumerate(zip(true_array_mean_used, true_array_result)):
        plt.plot(mean_used, mean_res, 'o-', label=f'Bloc {idx+1}')

    plt.xlabel('Mean S2 value')
    plt.ylabel('Mean decision (P[1])')
    plt.title('Psychometric Functions (all curves)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    
def reading_csv(file_path):
    df = pd.read_csv(file_path)
    mean_used = df['S2_val'].tolist()
    result = df['Decision (S1>S2)'].tolist()
    data_tuple = (mean_used, result)
    print(len(result))
    return data_tuple

def main():
    print('test')
    file_path = "C:\\Users\\gabri\\Desktop\\bayesian\\experiment_results.csv"  # Replace with your CSV file path
    data_tuple = reading_csv(file_path)
    array_mean_result, array_mean_used = psycometrique(data_tuple)
    ploter(array_mean_result, array_mean_used)
    
    
if __name__ == "__main__":
    main()