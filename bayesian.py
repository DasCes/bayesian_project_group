import matplotlib.pyplot as plt
import pandas as pd



def psycometrique(data_tuple):
    result, mean_used = data_tuple
    array_mean_result = []
    array_mean_used = []
    n = len(result)
    block_size = 50

    for i in range(0, n, block_size):
        # bloc de result
        block_r = result[i:i+block_size]
        mean_r = sum(block_r) / len(block_r)
        array_mean_result.append(mean_r)

        # bloc de mean_used
        block_m = mean_used[i:i+block_size]
        mean_m = sum(block_m) / len(block_m)
        array_mean_used.append(mean_m)

    return array_mean_result, array_mean_used


def ploter(array_mean_s2, array_mean_decision):
    plt.figure(figsize=(6,4))
    plt.plot(array_mean_s2, array_mean_decision, 'o-', color='steelblue', label='Psychometric curve')
    plt.xlabel('Mean S2 value')          # abscisse : stimulus
    plt.ylabel('Mean decision (P[1])')   # ordonnée : proba de répondre 1
    plt.title('Psychometric Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    
def reading_csv(file_path):
    df = pd.read_csv(file_path)
    mean_used = df['S2_val'].tolist()
    result = df['Decision (S1>S2)'].tolist()
    data_tuple = (mean_used, result)
    return data_tuple

def main():
    print('test')
    file_path = "C:\\Users\\gabri\\Desktop\\bayesian\\experiment_results.csv"  # Replace with your CSV file path
    data_tuple = reading_csv(file_path)
    array_mean_result, array_mean_used = psycometrique(data_tuple)
    ploter(array_mean_result, array_mean_used)
    
    
if __name__ == "__main__":
    main()