import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def psycometrique(data_tuple,var2):
    result, mean_used = data_tuple
    true_array_mean_used = []
    true_array_result = []
    block_size = 200          
    big_blocs = 11           # nb de valeurs de S2 par courbe
    size_one_block = block_size * big_blocs  

    for j in range(len(var2)):
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


def data_var2(file_path):
    df = pd.read_csv(file_path)
    var2 = df['S2_std'].to_numpy()
    var2 = pd.unique(var2)      
    return list(var2)           


def ploter(mean_used_all, mean_res_all, var, var1=0.2):
    """
    Combine les deux anciens graphiques :
    - subplot 1 : psychometric curves (x_star search)
    - subplot 2 : slopes = x_star / (var + var1)
    """

    seuil = 0.5
    x_stars = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    # ==========================================
    # ---------- SUBPLOT 1 : courbes ----------
    # ==========================================
    print("=== Points d’intersection avec P = 0.5 ===")

    for idx, (x_vals, y_vals) in enumerate(zip(mean_used_all, mean_res_all)):

        # tracer la courbe
        line, = ax1.plot(x_vals, y_vals, 'o-', label=f'var = {var[idx]}')
        color = line.get_color()

        # --- recherche du passage par 0.5 ---
        x_cross = None
        for x1, y1, x2, y2 in zip(x_vals[:-1], y_vals[:-1],
                                  x_vals[1:],  y_vals[1:]):
            if (y1 - seuil) * (y2 - seuil) <= 0 and y1 != y2:
                x_cross = x1 + (seuil - y1) * (x2 - x1) / (y2 - y1)
                break

        if x_cross is not None:
            ax1.scatter([x_cross], [seuil], color=color, zorder=5)
            ax1.text(x_cross, seuil + 0.03, f'{x_cross:.2f}',
                     color=color, ha='center', va='bottom', fontsize=8)
            print(f'Bloc {idx+1} : x = {x_cross:.3f}')
        else:
            print(f'Bloc {idx+1} : pas de croisement avec 0.5')

        x_stars.append(x_cross)

    ax1.axhline(seuil, color='red', linestyle='--', linewidth=1.5,
                label='Chance level (0.5)')

    ax1.set_xlabel('Mean S2 value')
    ax1.set_ylabel('Mean decision (P[1])')
    ax1.set_title('Psychometric Functions')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ==========================================
    # ---------- SUBPLOT 2 : slopes -----------
    # ==========================================

    slopes = slope_x(x_stars, var, var1)
    ax2.plot(range(1, len(slopes)+1), slopes, marker='o')
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Slope (≈ m0 / s0²)")
    ax2.set_title("Slope evolution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return x_stars, slopes


def reading_csv(file_path):
    df = pd.read_csv(file_path)
    mean_used = df['S2_val'].tolist()
    result = df['Decision (S1>S2)'].tolist()
    data_tuple = (result, mean_used)  
    print(len(result))
    return data_tuple


def slope_x(x_star, var2, var1=0.2):
    slopes = []
    for x, v in zip(x_star, var2):
        slopes.append(x / (v**2 + var1**2))

    return slopes


def main():
    print('test')
    file_path = "C:\\Users\\gabri\\Desktop\\bayesian\\experiment_results_test.csv"
    data_tuple = reading_csv(file_path)
    var2 = data_var2(file_path)
    array_mean_result, array_mean_used = psycometrique(data_tuple, var2)
    x_stars, slopes = ploter(array_mean_used, array_mean_result, var2)
    print("Slopes (≈ m0 / s0^2) :", slopes)
    print("x_stars :", x_stars)

if __name__ == "__main__":
    main()


# we have m0 / so**2 = 0.15 in average 