import os
import subprocess
import time
import pandas as pd
from scipy.integrate import simps
from modifyb18 import *
import numpy as np
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt import acquisition


# import pyautogui


def plot_bo(f, bo):
    x = np.linspace(1, 1.5, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()


def run_trnsys_simulation(dck_file):
    """
    Fonction pour exécuter une simulation TRNSYS avec un fichier .dck donné
    """
    trnsys_exe = r"C:\TRNSYS18\Exe\TrnEXE64.exe "

    if not os.path.exists(trnsys_exe):
        raise FileNotFoundError(f"TRNSYS executable not found at {trnsys_exe}")

    if not os.path.exists(dck_file):
        raise FileNotFoundError(f"DCK file not found: {dck_file}")

    command = [trnsys_exe, dck_file, "/N", "/h"]

    try:
        process = subprocess.Popen(command)

        # Attendre que la simulation se termine
        process.wait()

        # # Attendre un moment pour s'assurer que la fenêtre est prête
        # time.sleep(2)  # Ajustez ce délai si nécessaire

        # # Appuyer sur 'Enter' pour fermer la fenêtre de notification
        # pyautogui.press('enter')

        print(f"Simulation completed successfully for {dck_file}")

        return True

    except Exception as e:
        print(f"An error occurred while running the simulation for {dck_file}: {str(e)}")
        return False


def batch_run_simulations(dck_files):
    """
    Fonction pour exécuter une série de simulations TRNSYS
    """
    results = []
    for dck_file in dck_files:
        print(f"Starting simulation for {dck_file}")
        start_time = time.time()
        print(dck_file)
        success = run_trnsys_simulation(dck_file)
        end_time = time.time()

        results.append({
            'file': dck_file,
            'success': success,
            'duration': end_time - start_time
        })

    return results


# Exemple d'utilisation
def f(x):
    print(x)
    modifications = {
        'CONSTRUCTIONS': {
            'EXT_WALL': {'thickness': [0.015 * x, 0.175 * x, 0.167 * x, 0.021 * x]},
        },
    }

    # modify_b18_file('Simple-Step3.b18', 'Modified-Simple-Step3_1.b18', modifications)

    input_file = "C:\\TRNSYS18\\Examples\\3D_Building\\6_Step_Add_Daylight\\Building_step6.b18"
    output_file = "C:\\TRNSYS18\\Examples\\3D_Building\\6_Step_Add_Daylight\\Building_step62.b18"

    modify_b18_file(input_file, output_file, modifications)

    dck_files = [
        # "./outputs/Modified-Simple-Step3_1.dck",
        # "./outputs/Modified-Simple-Step3_2.dck",
        "C:\\TRNSYS18\\Examples\\3D_Building\\6_Step_Add_Daylight\\Building_step6.dck"
    ]

    results = batch_run_simulations(dck_files)

    print("\nSimulation Results:")
    for result in results:
        status = "Success" if result['success'] else "Failed"
        print(f"{result['file']}: {status} (Duration: {result['duration']:.2f} seconds)")

    df = pd.read_csv('C:\\TRNSYS18\\Examples\\3D_Building\\6_Step_Add_Daylight\\Building_step6.csv')

    data = {
        "time": df.loc[2::2].values.T[0],
        "power": df.loc[1::2].values.T[0]
    }

    energy = pd.DataFrame(data)
    result = simps(energy['time'], energy['power'])  # Integrate y with respect to x
    print(result, "cost: ", -abs(result - 200000))
    return -abs(result - 200000)


acquisition_function = acquisition.UpperConfidenceBound(kappa=0.1)

bo = BayesianOptimization(
    f=f,
    acquisition_function=acquisition_function,
    pbounds={"x": (1, 1.5)},
    verbose=0,
    random_state=987234,
)
bo.maximize(n_iter=10)

costs = [-res['target'] + 200000 for res in bo.res]  # Invert if needed for minimization
plt.plot(costs, marker="o", label="Real energy usage")
plt.plot(200000 * np.ones(15), marker="o", label="Objective energy usage")
plt.title("Énergie de chauffage nécéssaire>")  # Set plot title
plt.xlabel("Itération")  # Set X-axis label
plt.ylabel("Chauffage (J/m^2)")  # Set Y-axis label

# Set X and Y axis limits
plt.ylim(150000, 260000)  # Set Y-axis range

plt.show()
