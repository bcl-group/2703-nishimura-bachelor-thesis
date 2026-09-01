# main.py
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from model import Field
from config import *

# 1回のシミュレーションを実行し、最後10000ステップの平均を返す関数
def run_single_simulation(death_threshold):
    simulation = Field(death_threshold)
    
    pop_list = []
    alt_list = []
    
    record_start_step = TOTAL_STEPS - EVAL_STEPS
    
    for step_num in range(1, TOTAL_STEPS + 1):
        simulation.step()
        
        # 最後の10000ステップだけ記録
        if step_num > record_start_step:
            pop_list.append(len(simulation.agents))
            alt_list.append(simulation.altruism_count)
            
        simulation.altruism_count = 0 # 毎ステップリセット
        
        # 全滅した場合はループを打ち切り、残りのステップは0として扱う
        if len(simulation.agents) == 0:
            remaining_eval_steps = EVAL_STEPS - len(pop_list)
            if remaining_eval_steps > 0:
                pop_list.extend([0] * remaining_eval_steps)
                alt_list.extend([0] * remaining_eval_steps)
            break
            
    # この1回のシミュレーションにおける、最終10000ステップの時間平均を返す
    return np.mean(pop_list), np.mean(alt_list)

def main():
    # 死亡閾値のリスト (例として-300から0まで 50刻み。1刻み等に変更可能)
    thresholds = list(range(-300, 1, 1)) 
    trials = 10 # 各閾値ごとのシミュレーション回数
    
    mean_populations = []
    std_populations = []
    mean_altruism = []
    std_altruism = []

    print("シミュレーションを開始します。計算には時間がかかる場合があります...")

    for threshold in thresholds:
        print(f"計算中: DEATH_THRESHOLD = {threshold}")
        
        trial_pop_means = []
        trial_alt_means = []
        
        # 並列処理で100回シミュレーションを実行
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 100回分のタスクを投げる
            results = list(executor.map(run_single_simulation, [threshold] * trials))
        
        for pop_mean, alt_mean in results:
            trial_pop_means.append(pop_mean)
            trial_alt_means.append(alt_mean)
            
        # 100回のシミュレーション結果（時間平均）の、さらに平均と標準偏差を計算
        mean_populations.append(np.mean(trial_pop_means))
        std_populations.append(np.std(trial_pop_means))
        mean_altruism.append(np.mean(trial_alt_means))
        std_altruism.append(np.std(trial_alt_means))

    print("全シミュレーション完了。グラフを作成します。")

    # --- 個体数のグラフ作成 ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(thresholds, mean_populations, yerr=std_populations, 
                 fmt='-o', color='g', ecolor='lightgreen', capsize=5)
    plt.title('Population Size (Last 10000 Steps) vs Death Threshold', fontsize=15)
    plt.xlabel('Death Threshold', fontsize=15)
    plt.ylabel('Population Size (Mean ± Std)', fontsize=15)
    plt.tick_params(labelsize=15)
    plt.grid(True)
    plt.savefig('population_vs_threshold.png')
    plt.close()

    # --- 利他行動回数のグラフ作成 ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(thresholds, mean_altruism, yerr=std_altruism, 
                 fmt='-o', color='b', ecolor='lightblue', capsize=5)
    plt.title('Altruistic Actions (Last 10000 Steps) vs Death Threshold', fontsize=15)
    plt.xlabel('Death Threshold', fontsize=15)
    plt.ylabel('Altruistic Actions Count (Mean ± Std)', fontsize=15)
    plt.tick_params(labelsize=15)
    plt.grid(True)
    plt.savefig('altruism_vs_threshold.png')
    plt.close()

    print("グラフを保存しました。")

if __name__ == '__main__':
    main()