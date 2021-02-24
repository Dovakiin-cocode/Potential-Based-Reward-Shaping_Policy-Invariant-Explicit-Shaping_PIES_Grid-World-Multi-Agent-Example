from PBSR_PIES_SARSA.PIES_SARSA_Env import Env
from PBSR_PIES_SARSA.PIES_SARSA_Agent import Agent
import pandas as pd
import numpy as np

if __name__ == '__main__':
    num_moves_df = {}
    col_names = ["Q_Learning", "SARSA", "SARSA_PIES"]
    result = pd.DataFrame()
    for title in col_names:
        for i in range(0, 10):
            env = Env()
            env.do_experiment(title)
            num_moves_df[i] = env.moves_to_goal
            # print("i",i,"moves_to_goal",env.moves_to_goal)
        data = pd.DataFrame(num_moves_df)
        # print("data ",data)
        col_mean = np.mean(data.values, axis=1)
        # print("col_name ", col_mean," type ",type(col_mean))
        result[title] = col_mean
    result.to_csv("Result.csv")
