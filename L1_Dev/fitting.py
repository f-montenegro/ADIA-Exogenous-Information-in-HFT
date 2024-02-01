from L0_Library.config import *


class Fitting:

    def __init__(self):
        pass

    @staticmethod
    def jump_series(df_input: pd.DataFrame, df_jump_score: pd.DataFrame,
                    input_list: list, window_size: int
                    ) -> tuple[dict, dict]:

        df_input = df_input.reset_index(drop=True)
        df_jump_score = df_jump_score.reset_index(drop=True)

        idx = {}
        for i in input_list:
            idx_temp = []

            for j in df_input[df_input[i] == 1].index:
                idx_temp.append(j)
            idx[i] = idx_temp

        dict_result = {}
        for key in idx:
            dict_result[key] = {}

            for i, value in enumerate(idx[key]):
                if len(idx[key]) == 1:
                    left = max(0, value - window_size // 2)
                    right = min(389, value + window_size // 2)
                    s_jump_score = abs(df_jump_score.loc[left:right + 1, key])
                    s_jump_score.index = s_jump_score.index - value
                    dict_result[key][value] = s_jump_score

                elif i == 0:
                    left = max(0, value - window_size // 2)
                    right = min((value + idx[key][i + 1]) // 2, value + window_size // 2)
                    s_jump_score = abs(df_jump_score.loc[left:right + 1, key])
                    s_jump_score.index = s_jump_score.index - value
                    dict_result[key][value] = s_jump_score

                elif i == len(idx[key]):
                    left = max((idx[key][i - 1] + value) // 2, value - window_size // 2)
                    right = min((value + idx[key][i + 1]) // 2, value + window_size // 2)
                    s_jump_score = abs(df_jump_score.loc[left:right + 1, key])
                    s_jump_score.index = s_jump_score.index - value
                    dict_result[key][value] = s_jump_score

                else:
                    left = max((idx[key][i - 1] + value) // 2, value - window_size // 2)
                    right = min(389, value + window_size // 2)
                    s_jump_score = abs(df_jump_score.loc[left:right + 1, key])
                    s_jump_score.index = s_jump_score.index - value
                    dict_result[key][value] = s_jump_score

        return idx, dict_result
