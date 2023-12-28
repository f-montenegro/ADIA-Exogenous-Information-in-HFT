from L0_Library.config import *


class DisplayResults:
    def __init__(self, df_dict: pd.DataFrame):
        
        self.df_dict = df_dict

    #####################################################################
    ########################### Jump Quantity ###########################
    #####################################################################

    def jump_quantity(self):
        df_jumps = self.df_dict["jumps"]
        
        K = df_jumps.shape[0]
        S_K = (2 * np.log(K))**(-0.5)
        C_K = (2 * np.log(K))**0.5 - (np.log(np.pi) + np.log(np.log(K))) / (2 * (2 * np.log(K))**0.5)
        alpha = 0.01

        threshold = C_K - S_K * np.log(np.log(1/(1-alpha)))

        number_jumps = (np.abs(df_jumps) > threshold).sum().sum()
        return number_jumps
        