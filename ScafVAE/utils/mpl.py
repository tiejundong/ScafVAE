import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget
import seaborn as sns



class info_recorder():
    def __init__(self, name):
        self.name = name
        self.trj_dic = {}
        self.batch_dic = {}

    def reset_trj(self, k):
        self.trj_dic[k] = []
        self.batch_dic[k] = []

    def update_trj(self, batch_size=1):
        for k in self.trj_dic.keys():
            self.trj_dic[k].append(np.nanmean(self.batch_dic[k]) / batch_size)
            self.batch_dic[k] = []

    def __call__(self, k, x):
        self.batch_dic[k].append(x)

    def save_trj(self):
        np.savez('./{}.npz'.format(self.name), **self.trj_dic)

    def load_history(self, restart=None):
        history = np.load('./{}.npz'.format(self.name))
        if restart == None:
            for k in history.keys():
                self.trj_dic[k] = list(history[k])
        else:
            for k in history.keys():
                self.trj_dic[k] = list(history[k])[:restart]

    def plot(self, y_lim=None, plot_flag=True, figsize=5):
        text_size = 17
        ticklabel_size = 15
        legend_size = 15
        color_palette = sns.color_palette()
        fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))

        for i, k in enumerate(self.trj_dic.keys()):
            x = np.arange(len(self.trj_dic[k]))
            ax.plot(x, self.trj_dic[k], label=k, color=color_palette[i])

        ax.set_title(self.name, fontsize=text_size)
        ax.grid(False)
        if y_lim != None:
            ax.set_ylim((y_lim[0], y_lim[1]))
        # ax.set_ylabel('loss', fontsize=text_size)
        ax.set_xlabel('Iterations', fontsize=text_size)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)
        ax.axvline(x=0, color='grey', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        # x_major_locator=plt.MultipleLocator(20000)
        # ax.xaxis.set_major_locator(x_major_locator)
        ax.tick_params(labelsize=ticklabel_size)
        [ax.spines[ax_j].set_color('black') for ax_j in ax.spines.keys()]
        ax.tick_params(bottom=True, left=True, direction='out', width=2, length=5, color='black')
        fig.tight_layout()
        ax.legend(frameon=False, prop={'size': legend_size}) #, bbox_to_anchor=(1, 1))  # loc='upper right', markerscale=1000)
        if plot_flag:
            plt.show()
        else:
            plt.savefig('./' + self.name + '.svg', bbox_inches='tight', dpi=600)
        plt.close()

    def print_info(self):
        print(self.name)
        for k in self.trj_dic.keys():
            print(k.ljust(20, '.') + '{:.6f}'.format(self.trj_dic[k][-1]).rjust(20, '.'))



if __name__ == '__main__':
    # for testing
    pass
