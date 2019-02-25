import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ChainPlotter:

    @staticmethod
    def chain_plot(data, title='', ax=None):
        '''Plot both chain and posterior distribution'''
        if ax is None:
            ax = plt.gca()
        ax.plot(data)
        ax.title.set_text(title + " chain")

    @staticmethod
    def post_plot(data, title='', ax=None, true=None, prc=95):
        '''Plot the posterior distribution given MCMC samples'''
        if ax is None:
            ax = plt.gca()
        sns.kdeplot(data, ax=ax, shade=True)
        ax.axvline(x=np.percentile(data, (100 - prc) / 2), linestyle='--')
        ax.axvline(x=np.percentile(data, 100 - (100 - prc) / 2), linestyle='--')
        ax.title.set_text(title + " distribution")
        if true is not None:
            ax.axvline(x=true)

    @staticmethod
    def chain_post_plot(data, title='', ax=None, true=None):
        '''Plot a chain of MCMC samples'''
        ChainPlotter.chain_plot(data, title=title, ax=ax[0])
        ChainPlotter.post_plot(data, title=title, ax=ax[1], true=true)

    @staticmethod
    def plot(data, coeffs_samples, bias_samples, noise_std_samples):
        # Plot chains and distributions for coefficients
        fig, axes = plt.subplots(data.number_of_dimensions + 2, 2, sharex='col', sharey='col')
        fig.set_size_inches(6.4, 8)
        for i in range(data.number_of_dimensions):
            ChainPlotter.chain_post_plot(coeffs_samples[:, i],
                            title="x[{}]".format(i),
                            ax=axes[i], true=data.w_true[i])

        # Plot chains and distributions for bias
        ChainPlotter.chain_post_plot(bias_samples[:, 0],
                        title="Bias",
                        ax=axes[data.number_of_dimensions], true=data.b_true)

        # Plot chains and distributions for noise std dev
        ChainPlotter.chain_post_plot(noise_std_samples[:, 0],
                        title="Noise std",
                        ax=axes[data.number_of_dimensions + 1], true=data.noise_std_true)

        axes[data.number_of_dimensions + 1][1].set_xlabel("Parameter value")
        fig.tight_layout()
        plt.show()