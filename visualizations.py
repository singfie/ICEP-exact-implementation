import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def individual_box_plots(df, data_path):

    fig, axs = plt.subplots(ncols=5, figsize = (15,5), gridspec_kw=dict(width_ratios=(4,2,2,2,1)))

    # number of updates
    g0 = sns.boxplot(x="number_updates", y="evac_time_true", data=df, ax=axs[0])
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Number of information updates')

    # # gamma setting
    # g1 = sns.boxplot(x="gamma_setting", y="evac_time_true", data=df, ax=axs[0])
    # g1.set(ylabel='Evacuation time obtained with different gamma settings')
    # # g1.set(xticklabels=['low', 'high'])
    # g1.set(xlabel='Gamma setting (conservativeness)')

    # travel matrix var
    g2 = sns.boxplot(x="update_interval", y="evac_time_true", data=df, ax=axs[1])
    g2.set(yticklabels=[])  # remove the tick labels
    g2.set(ylabel=None)  # remove the axis label
    # g2.set(xticklabels=['low', 'high'])
    g2.set(xlabel='Update interval in min')

    # demand variance factor over iterations
    g3 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df, ax=axs[2])
    g3.set(yticklabels=[])  # remove the tick labels
    g3.set(ylabel=None)
    # g3.set(xticklabels=['low', 'high'])
    g3.set(xlabel='Variance factor relative to true demand')

    # demand variance factor over iterations
    g4 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df, ax=axs[3])
    g4.set(yticklabels=[])  # remove the tick labels
    g4.set(ylabel=None)
    # g4.set(xticklabels=['low', 'high'])
    g4.set(xlabel='Demand/capacity ratio')

    # total distribution
    g5 = sns.boxplot(y="evac_time_true", data=df, ax=axs[4])
    g5.set(yticklabels=[])  # remove the tick labels
    g5.set(ylabel=None)  # remove the axis label
    g5.set(xticklabels=[])  # remove the tick labels
    g5.set(xlabel='Population distribution')

    fig.set_facecolor("white")
    fig.suptitle('True evacuation times under varying parameter settings',
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/box_individual_factors.png'), dpi=300, transparent=False)

    for intervals in np.unique(df['update_interval']):

        for updates in np.unique(df['number_updates']):

            data = df[(df['update_interval'] == intervals) & (df['number_updates'] == updates)]

            fig, axs = plt.subplots(ncols=3, figsize = (8,5), gridspec_kw=dict(width_ratios=(2,2,1)))

            # demand variance factor over iterations
            g3 = sns.boxplot(x="variance_factor", y="evac_time_true", data=data, ax=axs[0])
            g3.set(ylabel=None)
            # g3.set(xticklabels=['low', 'high'])
            g3.set(xlabel='Variance factor relative to true demand')

            # demand variance factor over iterations
            g4 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=data, ax=axs[1])
            g4.set(yticklabels=[])  # remove the tick labels
            g4.set(ylabel=None)
            # g4.set(xticklabels=['low', 'high'])
            g4.set(xlabel='Demand/capacity ratio')

            # total distribution
            g5 = sns.boxplot(y="evac_time_true", data=data, ax=axs[2])
            g5.set(yticklabels=[])  # remove the tick labels
            g5.set(ylabel=None)  # remove the axis label
            g5.set(xticklabels=[])  # remove the tick labels
            g5.set(xlabel='Population distribution')

            fig.set_facecolor("white")
            fig.suptitle('True evacuation times under varying parameter settings',
                         ha='center',
                         fontsize=15,
                         fontweight=20)
            plt.savefig(os.path.join(data_path.split('/')[0], 'figures/box_individual_factors_i_' + str(intervals) + '_u_' + str(updates) + '.png'), dpi=300, transparent=False)

    return(-1)

def line_plots(df, data_path):

    # summary plot
    plt.figure()
    sns.lineplot(data=df, x="variance_factor", y="evac_time_true", hue="number_updates", style="update_interval",
                 units="id",estimator=None, lw=1)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/lineplot_x_variance_factor.png'), dpi=300, transparent=False)

    # individual plots

    # for intervals in np.unique(df['update_interval']):
    #
    #     for updates in np.unique(df['number_updates']):
    #
    #         data = df[(df['update_interval'] == intervals) & (df['number_updates'] == updates)]
    #
    #         plt.figure()
    #         sns.lineplot(data=data, x="variance_factor", y="evac_time_true", hue="number_updates", style="update_interval")
    #         plt.ylim(0,600)
    #         plt.savefig(os.path.join(data_path.split('/')[0], 'figures/lineplot_x_variance_factor_i_' + str(intervals) + '_u_' + str(updates) + '.png'), dpi=300, transparent=False)
    #
    # return(-1)

def main():

    parser = argparse.ArgumentParser(description='key parameters')
    parser.add_argument('-d', '--experiment_data', type=str)

    args = parser.parse_args()

    data_path = args.experiment_data

    df = pd.read_csv(os.path.join(os.getcwd(), data_path))

    ####  VISUALIZATIONS  ####

    # box plots
    individual_box_plots(df, data_path)

    # line plots
    # line_plots(df, data_path)


if __name__ == "__main__":
    main()