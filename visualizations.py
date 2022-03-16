import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def comparison_plots_updates(df, data_path, seed, demand_ratio):

    fig, axs = plt.subplots(ncols=7, figsize = (25,5), gridspec_kw=dict(width_ratios=(4,4,4,4,4,4,4)))

    df = df[(df['random_seed'] == seed) & (df['demand_capacity_ratio'] == demand_ratio)]

    # number of updates
    g0 = sns.lineplot(x="number_updates", y="evac_time_true", data=df[df['model'] == 'D-ICEP BENCHMARK'], ax=axs[0])
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Number of information updates')
    g0.set(title='Benchmark')

    g1 = sns.lineplot(x="number_updates", y="evac_time_true", data=df[df['model'] == 'D-ICEP'], ax=axs[1])
    # g0.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g1.set(xlabel='Number of information updates')
    g1.set(title='D-ICEP')

    g2 = sns.lineplot(x="number_updates", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 0)], ax=axs[2])
    # g0.set(yticklabels=[])  # remove the tick labels
    g2.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g2.set(xlabel='Number of information updates')
    g2.set(title='R-ICEP - Gamma: 0')

    g3 = sns.lineplot(x="number_updates", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 1)], ax=axs[3])
    # g0.set(yticklabels=[])  # remove the tick labels
    g3.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g3.set(xlabel='Number of information updates')
    g3.set(title='R-ICEP - Gamma: 1')

    g4 = sns.lineplot(x="number_updates", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 2)], ax=axs[4])
    # g0.set(yticklabels=[])  # remove the tick labels
    g4.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g4.set(xlabel='Number of information updates')
    g4.set(title='R-ICEP - Gamma: 2')

    g5 = sns.lineplot(x="number_updates", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 3)], ax=axs[5])
    # g0.set(yticklabels=[])  # remove the tick labels
    g5.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g5.set(xlabel='Number of information updates')
    g5.set(title='R-ICEP - Gamma: 3')

    g6 = sns.lineplot(x="number_updates", y="evac_time_true", data=df[df['model'] == 'RH-ICEP'], ax=axs[6])
    # g0.set(yticklabels=[])  # remove the tick labels
    g6.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g6.set(xlabel='Number of information updates')
    g6.set(title='RH-ICEP')

    fig.set_facecolor("white")
    fig.suptitle('True evacuation times vs information updates under models',
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/box_model_comparisons_updates_' + str(seed) + '.png'), dpi=300, transparent=False)
    plt.close()

def comparison_plots_interval(df, data_path, seed, demand_ratio):

    fig, axs = plt.subplots(ncols=7, figsize = (25,5), gridspec_kw=dict(width_ratios=(2,2,2,2,2,2,2)))

    df = df[(df['random_seed'] == seed) & (df['demand_capacity_ratio'] == demand_ratio)]

    # number of updates
    g0 = sns.boxplot(x="update_interval", y="evac_time_true", data=df[df['model'] == 'D-ICEP BENCHMARK'], ax=axs[0])
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Update interval')
    g0.set(title='Benchmark')

    g1 = sns.boxplot(x="update_interval", y="evac_time_true", data=df[df['model'] == 'D-ICEP'], ax=axs[1])
    # g0.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g1.set(xlabel='Update interval')
    g1.set(title='D-ICEP')

    g2 = sns.boxplot(x="update_interval", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 0)], ax=axs[2])
    # g0.set(yticklabels=[])  # remove the tick labels
    g2.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g2.set(xlabel='Update interval')
    g2.set(title='R-ICEP - Gamma: 0')

    g3 = sns.boxplot(x="update_interval", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 1)], ax=axs[3])
    # g0.set(yticklabels=[])  # remove the tick labels
    g3.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g3.set(xlabel='Update interval')
    g3.set(title='R-ICEP - Gamma: 1')

    g4 = sns.boxplot(x="update_interval", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 2)], ax=axs[4])
    # g0.set(yticklabels=[])  # remove the tick labels
    g4.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g4.set(xlabel='Update interval')
    g4.set(title='R-ICEP - Gamma: 2')

    g5 = sns.boxplot(x="update_interval", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 3)], ax=axs[5])
    # g0.set(yticklabels=[])  # remove the tick labels
    g5.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g5.set(xlabel='Number of information updates')
    g5.set(title='R-ICEP - Gamma: 3')

    g6 = sns.boxplot(x="update_interval", y="evac_time_true", data=df[df['model'] == 'RH-ICEP'], ax=axs[6])
    # g0.set(yticklabels=[])  # remove the tick labels
    g6.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g6.set(xlabel='Update interval')
    g3.set(title='RH-ICEP')

    fig.set_facecolor("white")
    fig.suptitle('True evacuation times vs update intervals under models',
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/box_model_comparisons_interval_' + str(seed) + '.png'), dpi=300, transparent=False)
    plt.close()

def comparison_plots_variance(df, data_path, seed, demand_ratio):

    fig, axs = plt.subplots(ncols=7, figsize = (25,5), gridspec_kw=dict(width_ratios=(2,2,2,2,2,2,2)))

    df = df[(df['random_seed'] == seed) & (df['demand_capacity_ratio'] == demand_ratio)]

    # number of updates
    g0 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df[df['model'] == 'D-ICEP BENCHMARK'], ax=axs[0])
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Demand variance factor')
    g0.set(title='Benchmark')

    g1 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df[df['model'] == 'D-ICEP'], ax=axs[1])
    # g0.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g1.set(xlabel='Demand variance factor')
    g1.set(title='D-ICEP')

    g2 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 0)], ax=axs[2])
    # g0.set(yticklabels=[])  # remove the tick labels
    g2.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g2.set(xlabel='Demand variance factor')
    g2.set(title='R-ICEP - Gamma: 0')

    g3 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 1)], ax=axs[3])
    # g0.set(yticklabels=[])  # remove the tick labels
    g3.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g3.set(xlabel='Demand variance factor')
    g3.set(title='R-ICEP - Gamma: 1')

    g4 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 2)], ax=axs[4])
    # g0.set(yticklabels=[])  # remove the tick labels
    g4.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g4.set(xlabel='Demand variance factor')
    g4.set(title='R-ICEP - Gamma: 2')

    g5 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 3)], ax=axs[5])
    # g0.set(yticklabels=[])  # remove the tick labels
    g5.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g5.set(xlabel='Demand variance factor')
    g5.set(title='R-ICEP - Gamma: 3')

    g6 = sns.boxplot(x="variance_factor", y="evac_time_true", data=df[df['model'] == 'RH-ICEP'], ax=axs[6])
    # g0.set(yticklabels=[])  # remove the tick labels
    g6.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g6.set(xlabel='Demand variance factor')
    g6.set(title='RH-ICEP')

    fig.set_facecolor("white")
    fig.suptitle('True evacuation times vs variance factor under models',
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/box_model_comparisons_variance_' + str(seed) + '.png'), dpi=300, transparent=False)
    plt.close()

def comparison_plots_demand(df, data_path, seed, demand_ratio):

    fig, axs = plt.subplots(ncols=7, figsize = (25,5), gridspec_kw=dict(width_ratios=(2,2,2,2,2,2,2)))

    df = df[(df['random_seed'] == seed) & (df['demand_capacity_ratio'] == demand_ratio)]

    # number of updates
    g0 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df[df['model'] == 'D-ICEP BENCHMARK'], ax=axs[0])
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Demand Capacity Ratio')
    g0.set(title='Benchmark')

    g1 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df[df['model'] == 'D-ICEP'], ax=axs[1])
    # g0.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g1.set(xlabel='Demand Capacity Ratio')
    g1.set(title='D-ICEP')

    g2 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 0)], ax=axs[2])
    # g0.set(yticklabels=[])  # remove the tick labels
    g2.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g2.set(xlabel='Demand Capacity Ratio')
    g2.set(title='R-ICEP - Gamma: 0')

    g3 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 1)], ax=axs[3])
    # g0.set(yticklabels=[])  # remove the tick labels
    g3.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g3.set(xlabel='Demand Capacity Ratio')
    g3.set(title='R-ICEP - Gamma: 1')

    g4 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 2)], ax=axs[4])
    # g0.set(yticklabels=[])  # remove the tick labels
    g4.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g4.set(xlabel='Demand Capacity Ratio')
    g4.set(title='R-ICEP - Gamma: 2')

    g5 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df[(df['model'] == 'R-ICEP') & (df['gamma_setting'] == 3)], ax=axs[5])
    # g0.set(yticklabels=[])  # remove the tick labels
    g5.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g5.set(xlabel='Demand Capacity Ratio')
    g5.set(title='R-ICEP - Gamma: 3')

    g6 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=df[df['model'] == 'RH-ICEP'], ax=axs[6])
    # g0.set(yticklabels=[])  # remove the tick labels
    g6.set(ylabel='True evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g6.set(xlabel='Demand Capacity Ratio')
    g6.set(title='RH-ICEP')

    fig.set_facecolor("white")
    fig.suptitle('True evacuation times vs variance factor under models',
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/box_model_demand_capacity_ratio_' + str(seed) + '.png'), dpi=300, transparent=False)
    plt.close()

def overview_plot(data, data_path):

    for demand_capacity_ratio in [2.0, 3.0, 4.0]:
        for variance_factor in [0.2, 0.4, 0.6]:
            df = data[(data['demand_capacity_ratio'] == demand_capacity_ratio) & (data['variance_factor'] == variance_factor)]

            fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20,10))

            df['model_type'] = df['model'] + ' ' + df['gamma_setting'].astype(str)
            for i in range(len(df)):
                if 'nan' in df['model_type'].iloc[i]:
                    df['model_type'].iloc[i] = df['model_type'].iloc[i][0:-4]
                if 'BENCHMARK' in df['model_type'].iloc[i]:
                    df['model_type'].iloc[i] = 'BENCHMARK'

            # df['model_type'] = df['model_type'] + ' ' + df['update_interval'].astype(str)
            # for i in range(len(df)):
            #     if 'RH-ICEP' not in df['model_type'].iloc[i]:
            #         df['model_type'].iloc[i] = df['model_type'].iloc[i][0:-5]

            # number of updates
            models = ["BENCHMARK", "D-ICEP", "RH-ICEP", "R-ICEP 0.0", "R-ICEP 1.0", "R-ICEP 2.0", "R-ICEP 3.0"]
            g0 = sns.boxplot(x="model_type", y="evac_time_true", data=df, ax=axs[0][0], order = models)
            # g0.set(yticklabels=[])  # remove the tick labels
            g0.set(ylabel='Evacuation time')  # remove the axis label
            # g0.set(xticklabels=['low', 'high'])
            g0.set(xlabel='Model type')

            # number of updates
            g1 = sns.barplot(x="model_type", y="evac_time_true", hue = "random_seed", data=df, ax=axs[0][1], order = models)
            # g0.set(yticklabels=[])  # remove the tick labels
            g1.set(ylabel='Evacuation time')  # remove the axis label
            # g0.set(xticklabels=['low', 'high'])
            # g1.set(xlim=(0.2,0.8))
            g1.set(xlabel='Model type')
            g1.legend(loc="lower right", title="Data set seed", title_fontsize="large")

            # number of updates
            g2 = sns.barplot(x="model_type", y="evac_time_true", hue = "update_interval", data=df, ax=axs[1][0], order = models)
            # g0.set(yticklabels=[])  # remove the tick labels
            g2.set(ylabel='Evacuation time')  # remove the axis label
            # g0.set(xticklabels=['low', 'high'])
            g2.set(xlabel='Model type')
            g2.legend(loc="lower right", title="Update interval", title_fontsize="large")

            g3 = sns.boxplot(x="model_type", y="avg_util_true", data=df, ax=axs[1][1], order = models)
            # g0.set(yticklabels=[])  # remove the tick labels
            g3.set(ylabel='Average utilization of resource')  # remove the axis label
            # g0.set(xticklabels=['low', 'high'])
            # g3.set(xlim=(0,0.8))
            g3.set(xlabel='Model type')

            fig.set_facecolor("white")
            fig.suptitle('Evacuation times for demand-capacity ratio ' + str(demand_capacity_ratio) + ' and demand variance factor ' + str(variance_factor),
                         ha='center',
                         fontsize=15,
                         fontweight=20)
            plt.savefig(os.path.join(data_path.split('/')[0], 'figures/overview_plot_' + str(variance_factor) + '_' + str(demand_capacity_ratio) + '.png'), dpi=300, transparent=False)
            plt.close()

    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20,10))

    data['model_type'] = data['model'] + ' ' + data['gamma_setting'].astype(str)
    for i in range(len(data)):
        if 'nan' in data['model_type'].iloc[i]:
            data['model_type'].iloc[i] = data['model_type'].iloc[i][0:-4]
        if 'BENCHMARK' in data['model_type'].iloc[i]:
            data['model_type'].iloc[i] = 'BENCHMARK'

    # data['model_type'] = data['model_type'] + ' ' + data['update_interval'].astype(str)
    # for i in range(len(data)):
    #     if 'RH-ICEP' not in data['model_type'].iloc[i]:
    #         data['model_type'].iloc[i] = data['model_type'].iloc[i][0:-5]

    # number of updates
    models = ["BENCHMARK", "D-ICEP", "RH-ICEP", "R-ICEP 0.0", "R-ICEP 1.0", "R-ICEP 2.0", "R-ICEP 3.0"]
    g0 = sns.boxplot(x="model_type", y="evac_time_true", data=data, ax=axs[0][0], order = models)
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='Evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Model type')

    # number of updates
    g1 = sns.barplot(x="model_type", y="evac_time_true", hue = "variance_factor", data=data, ax=axs[0][1], order = models)
    # g0.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel='Evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    # g1.set(xlim=(0.2,0.8))
    g1.set(xlabel='Model type')
    g1.legend(loc="lower right", title="Variance factor", title_fontsize="large")

    # number of updates
    g2 = sns.barplot(x="model_type", y="evac_time_true", hue = "demand_capacity_ratio", data=data, ax=axs[1][0], order = models)
    # g0.set(yticklabels=[])  # remove the tick labels
    g2.set(ylabel='Evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g2.set(xlabel='Model type')
    g2.legend(loc="lower right", title="Demand Capacity Ratio", title_fontsize="large")

    g3 = sns.boxplot(x="model_type", y="avg_util_true", data=data, ax=axs[1][1], order = models)
    # g0.set(yticklabels=[])  # remove the tick labels
    g3.set(ylabel='Average utilization of resource')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    # g3.set(xlim=(0,0.8))
    g3.set(xlabel='Model type')

    fig.set_facecolor("white")
    fig.suptitle('Evacuation times for different model types',
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/overview_plot.png'), dpi=300, transparent=False)
    plt.close()

    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (10,5))

    data['model_type'] = data['model'] + ' ' + data['gamma_setting'].astype(str)
    for i in range(len(data)):
        if 'nan' in data['model_type'].iloc[i]:
            data['model_type'].iloc[i] = data['model_type'].iloc[i][0:-4]
        if 'BENCHMARK' in data['model_type'].iloc[i]:
            data['model_type'].iloc[i] = 'BENCHMARK'

    # data['model_type'] = data['model_type'] + ' ' + data['update_interval'].astype(str)
    # for i in range(len(data)):
    #     if 'RH-ICEP' not in data['model_type'].iloc[i]:
    #         data['model_type'].iloc[i] = data['model_type'].iloc[i][0:-5]

    # number of updates
    models = ["BENCHMARK", "D-ICEP", "RH-ICEP", "R-ICEP 0.0", "R-ICEP 1.0", "R-ICEP 2.0", "R-ICEP 3.0"]
    g0 = sns.lineplot(x="model_type", y="evac_time_true", data=data, ax=axs)
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='Evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Model type')

    fig.set_facecolor("white")
    fig.suptitle('Evacuation times for different model types',
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/overview_plot_mean.png'), dpi=300, transparent=False)
    plt.close()



def comparison_plots_individual(df, data_path, seed, interval, demand_capacity_ratio, variance_factor):

    data = df[(df['random_seed'] == seed) & (df['update_interval'] == interval) &
              (df['demand_capacity_ratio'] == demand_capacity_ratio) &
              (df['variance_factor'] == variance_factor)]

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (20,5))

    data['model_type'] = data['model'] + ' ' + data['gamma_setting'].astype(str)
    for i in range(len(data)):
        if 'nan' in data['model_type'].iloc[i]:
            data['model_type'].iloc[i] = data['model_type'].iloc[i][0:-4]
        if 'BENCHMARK' in data['model_type'].iloc[i]:
            data['model_type'].iloc[i] = 'BENCHMARK'

    # data['model_type'] = data['model_type'] + ' ' + data['update_interval'].astype(str)
    # for i in range(len(data)):
    #     if 'RH-ICEP' not in data['model_type'].iloc[i]:
    #         data['model_type'].iloc[i] = data['model_type'].iloc[i][0:-5]

    # number of updates
    models = ["BENCHMARK", "D-ICEP", "RH-ICEP", "R-ICEP 0.0", "R-ICEP 1.0", "R-ICEP 2.0", "R-ICEP 3.0"]
    g0 = sns.barplot(x="model_type", y="evac_time_true", data=data, ax=axs[0])
    # g0.set(yticklabels=[])  # remove the tick labels
    g0.set(ylabel='Evacuation time')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    g0.set(xlabel='Model type')

    g1 = sns.barplot(x="model_type", y="avg_util_true", data=data, ax=axs[1])
    # g0.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel='Average utilization of resource')  # remove the axis label
    # g0.set(xticklabels=['low', 'high'])
    # g3.set(xlim=(0,0.8))
    g1.set(xlabel='Model type')

    fig.set_facecolor("white")
    fig.suptitle('Evacuation times for different model types for seed ' + str(seed) +
                 ' update interval: ' + str(interval) +
                 ' variance factor: ' + str(variance_factor) +
                 ' demand-capacity-ratio: ' + str(demand_capacity_ratio),
                 ha='center',
                 fontsize=15,
                 fontweight=20)
    plt.savefig(os.path.join(data_path.split('/')[0], 'figures/perfomance_s_' + str(seed) + '_i_' +
                str(interval) + '_v_' + str(variance_factor) + '_d_' + str(demand_capacity_ratio) +
                '.png'), dpi=300, transparent=False)
    plt.close()

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

    # for intervals in np.unique(df['update_interval']):
    #
    #     for updates in np.unique(df['number_updates']):
    #
    #         data = df[(df['update_interval'] == intervals) & (df['number_updates'] == updates)]
    #
    #         fig, axs = plt.subplots(ncols=3, figsize = (8,5), gridspec_kw=dict(width_ratios=(2,2,1)))
    #
    #         # demand variance factor over iterations
    #         g3 = sns.boxplot(x="variance_factor", y="evac_time_true", data=data, ax=axs[0])
    #         g3.set(ylabel=None)
    #         # g3.set(xticklabels=['low', 'high'])
    #         g3.set(xlabel='Variance factor relative to true demand')
    #
    #         # demand variance factor over iterations
    #         g4 = sns.boxplot(x="demand_capacity_ratio", y="evac_time_true", data=data, ax=axs[1])
    #         g4.set(yticklabels=[])  # remove the tick labels
    #         g4.set(ylabel=None)
    #         # g4.set(xticklabels=['low', 'high'])
    #         g4.set(xlabel='Demand/capacity ratio')
    #
    #         # total distribution
    #         g5 = sns.boxplot(y="evac_time_true", data=data, ax=axs[2])
    #         g5.set(yticklabels=[])  # remove the tick labels
    #         g5.set(ylabel=None)  # remove the axis label
    #         g5.set(xticklabels=[])  # remove the tick labels
    #         g5.set(xlabel='Population distribution')
    #
    #         fig.set_facecolor("white")
    #         fig.suptitle('True evacuation times under varying parameter settings',
    #                      ha='center',
    #                      fontsize=15,
    #                      fontweight=20)
    #         plt.savefig(os.path.join(data_path.split('/')[0], 'figures/box_individual_factors_i_' + str(intervals) + '_u_' + str(updates) + '.png'), dpi=300, transparent=False)

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
    # individual_box_plots(df, data_path)

    overview_plot(df, data_path)

    # for seed in [123, 124, 125, 126, 127]:
    #     for interval in [15.0, 30.0, 60.0]:
    #         for demand_capacity_ratio in [2.0, 3.0, 4.0]:
    #             for variance_factor in [0.2, 0.4, 0.6]:
    #
    #                 comparison_plots_individual(df, data_path, seed, interval, demand_capacity_ratio, variance_factor)


if __name__ == "__main__":
    main()