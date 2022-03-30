import pandas as pd
import argparse
import os
import researchpy as rp
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
import scipy.stats as stats
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--path_to_file", help="the path of the summary files")

    args = parser.parse_args()

    path = args.path_to_file

    # read in file
    data = pd.read_csv(os.path.join(os.getcwd(), path))
    # data = data[data['dataset'] == 'small']

    #### RH-ICEP

    # analyze what benefits the RH-ICEP

    rh_data = data[data['model'] == 'RH-ICEP']
    benchmark_data = data[data['model'] == 'D-ICEP BENCHMARK']

    rh_data['diff_benchmark'] = 0
    for i in range(len(rh_data)):
        bench = benchmark_data['evac_time_true'][(benchmark_data['dataset'] == rh_data['dataset'].iloc[i]) &
                                                 (benchmark_data['random_seed'] == rh_data["random_seed"].iloc[i]) &
                                                 (benchmark_data['demand_capacity_ratio'] == rh_data['demand_capacity_ratio'].iloc[i]) &
                                                 (benchmark_data['update_interval'] == rh_data['update_interval'].iloc[i]) &
                                                 (benchmark_data['variance_factor'] == rh_data['variance_factor'].iloc[i])].values
        # print(bench)
        if len(bench) != 0:
            rh_data['diff_benchmark'].iloc[i] = rh_data['evac_time_true'].iloc[i] / bench[0]
        else:
            rh_data['diff_benchmark'].iloc[i] = None

    relevant_data = rh_data[['diff_benchmark', 'demand_capacity_ratio', 'update_interval', 'variance_factor', 'dataset']]

    model = ols("diff_benchmark ~ C(demand_capacity_ratio, Sum) + C(update_interval, Sum) + C(variance_factor, Sum) + C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(update_interval, Sum)*C(variance_factor, Sum)*C(dataset, Sum)", data = relevant_data).fit()

    aov_table = sm.stats.anova_lm(model, typ = 3)

    model2 = ols("diff_benchmark ~ C(demand_capacity_ratio, Sum) + C(update_interval, Sum) + C(variance_factor, Sum) + C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(dataset, Sum)", data = relevant_data).fit()

    aov_table2 = sm.stats.anova_lm(model2, typ = 3)

    #### Robust ICEP

    # analyze what benefits the RH-ICEP

    data['corrected_model'] = data['model']
    for j in range(len(data)):
        if (data['dataset'].iloc[j] == 'small') and (data['model'].iloc[j] == 'R-ICEP') and (data['gamma_setting'].iloc[j] == 3.0):
            data['corrected_model'].iloc[j] = 'R-ICEP ROBUST'
        elif (data['dataset'].iloc[j] == '2') and (data['model'].iloc[j] == 'R-ICEP') and (data['gamma_setting'].iloc[j] == 4.0):
            data['corrected_model'].iloc[j] = 'R-ICEP ROBUST'

    r_data = data[(data['model'] == 'R-ICEP') & (data['gamma_setting'] == 0.0)]
    benchmark_data = data[data['model'] == 'D-ICEP BENCHMARK']

    r_data['diff_benchmark_R'] = 0
    for i in range(len(r_data)):
        bench = benchmark_data['evac_time_true'][(benchmark_data['dataset'] == r_data['dataset'].iloc[i]) &
                                                 (benchmark_data['random_seed'] == r_data["random_seed"].iloc[i]) &
                                                 (benchmark_data['demand_capacity_ratio'] == r_data['demand_capacity_ratio'].iloc[i]) &
                                                 (benchmark_data['update_interval'] == r_data['update_interval'].iloc[i]) &
                                                 (benchmark_data['variance_factor'] == r_data['variance_factor'].iloc[i])].values
        # print(bench)
        if len(bench) != 0:
            r_data['diff_benchmark_R'].iloc[i] = r_data['evac_time_true'].iloc[i] / bench[0]
        else:
            r_data['diff_benchmark_R'].iloc[i] = None

    relevant_data = r_data[['diff_benchmark_R', 'demand_capacity_ratio', 'update_interval', 'variance_factor', 'dataset']]

    model_r = ols("diff_benchmark_R ~ C(demand_capacity_ratio, Sum) + C(update_interval, Sum) + C(variance_factor, Sum) + C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(update_interval, Sum)*C(variance_factor, Sum) * C(dataset, Sum)", data = relevant_data).fit()

    aov_table_r = sm.stats.anova_lm(model_r, typ = 3)

    #model2_r = ols("diff_benchmark_R ~ C(demand_capacity_ratio, Sum) + C(update_interval, Sum) + C(variance_factor, Sum) + C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(update_interval, Sum) + C(update_interval, Sum)*C(dataset, Sum) + C(update_interval, Sum)*C(variance_factor, Sum)", data = relevant_data).fit()
    model2_r = ols("diff_benchmark_R ~ C(demand_capacity_ratio, Sum) + C(update_interval, Sum) + C(variance_factor, Sum) + C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(dataset, Sum)", data = relevant_data).fit()


    aov_table2_r = sm.stats.anova_lm(model2_r, typ = 3)

    #### Robust vs RH

    rh_r_data = data[data['model'] == 'RH-ICEP']
    benchmark_data = data[(data['model'] == 'R-ICEP') & (data['gamma_setting'] == 0.0)]

    rh_r_data['diff_RH_R'] = 0
    for i in range(len(rh_data)):
        bench = benchmark_data['evac_time_true'][(benchmark_data['dataset'] == rh_r_data['dataset'].iloc[i]) &
                                                 (benchmark_data['random_seed'] == rh_r_data["random_seed"].iloc[i]) &
                                                 (benchmark_data['demand_capacity_ratio'] == rh_r_data['demand_capacity_ratio'].iloc[i]) &
                                                 (benchmark_data['update_interval'] == rh_r_data['update_interval'].iloc[i]) &
                                                 (benchmark_data['variance_factor'] == rh_r_data['variance_factor'].iloc[i])].values
        print(bench)
        if len(bench) != 0:
            rh_r_data['diff_RH_R'].iloc[i] = rh_r_data['evac_time_true'].iloc[i] / bench[0]
        else:
            rh_r_data['diff_RH_R'].iloc[i] = None

    relevant_data = rh_r_data[['diff_RH_R', 'demand_capacity_ratio', 'update_interval', 'variance_factor', 'dataset']]
    print(relevant_data.head())

    model_r_rh = ols("diff_RH_R ~ C(demand_capacity_ratio, Sum) + C(update_interval, Sum) + C(variance_factor, Sum) + C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(update_interval, Sum)*C(variance_factor, Sum)*C(dataset, Sum)", data = relevant_data).fit()

    aov_table_r_rh = sm.stats.anova_lm(model_r_rh, typ = 3)

    model2_r_rh = ols("diff_RH_R ~ C(demand_capacity_ratio, Sum) + C(update_interval, Sum) + C(variance_factor, Sum) + C(dataset, Sum) + C(demand_capacity_ratio, Sum)*C(dataset, Sum) + C(update_interval, Sum)*C(dataset, Sum)", data = relevant_data).fit()

    aov_table2_r_rh = sm.stats.anova_lm(model2_r_rh, typ = 3)

    print("all interaction effects for RH-ICEP vs. benchmark")
    print(aov_table)
    # print(model.params)

    print("main effects and significant interaction effects for RH-ICEP vs benchmark")
    print(aov_table2)
    print(model2.params)

    print("all interaction effects for R-ICEP vs. benchmark")
    print(aov_table_r)
    # print(model_r.params)

    print("main effects and significant interaction effects for R-ICEP vs benchmark")
    print(aov_table2_r)
    print(model2_r.params)

    print("all interaction effects for RH-ICEP vs. R-ICEP")
    print(aov_table_r_rh)
    # print(model_r_rh.params)

    print("main effects and significant interaction effects for RH-ICEP vs. R-ICEP")
    print(aov_table2_r_rh)
    print(model2_r_rh.params)

    # interaction_groups = "DCR_" + relevant_data.demand_capacity_ratio.astype(str) + " & " + "UI_" + relevant_data.update_interval.astype(str) + " & " + "VF_" + relevant_data.variance_factor.astype(str) + " & " + "DS_" + relevant_data.dataset.astype(str)

    # comp = mc.MultiComparison(relevant_data["diff_RH_R"], interaction_groups)
    # post_hoc_res = comp.tukeyhsd()
    # print("Tukey Honestly Significant Difference (HSD) Test on RH vs R:")
    # print(post_hoc_res.summary())

    fig = plt.figure(figsize= (10, 10))
    ax = fig.add_subplot(111)

    normality_plot, stat = stats.probplot(model2_r_rh.resid, plot= plt, rvalue= True)
    ax.set_title("Probability plot of model residual's", fontsize= 20)
    ax.set

    plt.show()

    return(-1)


if __name__ == "__main__":
    main()