import pandas as pd
import researchpy as rp
import argparse
import os
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():

    """A method to conduct an n-way ANOVA on an experiment"""

    parser = argparse.ArgumentParser(description='key parameters')
    parser.add_argument('-e', '--experiment', type=str)
    parser.add_argument('-a', '--argument', type = str)

    args = parser.parse_args()

    experiment_source_folder = str(args.experiment)
    argument_of_interest = str(args.argument)

    results = 'experiment_results.csv'

    # load data
    df = pd.read_csv(os.path.join(os.getcwd(), '..', experiment_source_folder, results))

    print(rp.summary_cat(df[['demand_capacity_ratio', 'variance_factor','update_interval','number_updates','gamma_setting']]))

    # copy data frame and modify values
    # df = data.copy()
    # df['stops'][df['stops'] == 'many'] = 15
    # df['stops'][df['stops'] == 'few'] = 5
    # df['area'][df['area'] == 'large'] = 4
    # df['area'][df['area'] == 'small'] = 1
    # df['std_cruise'][df['std_cruise'] == 'two'] = 2
    # df['std_cruise'][df['std_cruise'] == 'half'] = 0.5
    # df['std_matrix'][df['std_matrix'] == 'high'] = 1.30
    # df['std_matrix'][df['std_matrix'] == 'low'] = 0.35
    # print(df['trip_savings_per_stop'])

    # add constant for log transformation
    # print(min(df['trip_savings_per_stop']))
    # min_param = -min(df['trip_savings_per_stop']) + 0.001
    # df['trip_savings_per_stop_constant'] = df['trip_savings_per_stop'] + min_param
    # df['square_trip_savings_per_stop'] = df['trip_savings_per_stop']**2
    #
    # print(df['square_trip_savings_per_stop'])

    plot correlations

    sns.pairplot(df[['evac_time_true', 'model_type','demand_capacity_ratio', 'variance_factor','update_interval','number_updates','gamma_setting']])
    plt.show()

    # with all interaction effects

    df["Response"] = df[argument_of_interest]

    model = ols("Response ~ C(model_type, Sum) + C(demand_capacity_ratio, Sum) + C(variance_factor, Sum) + C(update_interval, Sum) + C(number_updates, Sum) + C(gamma_setting, Sum) + C(model_type, Sum)*C(demand_capacity_ratio, Sum)*C(variance_factor, Sum)*C(update_interval, Sum)*C(number_updates, Sum)*C(gamma_setting, Sum)",
                data = df).fit()

    aov_table = sm.stats.anova_lm(model, typ=3)
    print("All interaction effects")
    print(aov_table)
    #
    # # with only three way interaction effects
    #
    # model2 = ols("Response ~ C(area, Sum) + C(stops, Sum) + C(std_cruise, Sum) + C(std_matrix, Sum) + C(area, Sum)*C(stops, Sum)*C(std_cruise, Sum) + C(area, Sum)*C(stops, Sum)*C(std_matrix, Sum) + C(area, Sum)*C(std_cruise, Sum)*C(std_matrix, Sum) + C(stops, Sum)*C(std_cruise, Sum)*C(std_matrix, Sum)",
    #              data = df).fit()
    #
    # aov_table2 = sm.stats.anova_lm(model2, typ=3)
    # print("Three way interaction effects only")
    # print(aov_table2)
    # #
    # # # with only two way interaction effects
    #
    # model3 = ols("Response ~ C(area, Sum) + C(stops, Sum) + C(std_cruise, Sum) + C(std_matrix, Sum) + C(area, Sum)*C(stops, Sum) + C(area, Sum)*C(std_matrix, Sum) + C(area, Sum)*C(std_cruise, Sum) + C(stops, Sum)*C(std_cruise, Sum) + C(stops, Sum)*C(std_matrix, Sum) + C(std_cruise, Sum)*C(std_matrix, Sum)",
    #              data = df).fit()
    #
    # aov_table3 = sm.stats.anova_lm(model3, typ=3)
    # print("Two way interaction effects only")
    # print(aov_table3)
    # #
    # # with only significant two way interaction effects
    #
    # model4 = ols("Response ~ C(area, Sum) + C(stops, Sum) + C(std_cruise, Sum) + C(std_matrix, Sum) + C(stops, Sum)*C(std_matrix, Sum)",
    #              data = df).fit()
    #
    # aov_table4 = sm.stats.anova_lm(model4, typ=3)
    # print("Significant two way interaction effects only")
    # print(aov_table4)
    # #
    # interaction_groups = "Area_" + df.area.astype(str) + " & " + "Stops_" + df.stops.astype(str) + " & " + "Std_Cruise_" + df.std_cruise.astype(str) + " & " + "Std_Matrix_" + df.std_matrix.astype(str)
    #
    # comp = mc.MultiComparison(df["Response"], interaction_groups)
    # post_hoc_res = comp.tukeyhsd()
    # print("Tukey Honestly Significant Difference (HSD) Test:")
    # print(post_hoc_res.summary())
    #
    # # post_hoc_res.plot_simultaneous(ylabel= "Drug Dose", xlabel= "Score Difference")
    # #
    # # print("Shapiro Test: W statistic, p-value")
    # # print(stats.shapiro(model4.resid))
    #
    # fig = plt.figure(figsize= (10, 10))
    # ax = fig.add_subplot(111)
    #
    # normality_plot, stat = stats.probplot(model4.resid, plot= plt, rvalue= True)
    # ax.set_title("Probability plot of model residual's", fontsize= 20)
    # ax.set
    #
    # plt.show()
    #
    # # Bonferroni correction
    # # stops to response
    # print("Bonferroni on stops and response:")
    # comp = mc.MultiComparison(df['Response'], df['stops'])
    # tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
    # print(tbl)
    #
    # # area to response
    # print("Bonferroni on area and response:")
    # comp = mc.MultiComparison(df['Response'], df['area'])
    # tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
    # print(tbl)
    #
    # # std deviation of cruising time to cruise
    # print("Bonferroni on standard deviation of cruise time and response:")
    # comp = mc.MultiComparison(df['Response'], df['std_cruise'])
    # tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
    # print(tbl)
    #
    # # std deviation of travel time matrix to cruise
    # print("Bonferroni on standard deviation of travel time matrix and response:")
    # comp = mc.MultiComparison(df['Response'], df['std_matrix'])
    # tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
    # print(tbl)
    #
    # # non-parametric tests to confirm ANOVA
    # print(stats.kruskal(df['Response'][df['stops'] == 'many'], df['Response'][df['stops'] == 'few']))
    # print(stats.kruskal(df['Response'][df['area'] == 'large'], df['Response'][df['area'] == 'small']))
    # print(stats.kruskal(df['Response'][df['std_matrix'] == 'high'], df['Response'][df['std_matrix'] == 'low']))
    # print(stats.kruskal(df['Response'][df['std_cruise'] == 'two'], df['Response'][df['std_cruise'] == 'half']))
    #
    # print(stats.kruskal(df['Response'][(df['std_matrix'] == 'high') & (df['stops'] == 'many')], df['Response'][(df['std_matrix'] == 'low') & (df['stops'] == 'few')]))
    # print(stats.kruskal(df['Response'][(df['std_matrix'] == 'high') & (df['stops'] == 'few')], df['Response'][(df['std_matrix'] == 'low') & (df['stops'] == 'many')]))

    return(-1)


if __name__ == '__main__':
    main()



