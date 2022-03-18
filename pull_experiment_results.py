import pandas as pd
import argparse
import os
import numpy as np

def main():

    """A method to gather experiment outputs and compile them into a compact file."""

    parser = argparse.ArgumentParser(description='key parameters')
    parser.add_argument('-d', '--experiment_data', type=str)
    parser.add_argument('-e', '--experiment_target_folder', type=str)
    parser.add_argument('-r', '--run_mode', type=str)

    args = parser.parse_args()

    instance = str(args.experiment_data)
    experiment_target_folder = str(args.experiment_target_folder)
    run_mode = str(args.run_mode)

    if not os.path.exists(os.path.join(os.getcwd(), experiment_target_folder)):
        os.makedirs(os.path.join(os.getcwd(), experiment_target_folder))

    experiment_results = pd.DataFrame()

    print("Number of experiments:", len(os.listdir(os.path.join(os.getcwd(), instance))))

    for instance_folder in os.listdir(os.path.join(os.getcwd(), instance)):

        iteration = instance_folder.split('_')[-3]

        resources = pd.read_csv(os.path.join(os.getcwd(), instance, instance_folder, 'input', 'vessels.csv'))

        if not any(e.endswith('GUROBI_iteration_' + iteration + '.csv') for e in os.listdir(os.path.join(os.getcwd(), instance, instance_folder, 'Solutions'))):
            print("Files missing in:", instance_folder)

        for solutions in os.listdir(os.path.join(os.getcwd(), instance, instance_folder, 'Solutions')):

            if 'route_plan' in solutions:

                if 'iteration_' + iteration in solutions:
                    true_result = pd.read_csv(os.path.join(os.getcwd(), instance, instance_folder, 'Solutions', solutions))

                    # performance parameters
                    r_true = true_result['load_end_time'].max()
                    r_as_per_alg = r_true
                    time_utilizations_true = []
                    capacity_utilized_true = []
                    for v in np.unique(true_result['resource_id']):
                        sub_frame_true = true_result[true_result['resource_id'] == v]
                        time_utilized_true = 0
                        for i in range(len(sub_frame_true)):
                            if sub_frame_true['evacuees'].iloc[i] > 0:
                                time_utilized_true += sub_frame_true['load_end_time'].iloc[i] - sub_frame_true['route_start_time'].iloc[i]
                            if 'Evac' in sub_frame_true['evacuated_location'].iloc[i]:
                                if sub_frame_true['evacuees'].iloc[i] == 0:
                                    capacity_utilized_true.append(0)
                                else:
                                    capacity_utilized_true.append(sub_frame_true['evacuees'].iloc[i]/float(resources['max_cap'][resources['Vessel_name'] == v]))
                        time_utilization = time_utilized_true/max(sub_frame_true['load_end_time'])
                        time_utilizations_true.append(time_utilization)
                    avg_utilization_true = np.mean(time_utilizations_true)
                    avg_utilization_as_per_alg = avg_utilization_true
                    avg_cap_util_true = np.mean(capacity_utilized_true)
                    avg_cap_util_as_per_alg = avg_cap_util_true

                    # independent variables
                    records = instance_folder.split('_')
                    demand_capacity_ratio = float(records[5])
                    variance_factor = float(records[8])
                    update_interval = float(records[11])
                    number_updates = float(records[14])
                    seed = float(records[-1])
                    if run_mode == 'robust':
                        gamma_setting = float(solutions.split('_')[5])
                    else:
                        gamma_setting = None
                    dataset = records[3]

                    if run_mode == 'robust':
                        model_type = 'R-ICEP'
                    elif run_mode == 'rolling-horizon':
                        model_type = 'RH-ICEP'
                    elif run_mode == 'deterministic':
                        model_type = 'D-ICEP'

                    # record to dataframe
                    experiment_results = experiment_results.append({'dataset': dataset,
                                                                    'model': model_type,
                                                                    'evac_time_per_alg': r_as_per_alg,
                                                                    'evac_time_true': r_true,
                                                                    'avg_util_per_alg': avg_utilization_as_per_alg,
                                                                    'avg_util_true': avg_utilization_true,
                                                                    'avg_util_cap_per_alg': avg_cap_util_as_per_alg,
                                                                    'avg_util_cap_true': avg_cap_util_true,
                                                                    'demand_capacity_ratio': demand_capacity_ratio,
                                                                    'variance_factor': variance_factor,
                                                                    'update_interval': update_interval,
                                                                    'number_updates': number_updates,
                                                                    'gamma_setting': gamma_setting,
                                                                    'random_seed': seed,
                                                                    'id': str(model_type) + '_' + str(variance_factor) + '_' +
                                                                          str(update_interval) + '_' + str(number_updates) +
                                                                          '_' + str(seed) + '_' + str(gamma_setting)},
                                                                   ignore_index = True)
                else:
                    pass

                # except:
                #     print("Results not available yet for ", instance_folder, solutions)

            else:
                pass

    experiment_results.to_csv(os.path.join(os.getcwd(),
                                           experiment_target_folder,
                                           'result_summary_' + str(run_mode) + '.csv'),
                              index = False)

    return(-1)

if __name__ == "__main__":
    main()