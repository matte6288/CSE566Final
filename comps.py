# Import necessary libraries
import csv
import pandas as pd
import argparse
import statsmodels.api as sm
import numpy as np



# Setup argument parser for command line inputs
def setup_parser():
    parser = argparse.ArgumentParser(description="Parameters needed on the researcher side")
    parser.add_argument('-i', '--dataset_file', type=str, help='Path to the dataset file', required=True)
    parser.add_argument('-e', '--epsilon', type=int, help='Privacy parameter', required=True)
    parser.add_argument('-l', '--l', type=int, help='Number of SNPs for GWAS statistics', required=True)
    parser.add_argument('-c', '--D_case_control_IDs_file', type=str, help='Path to the csv with case and control IDs', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to save output files', required=True)
    return parser.parse_args()

# Read user IDs from file
def get_user_IDs(file_path):
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        case_IDs = next(reader, [])
        control_IDs = next(reader, [])
    return case_IDs, control_IDs

# Compute statistical metrics
def compute_ps(case_minor, control_minor, case_major, control_major):
    contingency_table = np.array([[case_minor, control_minor], [case_major, control_major]])
    table = sm.stats.Table2x2(contingency_table)
    results = {
        'p_val': "%.8f" % table.log_oddsratio_pvalue()
    }
    return pd.Series(results)


# Perform GWAS analysis
def perform_GWAS(dataframe, case_IDs, control_IDs, dp=False, m=100, e=1.0):
    # Copy the dataframe to avoid modifying the original dataset
    data_copy = dataframe.copy()

    # Assuming 'case_IDs' and 'control_IDs' are lists of column names for case and control data
    case_data = data_copy[case_IDs].apply(pd.to_numeric, errors='coerce')
    control_data = data_copy[control_IDs].apply(pd.to_numeric, errors='coerce')

    # Initialize new columns for allele counts
    for allele in [0, 1, 2]:  # Assuming alleles are encoded as 0, 1, 2
        data_copy[f'case_{allele}'] = case_data.apply(lambda x: (x == allele).sum(), axis=1)
        data_copy[f'control_{allele}'] = control_data.apply(lambda x: (x == allele).sum(), axis=1)

    # Compute statistics using allele counts
    data_GWAS = data_copy.apply(
    lambda row: compute_ps(
        row['case_1'] + row['case_2'],  # Minor case count
        row['control_1'] + row['control_2'],  # Minor control count
        row['case_0'],  # Major case count
        row['control_0']),  # Major control count
    axis=1
    )

     # Calculate MAFs
    data_copy['case_minor_counts'] = data_copy['case_1'] + 2 * data_copy['case_2']
    data_copy['control_minor_counts'] = data_copy['control_1'] + 2 * data_copy['control_2']
    data_copy['case_total'] = 2 * (data_copy['case_0'] + data_copy['case_1'] + data_copy['case_2'])
    data_copy['control_total'] = 2 * (data_copy['control_0'] + data_copy['control_1'] + data_copy['control_2'])
    data_copy['case_MAF'] = data_copy['case_minor_counts'] / data_copy['case_total']
    data_copy['control_MAF'] = data_copy['control_minor_counts'] / data_copy['control_total']

    final_results = pd.concat([data_GWAS, data_copy[['case_MAF', 'control_MAF']]], axis=1)
    dp_final_results = None

    if dp:
        number_of_IDs = len(case_IDs) + len(control_IDs)
        number_of_SNPs = data_copy.shape[0]
        
        # Calculate the noise scale
        scale_p_val = ((4 * m) / e) * np.exp(-2/3)
        scale_MAF = (2 * number_of_SNPs) / (number_of_IDs * e)
        
        # Ensure p_val is numeric
        data_GWAS['p_val'] = pd.to_numeric(data_GWAS['p_val'], errors='coerce')
        
        # Generate and add Laplace noise to each p_val entry
        data_GWAS['p_val'] = data_GWAS['p_val'].apply(lambda x: x + np.random.laplace(loc=0, scale=scale_p_val))
        
        # Generate and add Laplace noise to each case_MAF and control_MAF entry
        data_copy['case_MAF'] = data_copy['case_MAF'].apply(lambda x: x + np.random.laplace(loc=0, scale=scale_MAF))
        data_copy['control_MAF'] = data_copy['control_MAF'].apply(lambda x: x + np.random.laplace(loc=0, scale=scale_MAF))

        data_copy['case_MAF'] = np.clip(data_copy['case_MAF'], 0, 1).round(6)
        data_copy['control_MAF'] = np.clip(data_copy['control_MAF'], 0, 1).round(6)

    dp_final_results = pd.concat([data_GWAS, data_copy[['case_MAF', 'control_MAF']]], axis=1)

    return final_results, dp_final_results

# Main execution block
if __name__ == "__main__":
    args = setup_parser()

    # Load and prepare data
    case_IDs, control_IDs = get_user_IDs(args.D_case_control_IDs_file)
    num_parts = len(case_IDs)+len(control_IDs)
    dataset = pd.read_csv(args.dataset_file, sep=',', index_col=0)
    print("Dataset loaded successfully.")

    # Perform analysis
    gwas_results, dp_gwas_results = perform_GWAS(dataset, case_IDs, control_IDs, dp=True, e=args.epsilon, m=args.l)
    print("GWAS analysis completed.")

    # Sort and save results
    #sorted_results = gwas_results.sort_values(by='p_val', ascending=True)
    gwas_results.head(args.l).to_csv(f"{args.output_dir}D_{args.l}len_GWAS.csv")
    print(f"GWAS results for the top {args.l} SNPs have been saved.")

    scale_p_val = ((4 * args.l) / args.epsilon) * np.exp(-2/3)
    dp_gwas_results['p_val'] = dp_gwas_results['p_val'].apply(lambda x: x + np.random.laplace(loc=0, scale=scale_p_val))
    dp_gwas_results['p_val'] = np.clip(dp_gwas_results['p_val'], 0, 1).round(6)
    dp_gwas_results.head(args.l).to_csv(f"{args.output_dir}D_{args.l}len_DP_GWAS.csv")
    print(f"DP GWAS results for the top {args.l} SNPs have been saved.")
