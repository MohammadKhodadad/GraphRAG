import os
import json
import pandas as pd 
import matplotlib.pyplot as plt




PATH_TO_RESPONSES='/media/torontoai/GraphRAG/GraphRAG/data_loader/responses'
MAIN_COLS = ["is_correct", "context_used", "input_tokens", "output_tokens", "reasoning_tokens", "latency"]
addresses = os.listdir(PATH_TO_RESPONSES)
analysis_folder = 'analysis'
os.makedirs(analysis_folder, exist_ok=True)



# Function to plot and save a figure given a dataframe and metric name
def plot_metric(df, metric, title_suffix, filename):
    plt.figure(figsize=(8, 6))
    models = df['model_name'].unique()
    for model in models:
        # Filter data for this model
        model_df = df[df['model_name'] == model]
        # Sort by number of hops for clarity
        model_df = model_df.sort_values('n_hops')
        plt.plot(model_df['n_hops'], model_df[metric], marker='o', label=model)
    plt.xlabel('Number of Hops')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Number of Hops ({title_suffix})')
    plt.legend(title='Model')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, filename))
    plt.close()



results= {}
for address in addresses:
    full_address = os.path.join(PATH_TO_RESPONSES,address)
    with open(full_address, 'r') as f:
        data = json.load(f)
    results[address.replace('.json','')]=data

######################## Make it a dataframe ###############################
main_df = []
for key, value in results.items():
    sub_df = pd.DataFrame(value)
    sub_df['model_name']=key
    main_df.append(sub_df)
main_df = pd.concat(main_df,axis=0).reset_index(drop=True)
main_df.to_csv(os.path.join(analysis_folder, 'main_df.csv'), index=False)

main_df['n_hops'] = main_df['raw'].apply(lambda x: x['length'])
######################### General Performance ################################
general_df = main_df.groupby(['model_name', 'context_used'])[
    [col for col in MAIN_COLS if col not in ['model_name', 'context_used']]
].mean().reset_index()
print(general_df)
general_df.to_csv(os.path.join(analysis_folder, 'general_df.csv'), index=False)


######################### Number of hops #####################################
hops_df = main_df.groupby(['model_name', 'context_used','n_hops'])[
    [col for col in MAIN_COLS if col not in ['model_name', 'context_used']]
].mean().reset_index()
print(hops_df)
hops_df.to_csv(os.path.join(analysis_folder, 'hops_df.csv'), index=False)

df_with_context = hops_df[hops_df['context_used'] == True]
df_without_context = hops_df[hops_df['context_used'] == False]

# Plotting for accuracy
plot_metric(df_with_context, 'is_correct', 'With Context', 'accuracy.png')
plot_metric(df_without_context, 'is_correct', 'Without Context', 'accuracy.png')


# Plotting for output_tokens
plot_metric(df_with_context, 'output_tokens', 'With Context', 'output_tokens_with_context.png')
plot_metric(df_without_context, 'output_tokens', 'Without Context', 'output_tokens_without_context.png')

# Plotting for latency
plot_metric(df_with_context, 'latency', 'With Context', 'latency_with_context.png')
plot_metric(df_without_context, 'latency', 'Without Context', 'latency_without_context.png')
