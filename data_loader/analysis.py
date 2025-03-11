import os
import json
import pandas as pd 
import numpy as np
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


def plot_grouped_by_n_hops(df, context_label, filename):
    # Create a pivot table with model_name as index and n_hops as columns (values = accuracy)
    pivot = df.pivot(index='model_name', columns='n_hops', values='is_correct')
    
    # List of models and n_hops values (sorted for consistency)
    models = pivot.index.tolist()
    n_hops_vals = sorted(pivot.columns.tolist())
    
    x = np.arange(len(models))  # positions for the models on the x-axis
    num_bars = len(n_hops_vals)
    # Width for each bar in a group (use 80% of the space per group)
    bar_width = 0.8 / num_bars

    plt.figure(figsize=(10, 6))
    
    # Plot a bar for each n_hops value for every model
    for i, hop in enumerate(n_hops_vals):
        # Get the accuracies for this hop value across models; pivot returns NaN if missing.
        accuracies = pivot[hop].values
        # Calculate the offset for this set of bars
        offset = (i - num_bars/2) * bar_width + bar_width/2
        plt.bar(x + offset, accuracies, width=bar_width, label=f'n_hops = {hop}')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy by Model Divided by n_hops ({context_label})')
    plt.xticks(x, models, rotation=45)
    plt.legend(title='n_hops')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure to the analysis folder
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


general_df['context_used'] = general_df['context_used'].astype(bool)

# Create a pivot table from general_df with model_name as rows,
# context_used as columns, and accuracy ("is_correct") as values.
pivot = general_df.pivot(index='model_name', columns='context_used', values='is_correct')

# Debug: Print the pivot table's columns to see what we have
print("Pivot columns:", pivot.columns.tolist())

# Reindex the pivot to ensure we only have two columns: False and True (if available)
pivot = pivot.reindex(columns=[False, True])

# Prepare the bar chart
models = pivot.index.tolist()
x = np.arange(len(models))  # label locations for models
bar_width = 0.35

plt.figure(figsize=(10, 6))
# Bars for "Without Context" (False)
plt.bar(x - bar_width/2, pivot[False], bar_width, label='Without Context')
# Bars for "With Context" (True)
plt.bar(x + bar_width/2, pivot[True], bar_width, label='With Context')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy by Model with/without Context')
plt.xticks(x, models, rotation=45)
plt.legend(title='Context Used')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(analysis_folder, 'accuracy_bar_models.png'))
plt.close()

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


# Plot for responses with context
plot_grouped_by_n_hops(df_with_context, "With Context", "accuracy_by_model_nhops_with_context.png")

# Plot for responses without context
plot_grouped_by_n_hops(df_without_context, "Without Context", "accuracy_by_model_nhops_without_context.png")