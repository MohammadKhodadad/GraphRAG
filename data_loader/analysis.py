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

model_name_mapping = {
    # Provider.OPENAI
    "gpt-4o": "GPT-4O",
    "gpt-4o-mini": "GPT-4O Mini",
    "o1-mini": "O1 Mini",
    "o3-mini": "O3 Mini",
    
    # Provider.BEDROCK
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "Anthropic Claude 3.5",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Anthropic Claude 3.7",
    "us.meta.llama3-3-70b-instruct-v1:0": "Meta LLaMA3 70B Instruct",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Anthropic Claude 3.5 (Alt)",
    "mistral.mistral-large-2402-v1:0": "Mistral Large",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Anthropic Claude 3.7 Reasoning",
    "us.deepseek.r1-v1:0-reasoning": "Deepseek R1 Reasoning",
    
    # Provider.NVIDIA
    "deepseek-ai/deepseek-r1": "Deepseek R1"
}

def get_model_type(model_name):
    # Mapping for claude models (Anthropic Claude)
    if any(keyword in model_name.lower() for keyword in ['claude']):
        return 'claude'
    # Mapping for llama (Meta LLaMA3)
    elif 'llama' in model_name.lower():
        return 'llama'
    # Mapping for mistral models
    elif 'mistral' in model_name.lower():
        return 'mistral'
    # Mapping for deepseek models
    elif 'deepseek' in model_name.lower():
        return 'deepseek'
    # Mapping for openai models (if none of the above match)
    else:
        return 'openai'

# Function to plot and save a figure given a dataframe and metric name
def plot_metric(df, metric, title_suffix, filename):
    plt.figure(figsize=(16, 12))
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

    plt.figure(figsize=(20, 12))
    
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
    sub_df['model_name']=key.replace('responses_','').replace('openai_','').replace('bedrock_','')
    main_df.append(sub_df)
main_df = pd.concat(main_df,axis=0).reset_index(drop=True)
main_df['model_name'] = main_df['model_name'].replace(model_name_mapping)

main_df.to_csv(os.path.join(analysis_folder, 'main_df.csv'), index=False)

main_df['n_hops'] = main_df['raw'].apply(lambda x: x['length'])
######################### General Performance ################################
general_df = main_df.groupby(['model_name', 'context_used'])[
    [col for col in MAIN_COLS if col not in ['model_name', 'context_used']]
].mean().reset_index()
# print(general_df)
general_df.to_csv(os.path.join(analysis_folder, 'general_df.csv'), index=False)


general_df['context_used'] = general_df['context_used'].astype(bool)

# Create a pivot table from general_df with model_name as rows,
# context_used as columns, and accuracy ("is_correct") as values.
custom_order = {'claude': 0, 'llama': 1, 'mistral': 2, 'deepseek': 3, 'openai': 4}
pivot = general_df.pivot(index='model_name', columns='context_used', values='is_correct')
pivot['overall_accuracy'] = pivot.mean(axis=1)
pivot['model_type'] = pivot.index.map(get_model_type)
pivot['type_order'] = pivot['model_type'].map(custom_order)
pivot =  pivot.sort_values(by=['type_order', 'overall_accuracy'], ascending=[True, False])
# Debug: Print the pivot table's columns to see what we have
# print("Pivot columns:", pivot.columns.tolist())

# Reindex the pivot to ensure we only have two columns: False and True (if available)
pivot = pivot.reindex(columns=[False, True])

# Prepare the bar chart
models = pivot.index.tolist()
models= [item.replace('responses_','') for item in models]
sorted_models = [item.replace('responses_', '') for item in pivot.index.tolist()]

x = np.arange(len(models))  # label locations for models
bar_width = 0.2

plt.figure(figsize=(15, 9))
# Bars for "Without Context" (False)
bars1 = plt.bar(x - bar_width/2, pivot[False], bar_width, label='Without Context')
# Bars for "With Context" (True)
bars2 = plt.bar(x + bar_width/2, pivot[True], bar_width, label='With Context')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy by Model with/without Context')
plt.xticks(x, models, rotation=0)  # Set labels horizontal (0 degrees)
plt.legend(title='Context Used')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# Add numerical values above each bar
for bar in bars1:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f'{bar.get_height():.2f}',
        ha='center', va='bottom'
    )
for bar in bars2:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f'{bar.get_height():.2f}',
        ha='center', va='bottom'
    )

# Save the figure
plt.savefig(os.path.join(analysis_folder, 'accuracy_bar_models.png'))
plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt

# Assuming main_df, sorted_models, and analysis_folder are already defined.
# sorted_models is the list of model names in desired order.
# Also, main_df should have a boolean column "context_used".

# --- Boxplot for Tokens (split by context) ---

# Prepare data lists for tokens used by context
token_data_without = []  # for context_used == False
token_data_with = []     # for context_used == True

for model in sorted_models:
    df_model = main_df[main_df['model_name'] == model]
    tokens_without = (df_model[df_model['context_used'] == False]['input_tokens'] +
                      df_model[df_model['context_used'] == False]['output_tokens']).values
    tokens_with = (df_model[df_model['context_used'] == True]['input_tokens'] +
                   df_model[df_model['context_used'] == True]['output_tokens']).values
    token_data_without.append(tokens_without)
    token_data_with.append(tokens_with)

# Set positions for side-by-side boxplots for each model
num_models = len(sorted_models)
x = np.arange(num_models)
box_width = 0.35  # width for each box

positions_without = x - box_width/2
positions_with = x + box_width/2

plt.figure(figsize=(20, 12))

# Create boxplots
bp_without = plt.boxplot(token_data_without,
                         positions=positions_without,
                         widths=box_width,
                         patch_artist=True,
                         showfliers=False)
bp_with = plt.boxplot(token_data_with,
                      positions=positions_with,
                      widths=box_width,
                      patch_artist=True,
                      showfliers=False)

# Optional: Color customization
for box in bp_without['boxes']:
    box.set(facecolor='lightblue')
for box in bp_with['boxes']:
    box.set(facecolor='lightgreen')

plt.yscale('log')  # Log-normalize the y-axis
plt.xticks(x, sorted_models)
plt.xlabel('Model')
plt.ylabel('Number of Tokens (log scale)')
plt.title('Distribution of Number of Tokens Used by Model (split by Context)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Custom legend using the first box from each group
plt.legend([bp_without['boxes'][0], bp_with['boxes'][0]],
           ['Without Context', 'With Context'],
           loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'boxplot_tokens_by_context.png'))
plt.close()


# --- Boxplot for Latency (split by context) ---

latency_data_without = []  # for context_used == False
latency_data_with = []     # for context_used == True

for model in sorted_models:
    df_model = main_df[main_df['model_name'] == model]
    latency_without = df_model[df_model['context_used'] == False]['latency'].values
    latency_with = df_model[df_model['context_used'] == True]['latency'].values
    latency_data_without.append(latency_without)
    latency_data_with.append(latency_with)

plt.figure(figsize=(20, 12))

# Use the same positions as before
bp_latency_without = plt.boxplot(latency_data_without,
                                 positions=positions_without,
                                 widths=box_width,
                                 patch_artist=True,
                                 showfliers=False)
bp_latency_with = plt.boxplot(latency_data_with,
                              positions=positions_with,
                              widths=box_width,
                              patch_artist=True,
                              showfliers=False)

# Optional: Color customization
for box in bp_latency_without['boxes']:
    box.set(facecolor='lightcoral')
for box in bp_latency_with['boxes']:
    box.set(facecolor='lightgoldenrodyellow')

plt.xticks(x, sorted_models)
plt.xlabel('Model')
plt.ylabel('Latency (ms)')  # Adjust unit if needed
plt.title('Distribution of Latency by Model (split by Context)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.legend([bp_latency_without['boxes'][0], bp_latency_with['boxes'][0]],
           ['Without Context', 'With Context'],
           loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'boxplot_latency_by_context.png'))
plt.close()





# # Prepare data list for tokens used
# token_data = []
# for model in sorted_models:
#     df_model = main_df[main_df['model_name'] == model]
#     token_data.append((df_model['output_tokens'] + df_model['input_tokens']).values)

# plt.figure(figsize=(20, 12))
# plt.boxplot(token_data, labels=sorted_models, patch_artist=True, showfliers=False)
# plt.title('Distribution of Number of Tokens Used')
# plt.xlabel('Model')
# plt.ylabel('Number of Tokens')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig(os.path.join(analysis_folder, 'boxplot_tokens.png'))
# plt.close()

# # Prepare data list for latency
# latency_data = []
# for model in sorted_models:
#     df_model = main_df[main_df['model_name'] == model]
#     latency_data.append(df_model['latency'].values)

# plt.figure(figsize=(20, 12))
# plt.boxplot(latency_data, labels=sorted_models, patch_artist=True, showfliers=False)
# plt.title('Distribution of Latency')
# plt.xlabel('Model')
# plt.ylabel('Latency (ms)')  # Adjust unit if needed
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(analysis_folder, 'boxplot_latency.png'))
# plt.close()





######################### Number of hops #####################################
hops_df = main_df.groupby(['model_name', 'context_used','n_hops'])[
    [col for col in MAIN_COLS if col not in ['model_name', 'context_used']]
].mean().reset_index()
# print(hops_df)
hops_df.to_csv(os.path.join(analysis_folder, 'hops_df.csv'), index=False)

df_with_context = hops_df[hops_df['context_used'] == True]
df_without_context = hops_df[hops_df['context_used'] == False]

# # Plotting for accuracy
# plot_metric(df_with_context, 'is_correct', 'With Context', 'accuracy_with_context.png')
# plot_metric(df_without_context, 'is_correct', 'Without Context', 'accuracy_without_context.png')


# # Plotting for output_tokens
# plot_metric(df_with_context, 'output_tokens', 'With Context', 'output_tokens_with_context.png')
# plot_metric(df_without_context, 'output_tokens', 'Without Context', 'output_tokens_without_context.png')

# # Plotting for latency
# plot_metric(df_with_context, 'latency', 'With Context', 'latency_with_context.png')
# plot_metric(df_without_context, 'latency', 'Without Context', 'latency_without_context.png')


# Plot for responses with context
plot_grouped_by_n_hops(df_with_context, "With Context", "accuracy_by_model_nhops_with_context.png")

# Plot for responses without context
plot_grouped_by_n_hops(df_without_context, "Without Context", "accuracy_by_model_nhops_without_context.png")






################## Number of Tokens Distribution ##############################


# # Create a new column for total tokens used
# main_df['total_tokens'] = main_df['input_tokens'] + main_df['output_tokens'] 

# # Get the unique models
# models = main_df['model_name'].unique()

# # Prepare data: for each model, extract the distribution of total tokens
# data = [main_df.loc[main_df['model_name'] == model, 'total_tokens'] for model in models]

# # Create a box plot for the distribution of total tokens used per model, without outlier markers
# plt.figure(figsize=(20, 12))
# plt.boxplot(data, tick_labels=models, showfliers=False)
# plt.xlabel('Model')
# plt.ylabel('Total Tokens')
# plt.title('Distribution of Total Tokens Used per Model (Outliers Hidden)')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()

# # Save the figure in the analysis folder
# plt.savefig(os.path.join(analysis_folder, 'total_tokens_boxplot.png'))
# plt.close()






##################### TOKEN DISTRIBUTION PER MODEL #########################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Compute a new column for total tokens used if not already computed
if 'total_tokens' not in main_df.columns:
    main_df['total_tokens'] = main_df['input_tokens'] + main_df['output_tokens'] + main_df['reasoning_tokens']

# Get sorted unique models and n_hops values
models = sorted(main_df['model_name'].unique())
n_hops_vals = sorted(main_df['n_hops'].unique())
n_models = len(models)
n_groups = len(n_hops_vals)

# Settings for grouping: total width allocated for boxes for each model
group_width = 0.8  
box_width = group_width / n_groups

# Prepare data and positions for each box
box_data = []  # List to hold the data for each box
positions = [] # List to hold the x position for each box

for i, model in enumerate(models):
    for j, hop in enumerate(n_hops_vals):
        # Compute x position for this box for the current model
        pos = i - group_width/2 + (j + 0.5) * box_width
        positions.append(pos)
        # Filter data for this model and n_hops value
        data = main_df[(main_df['model_name'] == model) & (main_df['n_hops'] == hop)]['total_tokens']
        box_data.append(data)

# Create a color map: one color per n_hops category
colors = plt.cm.viridis(np.linspace(0, 1, n_groups))

# Create the grouped box plot with patch_artist to allow coloring
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(box_data, positions=positions, widths=box_width, patch_artist=True, showfliers=False)

# Color each box according to its n_hops category.
# The boxes are organized such that for each model, the boxes appear in order of n_hops.
for box_idx, patch in enumerate(bp['boxes']):
    # Determine which n_hops group this box belongs to:
    group_idx = box_idx % n_groups
    patch.set_facecolor(colors[group_idx])

# Set the x-ticks to the center of each model group
group_centers = np.arange(n_models)
ax.set_xticks(group_centers)
ax.set_xticklabels(models, rotation=0)  # Horizontal labels
ax.set_xlabel('Model')
ax.set_ylabel('Total Tokens')
ax.set_title('Distribution of Total Tokens by Model and n_hops')

# Create legend handles for each n_hops category
legend_handles = [mpatches.Patch(color=colors[j], label=f'n_hops = {n_hops_vals[j]}') for j in range(n_groups)]
ax.legend(handles=legend_handles, title='n_hops')

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'total_tokens_grouped_by_n_hops.png'))
plt.close()



# Compute a new column for total tokens used if not already computed
if 'total_tokens' not in main_df.columns:
    main_df['total_tokens'] = main_df['input_tokens'] + main_df['output_tokens'] + main_df['reasoning_tokens']

# Get sorted unique models
models = sorted(main_df['model_name'].unique())
n_models = len(models)

# Create a figure with a subplot for each model
fig, axes = plt.subplots(1, n_models, figsize=(10*n_models,6 ), sharey=True)

# If there's only one model, ensure axes is iterable
if n_models == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    # Filter data for the current model
    model_data = main_df[main_df['model_name'] == model]
    
    # Get sorted unique n_hops values for this model
    n_hops_vals = sorted(model_data['n_hops'].unique())
    
    # Prepare the box plot data: one distribution per n_hops value
    box_data = [model_data[model_data['n_hops'] == hop]['total_tokens'] for hop in n_hops_vals]
    
    # Create the box plot for this model
    ax.boxplot(box_data, tick_labels=n_hops_vals, showfliers=False)
    ax.set_xlabel('n_hops')
    ax.set_ylabel('Total Tokens')
    ax.set_title(f'{model}: Distribution of Total Tokens by n_hops')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'total_tokens_multiplot.png'))
plt.close()



############################ Latency Distribution #################



import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Get sorted unique models and n_hops values
models = sorted(main_df['model_name'].unique())
n_hops_vals = sorted(main_df['n_hops'].unique())
n_models = len(models)
n_groups = len(n_hops_vals)

# Settings for grouping: total width allocated for boxes for each model
group_width = 0.8  
box_width = group_width / n_groups

# Prepare data and positions for each box
box_data = []  # List to hold the data for each box
positions = [] # List to hold the x position for each box

for i, model in enumerate(models):
    for j, hop in enumerate(n_hops_vals):
        # Compute x position for this box for the current model
        pos = i - group_width/2 + (j + 0.5) * box_width
        positions.append(pos)
        # Filter data for this model and n_hops value
        data = main_df[(main_df['model_name'] == model) & (main_df['n_hops'] == hop)]['latency']
        box_data.append(data)

# Create a color map: one color per n_hops category
colors = plt.cm.viridis(np.linspace(0, 1, n_groups))

# Create the grouped box plot with patch_artist to allow coloring
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(box_data, positions=positions, widths=box_width, patch_artist=True, showfliers=False)

# Color each box according to its n_hops category.
# The boxes are organized such that for each model, the boxes appear in order of n_hops.
for box_idx, patch in enumerate(bp['boxes']):
    # Determine which n_hops group this box belongs to:
    group_idx = box_idx % n_groups
    patch.set_facecolor(colors[group_idx])

# Set the x-ticks to the center of each model group
group_centers = np.arange(n_models)
ax.set_xticks(group_centers)
ax.set_xticklabels(models, rotation=0)  # Horizontal labels
ax.set_xlabel('Model')
ax.set_ylabel('Latency')
ax.set_title('Distribution of Latency by Model and n_hops')

# Create legend handles for each n_hops category
legend_handles = [mpatches.Patch(color=colors[j], label=f'n_hops = {n_hops_vals[j]}') for j in range(n_groups)]
ax.legend(handles=legend_handles, title='n_hops')

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'latency_grouped_by_n_hops.png'))
plt.close()


import os
import matplotlib.pyplot as plt



# Get sorted unique models
models = sorted(main_df['model_name'].unique())
n_models = len(models)

# Create a figure with a subplot for each model
fig, axes = plt.subplots(1, n_models, figsize=(10*n_models,6 ), sharey=True)

# If there's only one model, ensure axes is iterable
if n_models == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    # Filter data for the current model
    model_data = main_df[main_df['model_name'] == model]
    
    # Get sorted unique n_hops values for this model
    n_hops_vals = sorted(model_data['n_hops'].unique())
    
    # Prepare the box plot data: one distribution per n_hops value
    box_data = [model_data[model_data['n_hops'] == hop]['latency'] for hop in n_hops_vals]
    
    # Create the box plot for this model
    ax.boxplot(box_data, tick_labels=n_hops_vals, showfliers=False)
    ax.set_xlabel('n_hops')
    ax.set_ylabel('Latency')
    ax.set_title(f'{model}: Distribution of Latency by n_hops')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'latency_multiplot.png'))
plt.close()

