import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
df = df.drop('id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,cardio', 1)
#print(df.head())

# Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height']/100)**2)
df.loc[df.overweight <= 25, 'overweight'] = 0 #any column value over <= 25 is converted to 0 in the overweight column
df.loc[df.overweight > 25, 'overweight'] = 1 # 1 means someone's overweight for their height
df.overweight = df.overweight.astype('int32') #converts column from float to int
# print(df.head(10))

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.loc[df.cholesterol == 1, 'cholesterol'] = 0
df.loc[df.cholesterol > 1, 'cholesterol'] = 1
df.loc[df.gluc == 1, 'gluc'] = 0
df.loc[df.gluc > 1, 'gluc'] = 1
# print(df.head(10))

# a = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke', 'alco', 'active', 'overweight'])
# print(a)
# a['total'] = 0 #need to add this column for the below groupby to work
# # print(a)
# b = a.groupby(['cardio', 'variable', 'value'], as_index=False).count() #as_index = False -- is SQL styled grouped output
# print(b)
#
# fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=b, kind='bar').fig
# fig.savefig('catplot.png')

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
    df_cat['total'] = 0
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

#
# pd.set_option('display.max_columns', None)
df = df[
(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))
]
x = df.corr()
y = x.mask(np.triu(np.ones(x.shape)).astype(bool))
print(y)

fig, ax = plt.subplots(figsize=(11,9))

sns.heatmap(y, annot=True, fmt='.1f', vmin= -0.16, vmax= 0.32, cbar_kws={'ticks': [-0.08, 0.00, 0.08, 0.16, 0.24]}) #fmt is to 1 decimal place
fig.savefig('heatmap.png')

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.copy()

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = corr.mask(np.triu(np.ones(corr.shape)).astype(bool)) # to eliminate duplicate correlation values

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11,9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(mask, annot=True, fmt='.1f', vmin= -0.16, vmax= 0.32, cbar_kws={'ticks': [-0.08, 0.00, 0.08, 0.16, 0.24]})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
