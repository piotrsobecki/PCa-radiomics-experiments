import matplotlib.pyplot as plt

def plot_datalog(datalog_df,figsize=(8, 6), dpi=400):
    fig =plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(range(len(datalog_df)), datalog_df['Max Fitness'],linewidth=2)
    plt.figtext(.95, .60,'Min: %.3f | Max: %.3f ' %(datalog_df['Max Fitness'].min(),datalog_df['Max Fitness'].max()), rotation='vertical' )
    plt.title('AUC value of best individual in population')
    plt.ylabel('AUC')
    plt.xlabel('Generation')
    return fig

def plot_genlog(genlog_df,figsize=(8, 6), dpi=400):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title('Mean and standard error of AUC values of individuals in population')
    x = range(len(genlog_df))
    me = int(len(x)/100)
    plt.plot(x, genlog_df.max(axis=1), markevery=me, color='#008800', label='Max')
    plt.plot(x, genlog_df.min(axis=1), markevery=me, color='#880000', label='Min')
    ##plt.errorbar(x, genlog_df.mean(axis=1), yerr=genlog_df.sem(axis=1), color='b', markevery=me, errorevery=me,  fmt='-_', label='Mean')
    plt.fill_between(x, genlog_df.mean(axis=1) - genlog_df.sem(axis=1),   genlog_df.mean(axis=1) + genlog_df.sem(axis=1), color="#3F5D7D")
    plt.plot(x, genlog_df.mean(axis=1), markevery=me, color='white', label='Mean', lw=0.25)
    plt.ylabel('AUC')
    plt.figtext(.95, .60,'Min: %.3f | Max: %.3f ' %(genlog_df.min(axis=1).min(),genlog_df.max(axis=1).max()), rotation='vertical' )
    plt.xlabel('Generation')
    plt.legend(loc=2, prop={'size': 8})
    return fig