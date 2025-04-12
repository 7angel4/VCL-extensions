from constants import *
import h5py

def get_results_dir(task_type):
    results_dir = f"{RESULTS_DIR}/{task_type}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def plot_filepath(task_type, prefix=''):
    return f"{get_results_dir(task_type)}/{prefix}.png"

def data_filepath(task_type, prefix=''):
    return f"{get_results_dir(task_type)}/{prefix}_data.h5"


def export_results(data, task_type, datatype='mean'):
    """ Exports the dictionary `data` to an HDF5 file. """
    with h5py.File(data_filepath(task_type, prefix=datatype), 'w') as f:
        # Iterate over the dictionary and save each array as a dataset
        for key, array in data.items():
            f.create_dataset(key, data=array)

def import_results(task_type, datatype='mean'):
    """ Imports the data dictionary from an HDF5 file. """
    imported_data = {}
    with h5py.File(data_filepath(task_type, prefix=datatype), 'r') as f:
        # Iterate through the keys in the file and load the datasets
        for key in f.keys():
            imported_data[key] = f[key][:]
    return imported_data

def aggr_results(model_to_results, mname_filter=None, aggr='final'): 
    """
    Mean results for the given list of models, using the provided aggregation.
    `aggr` is one of: 
    - 'all' (average across the lifetime of all tasks), 
    - 'final' (average across the final result of all tasks)
    """
    if model_to_results is None:
        return None
    if mname_filter is None:
        mname_filter = (lambda _ : True)
    aggregate = (lambda res: res[-1]) if aggr == 'final' else \
                (lambda res: np.mean(res, axis=0, where=(res > 1e-6)))
    return {m:aggregate(res) for m,res in model_to_results.items() if mname_filter(m)}

def mean_aggr_results(model_to_results, mname_filter=None, aggr='final', ret_std=False): 
    aggregated_results = aggr_results(model_to_results, mname_filter=mname_filter, aggr=aggr)
    means = { m: np.mean(res) for m,res in aggregated_results.items() } 
    stds = { m: np.std(res) for m,res in aggregated_results.items() }
    return (means, stds) if ret_std else means

# plot styling based on model name
def get_markerstyle(m):
    if m.startswith('Gaussian'):
        return 'o'
    elif m.startswith('Exp'):
        return 's'
    else:
        return 'v'

def get_linestyle(m):
    if m == VANILLA_MODEL:
        return 'dotted'
    elif 'None' in m:
        return 'dashed'
    elif '50' in m:
        return 'dashdot'
    elif '100' in m:
        return (0,(1,1))
    return 'solid'

def get_colour(m):
    tab10 = plt.get_cmap('tab10')
    if 'Gaussian' in m:
        return tab10(0)
    elif 'Exp' in m:
        return tab10(1)
    return tab10(2)


def plot_results(config, results, results_std=None, export=True, title='Mean', legend=True,
                 loc='upper left', bbox_to_anchor=(1,1), figsize=(18, 8)):
    """Plot comparison of results with optional error bars."""
    MARKER_SIZE = 9
    CAPSIZE = 4
    ELINEWIDTH = 1
    fig, ax = plt.subplots(figsize=figsize)
    x_ticks = range(1, len(config.tasks)+1)
    model_names = sorted(results.keys(), reverse=True)
    
    for model in model_names:
        x_vals = np.arange(len(results[model])) + 1
        # If std info is provided, add error bars
        if results_std is not None:
            ax.errorbar(x_vals, results[model], yerr=results_std[model], 
                         label=model, marker=get_markerstyle(model), 
                         linestyle=get_linestyle(model), color=get_colour(model),
                         markersize=MARKER_SIZE, capsize=CAPSIZE, elinewidth=ELINEWIDTH)
        else:
            ax.plot(x_vals, results[model], 
                    label=model, marker=get_markerstyle(model), linestyle=get_linestyle(model),
                    markersize=MARKER_SIZE, color=get_colour(model))
    
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f"{title} {config.eval_metric}", fontsize=18)
    ax.set_xlabel("# tasks", fontsize=15)
    ax.set_ylabel(config.eval_metric, fontsize=15)
    if legend:
        ax.legend(fontsize=18, loc=loc, bbox_to_anchor=bbox_to_anchor)

    plt.tight_layout()
    if export:
        plt.savefig(plot_filepath(config.task_type, prefix=title.lower()),
                   bbox_inches='tight', pad_inches=.03)
    
    plt.show()


def plot_mean_results(config, results, results_std=None, export=True, mname_filter=None, legend=True,
                      loc='upper left', bbox_to_anchor=(1,1), figsize=(18, 8)):    
    mean_results = aggr_results(results, mname_filter=mname_filter, aggr='all')
    mean_results_std = aggr_results(results_std, mname_filter=mname_filter, aggr='all')
    if mname_filter is not None:
        mean_results = {m: res for m, res in mean_results.items() if mname_filter(m)}
        if mean_results_std is not None:
            mean_results_std = {m: res for m, res in mean_results_std.items() if mname_filter(m)}
    return plot_results(config, mean_results, results_std=mean_results_std, export=export, 
                        title=f"mean_{config.eval_metric}", legend=legend, loc=loc, 
                        bbox_to_anchor=bbox_to_anchor, figsize=figsize)

def plot_final_results(config, results, results_std=None, export=True, mname_filter=None, legend=True,
                       loc='upper left', bbox_to_anchor=(1,1), figsize=(18, 8)):
    final_results = aggr_results(results, mname_filter=mname_filter, aggr='final')
    final_results_std = aggr_results(results_std, mname_filter=mname_filter, aggr='final')
    if mname_filter is not None:
        final_results = {m: res for m, res in final_results.items() if mname_filter(m)}
        if final_results_std is not None:
            final_results_std = {m: res for m, res in final_results_std.items() if mname_filter(m)}
    return plot_results(config, final_results, results_std=final_results_std, export=export, 
                        title=f"final_{config.eval_metric}", legend=legend, loc=loc, 
                        bbox_to_anchor=bbox_to_anchor, figsize=figsize)

def print_if(msg, print_progress):
    if print_progress:
        print(msg)
