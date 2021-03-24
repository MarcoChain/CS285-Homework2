import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y
	
if __name__ == '__main__':
    import glob

    logdirs = ['data/q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_10-13-12/events*',
    		"data/q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_10-28-42/events*",
    		"data/q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_15-17-56/events*",
    		"data/q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_11-11-50/events*",
    		"data/q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_12-58-57/events*",
    		"data/q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_10-39-43/events*",
    		"data/q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_14-26-22/events*",
    		"data/q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_13-29-02/events*",
    		"data/q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_24-03-2021_15-29-56/events*"]
    labels = ["b10000-lr0.01", "b10000-lr0.02", "b10000-lr0.005", "b30000-lr0.01", "b30000-lr0.005", "b30000-lr0.02", "b50000-lr0.01", "b50000-lr0.005", "b50000-lr0.02"]
    plt.figure(figsize = (15,15))
    for label, logdir in zip(labels, logdirs):
	    eventfile = glob.glob(logdir)[0]
	    X, Y = get_section_results(eventfile)
	    res = np.zeros([100])
	    for i, (x, y) in enumerate(zip(X, Y)):
	    	res[i] = y
	    plt.plot(res, label = label)
    plt.ylabel("Eval_average_return")
    plt.xlabel("Iteration number")
    plt.legend()
    plt.savefig("ex_7_4.png")
    
    
