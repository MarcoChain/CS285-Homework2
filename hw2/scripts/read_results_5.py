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

def average(vect, window_size):
	res = np.zeros(len(vect))
	res[:window_size] = vect[:window_size]
	for i in range(window_size, len(vect)):
		res[i] = vect[i-window_size:i].mean()
	return res

if __name__ == '__main__':
    import glob
    logs = []
    #logs.append('data/q1_sb_no_rtg_dsa_CartPole-v0_19-03-2021_15-36-24/events*')
    #logs.append('data/q1_sb_rtg_dsa_CartPole-v0_19-03-2021_15-37-08/events*')
    #logs.append('data/q1_sb_rtg_na_CartPole-v0_19-03-2021_15-38-25/events*')
    titles = ["dsa_only", "dsa_rtg", "rtg_oly"]
    logs.append('data/q1_lb_no_rtg_dsa_CartPole-v0_19-03-2021_16-05-35/events*')
    logs.append('data/q1_lb_rtg_dsa_CartPole-v0_19-03-2021_16-08-31/events*')
    logs.append('data/q1_lb_rtg_na_CartPole-v0_19-03-2021_16-13-36/events*')
    eval_avg_ret = np.zeros([3,100])
    
    for i, logdir in enumerate(logs):
	    eventfile = glob.glob(logdir)[0]
	    X, Y = get_section_results(eventfile)
	    for j, (x, y) in enumerate(zip(X, Y)):
	    	eval_avg_ret[i][j] = y
		#print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (25,15), dpi = 480)
    for i, title in enumerate(titles):
	    axs[i].plot(eval_avg_ret[i], label = "punctual")
	    axs[i].plot(average(eval_avg_ret[i], 10), label = "avg")
	    axs[i].set_ylabel("Eval_average_return")
	    axs[i].set_xlabel("Iteration number")
	    axs[i].set_title(title)
	    axs[i].legend()
    plt.savefig("results_batch_5000.png")
			
    
