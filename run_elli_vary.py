from l2o_tools import train_meta_opt, test_opts
from first_order_optims import Adam_Opt
from plot import plot_all, plot_trajectories

train_runs = 500
hidden_sz = 100
K = [2,5,8,11,14,17]
task_dist = "elli_vary"
test_runs = 200
test_epochs = 12

# train L2O optimizers
vanilla_l2o = train_meta_opt(n_run=train_runs, hidden_size=hidden_sz, learn_tele=False, input_theta_grad=False, task_dist=task_dist)
sparse_tel = train_meta_opt(n_run=train_runs, hidden_size=hidden_sz, learn_tele=True, input_theta_grad=False, tel_sched=K, task_dist=task_dist)
sparse_tel_theta_grad = train_meta_opt(n_run=train_runs, hidden_size=hidden_sz, learn_tele=True, input_theta_grad=True, tel_sched=K, task_dist=task_dist)
theta_grad = train_meta_opt(n_run=train_runs, hidden_size=hidden_sz, learn_tele=True, input_theta_grad=True, task_dist=task_dist)

opts = [vanilla_l2o, sparse_tel, sparse_tel_theta_grad, theta_grad]

# test L2O optimizers
times, losses, grad_norms, iter_arrs = test_opts(opts, task_dist, test_runs, test_epochs)

#Plot Losses and Gradient Norms vs Epochs
labels = ["vanilla l2o", "tel sched", "tel sched + input theta grad", "input theta grad"]
plot_all(times, losses, grad_norms, labels, n_epoch=test_epochs, xticks=[0,5,10], loc='elli_vary_figs', ylim=0.5)

#Plot Trajectories
labels = [["vanilla l2o", ""], \
            ["tel sched", "tel sched (tele)"], \
            ["tel sched + input theta grad", "tel sched + input theta grad (tele)"], \
            ["input theta grad", "input theta grad (tele)"]]
plot_trajectories(iter_arrs, task_dist, labels, loc='elli_vary_figs')
