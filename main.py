from threading import Thread, Lock

import gym
import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'CartPole-v0', 'The game to load from OpenAI\'s gym.')
flags.DEFINE_integer('n_threads', 4, 'The number of threads running in parallel.')
flags.DEFINE_integer('T_MAX', 20000000, 'Total maximum number of steps.')
flags.DEFINE_integer('t_max', 5, 'Maximum number of steps before ation-value network update.')
flags.DEFINE_integer('I_TARGET', 4000, 'Synchronize target and action-value network every I_UPDATE global steps.')
flags.DEFINE_integer('T_ANNEAL', 40000, 'Duration over which epsilons are annealed linearly from 1 to their final value.')
flags.DEFINE_float('gamma', 0.99, 'The discount factor  for rewards.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('decay_steps', 100000, 'Regulates the speed of learning rate decay. The learning rate decays following: lr = initial_lr * decay_rate ^ (current_step/decay_steps).')
flags.DEFINE_integer('decay_rate', 0.96, 'Learning rate\'s rate of decay.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_boolean('render', False, 'Wether or not to render the learning environments.')
flags.DEFINE_boolean('monitor', False, 'Wether or not to monitor the learning environments for evaluation.')
# Final epsilon values that each thread converges to after T_ANNEAL global steps
FINAL_EPSILONS = np.random.choice([0.01,0.1,0.3], size=FLAGS.n_threads, p=[0.3,0.4,0.3])
# Global step counter
T = 0

ENVS = {'CartPole-v0':
            {'n_actions': 2,
             'state_dim': 4}
        }

def nStepQModel():
    """
    Defines the model for n-step Q-learning
    """
    with tf.device('/cpu:0'):
        # Placeholder for the environment's state
        state = tf.placeholder('float', [None, FLAGS.state_dim], name='state')
        # Define a feedforward network
        inputs = Input(shape=(FLAGS.state_dim,))
        hidden = Dense(64, activation='relu')(inputs)
        # hidden = BatchNormalization(mode=2)(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        # hidden = BatchNormalization(mode=2)(hidden)
        q_values = Dense(FLAGS.n_actions, activation='linear')(hidden)

        model = Model(input=inputs, output=q_values)
        model_params = model.trainable_weights
    return state, model(state), model_params

def buildOptimizer(global_step, state, q_values, model_params):
    # Batch of 1 to n step returns
    R = tf.placeholder('float', [None], name = 'R')
    #  Actions taken during those steps
    actions = tf.placeholder('float', [None, FLAGS.n_actions], name = 'actions')
    past_q_values = tf.reduce_sum(tf.mul(actions, q_values), reduction_indices=1)
    loss = tf.reduce_mean(tf.square(R - past_q_values))
    # Exponentially decaying learning rate
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    minimize = optimizer.minimize(loss, var_list = model_params, global_step=global_step) # Minimize will automatically increment global_step every time it is ran

    return R, actions, minimize

def epsilonGreedy(sess, thread_id, q_values):
    final_eps = FINAL_EPSILONS[thread_id]
    current_T = np.minimum(T, FLAGS.T_ANNEAL)
    current_eps = 1 - current_T*(1-final_eps)/FLAGS.T_ANNEAL
    # Sample from Bernoulli with mean epsilon
    exploration = np.random.binomial(1, current_eps)
    if exploration == 1:
        return np.random.choice(FLAGS.n_actions)
    else:
        return np.argmax(q_values)

def threadLearner(sess, thread_id, act_val_ops, target_ops, thread_ops, optim_ops, placeholders, render_lock):
    global T
    # Unpack operations
    q_act_val, act_val_params = act_val_ops
    q_target, target_params = target_ops
    s_thread, q_thread, thread_params = thread_ops
    R, actions, minimize = optim_ops
    s_act_val, s_target= placeholders
    # Operations list for synchronizing shared q model and shared target model
    sync_target = [target.assign(act_val) for act_val, target in zip(act_val_params, target_params)]
    # Operations list for updating globally shared parameters with thread-specific ones
    update_act_val = [act_val.assign(thread) for act_val, thread in zip(act_val_params, thread_params)]
    # Inverse operations for synchronizing thread-specific model and shared model
    sync_thread = [thread.assign(act_val) for act_val, thread in zip(act_val_params, thread_params)]

    # Create the environment
    env = gym.make(FLAGS.game)
    if FLAGS.monitor:
        env.monitor.configure(video_callable=lambda count: False)
        env.monitor.start('/tmp/{0}-experiment-1'.format(FLAGS.game))
        # env.monitor.start('/tmp/{0}-experiment-{1}'.format(FLAGS.game, thread_id))

    while T < FLAGS.T_MAX:
        t = 1
        s = env.reset()
        terminal = False
        total_reward = 0
        while True:
            # Render the environment? Make sure to lock as rendering is not thread-safe
            if FLAGS.render:
                with render_lock:
                    env.render()
            # Synchronize thread-specific parameters
            sess.run(sync_thread)
            # Initialize loop counters and environment
            t_start = t
            R_batch = []
            actions_batch = []
            states_batch = []
            while not terminal and (t-t_start != FLAGS.t_max):
                q_values = sess.run(q_act_val, feed_dict={s_act_val: [s]})
                action = epsilonGreedy(sess, thread_id, q_values)
                # Take action in the environment
                s_prime, r, terminal, info = env.step(action)
                # One-hot enconding of action taken
                action_onehot = np.zeros(FLAGS.n_actions)
                action_onehot[action] = 1
                # Save history
                actions_batch.append(action_onehot)
                states_batch.append(s)
                R_batch.append(r)
                # Increment loop
                s = s_prime
                t += 1
                T += 1
                total_reward += r

            if terminal:
                R_batch.append(0)
            else:
                R_batch.append(np.max(sess.run(q_target, feed_dict={s_target: [s]})))
            for i in reversed(range(len(R_batch)-1)):
                R_batch[i] += FLAGS.gamma * R_batch[i+1]
            # Remove the final entry in R_batch which corresponds to the estimate of total discounted reward from final visited state
            del R_batch[-1]
            # Update the shared action-value model with gradients computed from thread-specific model
            sess.run(minimize, feed_dict={R: R_batch,
                                          actions: actions_batch,
                                          s_thread: states_batch})
            # Update the shared action-value model
            sess.run(update_act_val)
            # Synchronize the shared target and action-value models every I_UPDATE global steps
            if T % FLAGS.I_TARGET == 0:
                sess.run(sync_target)
            # Re-initialize an environment if a terminal state is reached
            if terminal:
                print('Thread {0} || {1} Steps || {2} Total reward || T = {3}'.format(thread_id, t, total_reward, T))
                break
    # Don't forget to close the environment once we're finished
    if FLAGS.monitor:
        envs.monitor.close()
    env.close()

def train():
    sess = tf.Session()
    K.set_session(sess)
    # Set seed for repeatable experiments
    tf.set_random_seed(123)
    # Set environment action and observation spaces dimensions
    FLAGS.n_actions = ENVS[FLAGS.game]['n_actions']
    FLAGS.state_dim = ENVS[FLAGS.game]['state_dim']
    # Global shared timestep counter
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Global shared action-value and target networks
    s_act_val, q_act_val, act_val_params = nStepQModel()
    s_target, q_target, target_params = nStepQModel()

    act_val_ops = (q_act_val, act_val_params)
    target_ops = (q_target, target_params)
    placeholders = (s_act_val, s_target)

    thread_ops = [nStepQModel() for thread_id in range(FLAGS.n_threads)]
    optimizers = [buildOptimizer(global_step, *thread_ops[thread_id]) for thread_id in range(FLAGS.n_threads)]

    # Initialize shared global variables
    sess.run(tf.initialize_all_variables())
    # Rendering is not thread-safe. Set a lock to be used when rendering environments
    render_lock = Lock()

    threads = [Thread(target=threadLearner,
                      args=(sess, thread_id, act_val_ops, target_ops, thread_ops[thread_id], optimizers[thread_id], placeholders, render_lock))
               for thread_id in range(FLAGS.n_threads)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def main(*argv):
    train()

if __name__ == '__main__':
    tf.app.run()