import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
from collections import deque
import threading
import multiprocessing


class A3C(tf.keras.Model):

    ''' Asynchronous Advantage Actor-Critic (A3C)

    In this implementation, we'll use thread-based workers to run on multiple copies of
    the environment since we only have one machine.
    If we have mutiple machines(many CPU cores), we can use core-based implementation instead.

    '''

    def __init__(self, state_size=4, action_size=2, layer_size=32):
        super(A3C, self).__init__()

        inputs = tf.keras.layers.Input(shape=[state_size])
        x =tf.keras.layers.Dense(layer_size, activation='elu')(inputs)
        x =tf.keras.layers.Dense(layer_size, activation='elu')(x)
        action_logits=tf.keras.layers.Dense(action_size)(x)
        state_value=tf.keras.layers.Dense(1)(x)
        self.twin_function=tf.keras.models.Model(inputs=inputs, outputs=[action_logits, state_value])

    def call(self, state):
        # wrap actions logits with e-greedy policy to select action
        action_logits, _ = self.twin_function(state)
        action_probs=tf.keras.layers.Softmax()(action_logits)
        action = np.random.choice(2, p=action_probs.numpy()[0])

        return action


class Workers(threading.Thread):

    def __init__ (self,
                  global_network=A3C(),
                  max_n_steps=200,
                  discount_rate=0.95,
                  optimizer=tf.keras.optimizers.Adam(),
                  n_episodes=3):
        super(Workers, self).__init__()
        self.env=gym.make("CartPole-v1")
        # global netwro update paras
        self.global_network=global_network
        self.max_n_steps = max_n_steps
        self.memory=deque(maxlen=max_n_steps)
        self.discount_rate=discount_rate
        self.optimizer=optimizer
        self.n_episodes = n_episodes

        # initialize local network with default paras,
        # and then copy the initial paras of the globl network to local network
        self.local_network=A3C()
        self.local_network.twin_function=tf.keras.models.clone_model(global_network.twin_function)

        self.episodic_rewards=np.empty(self.n_episodes)


    def run(self, show_info=True):

        for episode in range(self.n_episodes):
            # clear memory
            self.memory.clear()
            # reset the weight
            self.local_network.twin_function.set_weights(self.global_network.twin_function.get_weights())

            cumulative_reward = 0
            state = self.env.reset()
            # run an episode, store experiences in memory
            for step in range(self.max_n_steps):
                #state, reward, done=self._play_one_step(state)
                state_ = tf.cast(state[np.newaxis], dtype = 'float32')
                action = self.local_network(state_)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append([state, action, reward])
                cumulative_reward += reward
                if done:
                    break
                state=next_state

            if show_info:
                print("\rCurrent episodic reward: {}".format(cumulative_reward), end='')

            # add in reward
            self.episodic_rewards[episode]=cumulative_reward
            
            # compute loss from episodic memory
            with tf.GradientTape() as tape:
                total_loss= self._loss_function()

            grads = tape.gradient(total_loss, self.local_network.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.global_network.trainable_weights))

            # clear memory
            self.memory.clear()


    def _loss_function(self):

        states=[self.memory[step_index][0] for step_index in range(len(self.memory))]
        actions=[self.memory[step_index][1] for step_index in range(len(self.memory))]
        rewards=[self.memory[step_index][2] for step_index in range(len(self.memory))]

        discounted_rewards=[]
        cumulative_rewards = 0
        for i in reversed(range(len(rewards))):
            cumulative_rewards = rewards[i] + cumulative_rewards * self.discount_rate
            discounted_rewards.append(cumulative_rewards)
        discounted_rewards.reverse()

        # get states as a batch
        states_tensor= tf.cast(np.vstack(states), dtype = 'float32')
        action_logits, state_values = self.local_network.twin_function(states_tensor)

        # A(s, a) = Q(s, a) - V(s), here we use actual return G to estimate Q(s, a)
        advantages = tf.cast(np.vstack(discounted_rewards), dtype = 'float32') - state_values
        # value loss: A(s,a) = r + V(s') - V(s)
        value_loss = tf.square(advantages)
        # entropy bonus
        entropy_loss=tf.keras.losses.categorical_crossentropy(action_logits, action_logits, from_logits=True)
        # policy loss
        policy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(actions, action_logits)
        # add entropy bonus to policy loss
        policy_loss -= entropy_loss * 0.001
        # weight policy loss by advantage
        policy_loss *= tf.stop_gradient(advantages)

        total_loss = tf.reduce_mean(policy_loss + value_loss * 0.5)

        return total_loss


class Session():

    def __init__(self,
                  max_n_steps=200,
                  discount_rate=0.95,
                  optimizer=tf.keras.optimizers.Adam(),
                  n_episodes=3,
                  n_workers=2):

        # worker-specific paras
        self.max_n_steps = max_n_steps
        self.discount_rate=discount_rate
        self.optimizer=optimizer
        self.n_episodes = n_episodes

        # session-specific paras
        self.n_workers=n_workers

    def train(self, globel_network):
        # create n workers
        workers=[Workers(globel_network,
                         self.max_n_steps,
                         self.discount_rate,
                         self.optimizer,
                         self.n_episodes) for _ in range(self.n_workers)]

        overall_rewards = []
        for index, worker in enumerate(workers):
            worker.start()
            overall_rewards.append(worker.episodic_rewards)

        # join threads back into one
        op=[worker.join() for worker in workers]

        return overall_rewards

    def plot(self, overall_rewards, window=0):
        # return the mean episodic rewards from all workers(local networks)
        rewards=[]
        for eps_index in range(self.n_episodes):
            mean_reward=np.mean([overall_rewards[worker_index][eps_index]
                     for worker_index, worker in enumerate(overall_rewards)])
            rewards.append(mean_reward)
        if window !=0:
            weights = np.repeat(1.0, window)/window
            rewards = np.convolve(rewards, weights, 'valid')
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=16)
        plt.ylabel("Mean worker rewards", fontsize=16)
        plt.show()


if __name__ == '__main__':

    # use 42 to reproduce result
    np.random.seed(42)
    tf.random.set_seed(42)
    
    env = gym.make("CartPole-v1")

    global_network_args={
        'state_size':4,
        'action_size': 2,
        'layer_size':32
    }

    session_args={
        'max_n_steps':200,
        'discount_rate': 0.95,
        'optimizer': tf.keras.optimizers.Adam(1e-3),
        'n_episodes': 500, # ideally, all workers should reach the target score
        'n_workers':multiprocessing.cpu_count() # keep this fixed
    }

    global_network=A3C(**global_network_args)
    sess=Session(**session_args)

    rewards=sess.train(global_network)
    sess.plot(rewards)
