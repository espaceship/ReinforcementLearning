import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
from collections import deque


class DDQN(tf.keras.Model):

    '''Dueling Deep Q Network

    In this implementation, we'll use Double DQN  + Dueling Architecture.

    '''

    def __init__(self, state_size=4, action_size=2, layer_size=32):
        super(DDQN, self).__init__()

        inputs = tf.keras.layers.Input(shape=[state_size])
        x = tf.keras.layers.Dense(layer_size, activation="elu")(inputs)
        x = tf.keras.layers.Dense(layer_size, activation="elu")(x)
        z= tf.keras.layers.Dense(layer_size, activation="elu")(x)
        # state value & advantages
        state_values = tf.keras.layers.Dense(1)(z)
        raw_advantages = tf.keras.layers.Dense(action_size)(z)
        # combined to form q-values
        advantages = raw_advantages - tf.math.reduce_max(raw_advantages, axis=1, keepdims=True)
        q_values = state_values + advantages

        self.q_function=tf.keras.models.Model(inputs=inputs, outputs=q_values)

    def call(self, state, epsilon=0):

        action_q_values=0
        # wrappeed with e-greedy
        if np.random.rand() < epsilon:
            action=np.random.randint(2)
        else:
            action_q_values=self.q_function(state)
            action=np.argmax(action_q_values, axis=1)

        return action


class Session():

    def __init__(self, env,
                 loss_fn,
                 optimizer,
                 batch_size,
                 n_episodes,
                 discount_rate=0.95,
                 buffer_len=2000,
                 update_target=10,
                 ma_window=20,
                 exploration_decay_factor=500):
        self.env=env
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.batch_size=batch_size
        self.n_episodes=n_episodes
        self.discount_rate=discount_rate
        self.replay_memory=deque(maxlen=buffer_len)
        self.update_target=update_target
        self.ma_window=ma_window
        self.exploration_decay_factor=exploration_decay_factor

    def train(self, model, show_info=True):
        # specify target network
        target = tf.keras.models.clone_model(model.q_function)
        target.set_weights(model.q_function.get_weights())

        # use 42 to reproduce result
        env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        rewards=[]
        best_score=0
        for episode in range(self.n_episodes):
            state = self.env.reset()
            for step in range(200):
                # reduce epsilon over episode
                epsilon = max(1 - episode/self.exploration_decay_factor, 0.01)
                state, reward, done = self._play_one_step(state, model, epsilon)
                if done:
                    break
            # since we use e-greedy policy, the score will bounce around.
            rewards.append(step+1)  # In our case, cumulative reward = cumulative step+1
            if step > best_score:
                best_weights = model.q_function.get_weights()
                best_score = step
            # debugging
            if show_info:
                print("\rEpisode: {}, \
                        Steps: {},  \
                        eps: {:.3f}".format(episode, step + 1, epsilon), end="")
            if episode > 50: # update after watching it play more than 50 episodes`
                self._training_step(model, target)
            if episode % self.update_target == 0:  # update target network
                target.set_weights(model.q_function.get_weights())
        # return best weights
        model.q_function.set_weights(best_weights)

        return rewards

    def _training_step(self, model, target):
        # get experiences each of shape (batch_size,)
        states, actions, rewards, next_states, dones = self._sample_experiences() # turple
        # action selection
        next_Q_values = model.q_function.predict(next_states) # (5, 2)
        # choose the max value (best action)
        best_next_actions = np.argmax(next_Q_values, axis=1)

        # action evaluation Q(s,  best_next_actions)
        next_mask = tf.one_hot(best_next_actions, 2).numpy()
        next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)

        # get temporal difference target
        target_Q_values = rewards + (1 - dones) * self.discount_rate * next_best_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)

        mask = tf.one_hot(actions, 2)

        with tf.GradientTape() as tape:
            # Q_values = model.predict(state[np.newaxis])
            all_Q_values = model.q_function((tf.cast(states, dtype = 'float32')))
            # you mask the values you didn't choose
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, model.q_function.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, model.q_function.trainable_variables))

    def _play_one_step(self, state, model, epsilon):

        state_ = tf.cast(state[np.newaxis], dtype = 'float32')
        action = model(state_, epsilon)
        next_state, reward, done, info = self.env.step(int(action))
        # append experience to memory
        self.replay_memory.append((state, int(action), reward, next_state, done))

        return next_state, reward, done

    def _sample_experiences(self):
        # sample experience of size "batch_size"
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        # this gives out a list of (obs, actions ,rewards , next_obs, dones)
        batch = [self.replay_memory[index] for index in indices]

        # group each field into a np.array of shape (batch_size,...)
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        # shape: (batch_size, feature_dim)
        return states, actions, rewards, next_states, dones

    def plot(self, rewards, window=0):
        # smoothing window
        if window !=0:
            weights = np.repeat(1.0, window)/window
            rewards = np.convolve(rewards, weights, 'valid')
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=16)
        plt.ylabel("Total Rewards", fontsize=16)
        plt.show()


if __name__ == '__main__':

    # use 42 to reproduce result
    np.random.seed(42)
    tf.random.set_seed(42)


    env = gym.make("CartPole-v1")

    model_args={
        "state_size":4,
        "action_size":2,
        "layer_size":32
        }

    session_args={
        'env':env,
        'loss_fn':tf.keras.losses.mean_squared_error,
        'optimizer':tf.keras.optimizers.Adam(lr=1e-3),
        'batch_size':32,
        'n_episodes': 600,
        'discount_rate': 0.95,
        'buffer_len': 2000,
        'update_target': 10, # 1 means we don't use fixed target at all
        'exploration_decay_factor':600,
        'ma_window':50
        }

    ddqn=DDQN(**model_args)
    sess=Session(**session_args)

    rewards=sess.train(ddqn)
    sess.plot(rewards, 20)
