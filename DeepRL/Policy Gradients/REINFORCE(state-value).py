import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class PG(tf.keras.Model):

    ''' REINFORCE with state-value baseline

    In this implementation, we'll use a seperate optimizer to learn
    a value function to be used as the baseline.

    '''

    def __init__(self, state_size=4, action_size=2, layer_size=32):
        super(PG, self).__init__()

        # shared layers
        inputs = tf.keras.layers.Input(shape=[state_size,])
        x = tf.keras.layers.Dense(layer_size, activation="elu")(inputs)
        x = tf.keras.layers.Dense(layer_size, activation="elu")(x)
        # seperate heads
        actions_logits= tf.keras.layers.Dense(action_size)(x)
        state_value = tf.keras.layers.Dense(1)(x)
        # we use two models because we want to mess with the gradients of the shared layers;
        # a more efficient way is to use outputs=[actions_probs, state_value] and 1 optimizer only.
        self.policy_function=tf.keras.models.Model(inputs=inputs, outputs=actions_logits)
        self.value_function=tf.keras.models.Model(inputs=inputs, outputs=state_value)

    def call(self, state):
        # wrap actions logits with e-greedy policy to select action
        action_probs=tf.keras.layers.Softmax()(self.policy_function(state))
        action = np.random.choice(2, p=action_probs.numpy()[0])

        return action


class Session():

    def __init__(self,
                 env,
                 discount_rate=0.95,
                 n_interations=50, # the number of iteration, each i runs n-episodes
                 n_episdoes_per_update=10, # number of episodes per update step
                 n_max_steps = 200, # maximum steps per episode
                 loss_functions=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 tf.keras.losses.mean_squared_error],
                 optimizers=[tf.keras.optimizers.Adam(lr=0.01), # for policy function
                             tf.keras.optimizers.Adam(lr=0.01), # for actor function
                            ]):
        self.env=env
        self.discount_rate=discount_rate
        self.n_interations=n_interations
        self.n_episdoes_per_update=n_episdoes_per_update
        self.n_max_steps=n_max_steps
        self.loss_functions=loss_functions
        self.optimizers=optimizers

    def train(self, model, show_info=True):

        # use 42 to reproduce result
        env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)


        mean_reward_log=[]
        best_score=0

        for iteration in range(self.n_interations):

            # lists are slow, a more efficient way would be to use placeholder arrays
            all_episode_state_values, all_episode_rewards = [], []
            all_episode_gradients, all_episode_value_gradients = [], []
            for episode in range(self.n_episdoes_per_update):

                current_rewards, current_state_values = [], []
                current_gradients, current_value_gradients = [], []
                state_raw = self.env.reset()

                for step in range(self.n_max_steps):
                    state = tf.cast(state_raw[np.newaxis], dtype = 'float32')
                    with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
                        # compute policy loss, get its gradients
                        action_logits = model.policy_function(state)
                        action = model(state)

                        # get policy loss and gradients
                        policy_loss = self.loss_functions[0](action, action_logits)
                        policy_gradients = policy_tape.gradient(policy_loss, model.policy_function.trainable_variables)

                        # get value loss and gradients
                        state_value = model.value_function(state)
                        next_state, reward, done, info = self.env.step(action)
                        next_state_ = tf.cast(next_state[np.newaxis], dtype = 'float32')
                        next_state_value = model.value_function(next_state_)
                        temporal_target = reward + self.discount_rate*next_state_value
                        value_loss= self.loss_functions[1](temporal_target, state_value)
                        value_gradients = value_tape.gradient(value_loss, model.value_function.trainable_variables)

                        # append experiences
                        current_state_values.append(float(state_value))
                        current_rewards.append(reward) 
                        current_gradients.append(policy_gradients)
                        current_value_gradients.append(value_gradients)

                        state_raw=next_state
                        if done:
                            break

                all_episode_state_values.append(current_state_values)
                all_episode_rewards.append(current_rewards)
                all_episode_gradients.append(current_gradients)
                all_episode_value_gradients.append(current_value_gradients)

            # debugging
            if show_info:
                total_rewards = sum(map(sum, all_episode_rewards))
                mean_rewards = total_rewards / self.n_episdoes_per_update
                print("\rIteration: {}, mean rewards: {:.1f}".format(iteration, mean_rewards), end="")
                # make a record of it
                mean_reward_log.append(mean_rewards)

            if mean_rewards > best_score:
                best_weights = model.policy_function.get_weights()
                best_score = step


            # action score
            all_final_rewards = self._discount_biased_reward_over_n_episodes(all_episode_rewards,
                                                                          all_episode_state_values)
            # compute mean policy gradients
            all_mean_policy_grads = []
            for var_index in range(len(model.policy_function.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_episode_gradients[episode_index][step][var_index]
                     for episode_index, final_rewards in enumerate(all_final_rewards)
                         for step, final_reward in enumerate(final_rewards)], axis=0)
                all_mean_policy_grads.append(mean_grads)

            # compute mean value gradients
            all_mean_value_grads = []
            for var_index in range(len(model.policy_function.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [all_episode_value_gradients[episode_index][step][var_index]
                     for episode_index, episode in enumerate(all_episode_gradients)
                        for step, var in enumerate(episode)], axis=0)

            # apply the Dual Gradient Descent
            self.optimizers[0].apply_gradients(zip(all_mean_policy_grads, model.policy_function.trainable_variables))
            self.optimizers[1].apply_gradients(zip(all_mean_value_grads, model.value_function.trainable_variables))
        # return best weights
        model.policy_function.set_weights(best_weights)

        return mean_reward_log

    def _discount_biased_reward_over_n_episodes(self, all_episode_rewards, all_episode_state_values):
        # calculate discounted rewards for each episode
        all_discounted_episode_rewards = []
        for episode_rewards in all_episode_rewards:
            discounted_episode_rewards = np.empty(len(episode_rewards)) # initiate tensor
            cumulative_rewards = 0
            for i in reversed(range(len(episode_rewards))):
                cumulative_rewards = episode_rewards[i] + cumulative_rewards * self.discount_rate
                discounted_episode_rewards[i] = cumulative_rewards
            all_discounted_episode_rewards.append(discounted_episode_rewards)

        # have the discounted rewards substracted from state value V(s)
        biased_discounted_episode_rewards=[]
        for rewards, state_values in zip(all_discounted_episode_rewards, all_episode_state_values):
            biased_discounted_rewards=[]
            for reward, state_value in zip(rewards, state_values):
                biased_discounted_rewards.append(reward - state_value)
            biased_discounted_episode_rewards.append(biased_discounted_rewards)

        return biased_discounted_episode_rewards

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
        "layer_size":16
        }

    session_args={
        'env':env,
        'loss_functions':[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), tf.keras.losses.mean_squared_error],
        'optimizers':[tf.keras.optimizers.Adam(lr=0.01), tf.keras.optimizers.Adam(lr=0.01)],
        'n_interations': 40,
        'discount_rate': 0.95,
        'n_episdoes_per_update': 10,
        'n_max_steps': 200,
        }

    pg=PG(**model_args)
    sess=Session(**session_args)

    rewards=sess.train(pg)
    sess.plot(rewards)
