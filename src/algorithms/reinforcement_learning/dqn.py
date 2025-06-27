# dqn.py
# You can add your DQN algorithm code here
# Example:
import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, state_size, action_size, params=None):
        """
        Initializes the DQN agent.

        :param state_size: Dimension of the state space.
        :param action_size: Dimension of the action space.
        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.state_size = state_size
        self.action_size = action_size
        self.params = params

        # Set default parameters
        self.hidden_units = self.params.get('hidden_units', [24, 24])
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.gamma = self.params.get('gamma', 0.95)  # Discount factor
        self.epsilon = self.params.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = self.params.get('epsilon_min', 0.01)
        self.epsilon_decay = self.params.get('epsilon_decay', 0.995)
        self.memory = []  # Replay memory
        self.batch_size = self.params.get('batch_size', 32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate) # Use the learning rate

        self.model = self._build_model()
        self.target_model = self._build_model()  # Add a target network
        

    def _build_model(self):
        """
        Builds the DQN model.

        :return: Keras model.
        """
        # Define default parameters
        input_dim = self.state_size
        hidden_units = self.hidden_units
        action_size = self.action_size

        # Design a deep neural network (DNN) model here
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(hidden_units[0], input_dim=input_dim, activation='relu'))
        for units in hidden_units[1:]:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def update_target_model(self):
        """
        Updates the target model with the weights of the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Adds a memory of an experience to the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action based on the current state.  Epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            state = np.array([state]) # Add a dimension to state
            options = self.model.predict(state)[0]
            return np.argmax(options)  # Exploit

    def train(self, state, action, reward, next_state, done):
        """
        Implements the DQN training step here.  Now, this function performs a training step from the replay memory.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in memory to train

        # Sample a minibatch from replay memory
        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch] # Get the actual samples

        states, actions, rewards, next_states, dones = zip(*minibatch)  # Unpack the batch
        states = np.array(states)
        next_states = np.array(next_states)

        # Calculate target Q values
        target_q_values = self.target_model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        self.model.fit(states, target_q_values, epochs=1, verbose=0)  # Train the main model

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, state):
        """
        Returns the predicted Q-values for a given state.
        """
        state = np.array([state])  # Add a dimension to the state
        return self.model.predict(state)[0]

    def evaluate(self, X, y):
        """
        Evaluates the model's performance (this is more complex for RL).  
        For DQN, we might evaluate how well it performs in a simulated environment.
        This is highly environment-dependent, and a general-purpose evaluation is difficult.
        Here's a placeholder that returns the mean Q-value for the given states.  This
        is NOT a proper RL evaluation metric.  You'll need to customize this.
        """
        q_values = self.model.predict(X)
        mean_q = np.mean(q_values)
        return {"mean_q_value": mean_q}

    def get_model(self):
        """
        Returns the trained model.
        """
        return self.model
