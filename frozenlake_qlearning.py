"""
Q-learning agent for FrozenLake (4x4).

The agent starts with an empty Q-table and learns by trial and error:
    Q[state][action] = how good it thinks an action is in a state
    It explores randomly at first, then relies more on what it has learnt
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# settings
LEARNING_RATE = 0.05     # how much a new experience can effect Q table
GAMMA = 0.95             # how much future reward is worth compared with now
EPSILON_START = 1.0      # 1.0 => knows nothing so explore at beginning
EPSILON_DECAY = 0.9995   # epsilon shrinks a little after every episode keep close to 1 so it will explore enough
EPSILON_MIN = 0.01       # can never fully stop exploring
EPISODES = 10000         # self explanatory
IS_SLIPPERY = True      # True makes the ice random and much harder
SEED = 42


def make_environment(is_slippery):
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)


def choose_action(env, q_table, state, epsilon):
    # explore with probability epsilon otherwise pick the best known action
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(q_table[state]))


def update_q_table(q_table, state, action, reward, next_state, reached_end):
    if reached_end:
        # no next state so there is no future reward to add
        target = reward
    else:
        target = reward + GAMMA * np.max(q_table[next_state])

    error = target - q_table[state][action]
    q_table[state][action] = q_table[state][action] + LEARNING_RATE * error


def train(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    epsilon = EPSILON_START
    results = []  # 1 if the agent reached the goal in that episode else 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        finished = False
        reward = 0

        while not finished:
            action = choose_action(env, q_table, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # terminated for when fell in a hole or reached the goal
            # truncated for when ran out of time so not a real ending
            update_q_table(q_table, state, action, reward, next_state, terminated)

            state = next_state
            finished = terminated or truncated

        results.append(reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return q_table, results


def test(env, q_table, games=100):
    wins = 0
    for _ in range(games):
        state, _ = env.reset()
        finished = False
        reward = 0
        while not finished:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
        wins += reward
    return wins / games


def plot_learning_curve(results, window=100):
    averages = []
    for i in range(len(results) - window + 1):
        averages.append(np.mean(results[i:i + window]))

    plt.plot(range(window, len(results) + 1), averages)
    plt.xlabel("Episode")
    plt.ylabel(f"Success rate (average of last {window})")
    plt.title("Q-learning on FrozenLake 4x4")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("learning_curve.png", dpi=150)



np.random.seed(SEED)
env = make_environment(IS_SLIPPERY)
env.reset(seed=SEED)
env.action_space.seed(SEED)

q_table, results = train(env)
win_rate = test(env, q_table)

print(f"Win rate after training: {win_rate * 100:.0f}%")
print("Learned best action per tile (0=left, 1=down, 2=right, 3=up):")
print(np.argmax(q_table, axis=1).reshape(4, 4))

plot_learning_curve(results)
print("Saved learning_curve.png")
