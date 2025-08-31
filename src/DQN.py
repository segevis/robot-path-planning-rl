import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# יצירת סביבה דמוית גריד
class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.pos = [0, 0]  # מיקום התחלתי
        return self.pos

    def step(self, action):
        x, y = self.pos
        if action == 0:  # למעלה
            x = max(0, x - 1)
        elif action == 1:  # למטה
            x = min(self.size - 1, x + 1)
        elif action == 2:  # שמאלה
            y = max(0, y - 1)
        elif action == 3:  # ימינה
            y = min(self.size - 1, y + 1)

        self.pos = [x, y]

        reward = -1
        done = False
        if self.pos == [self.size - 1, self.size - 1]:  # הגיע ליעד
            reward = 100
            done = True

        return self.pos, reward, done


# רשת נוירונים (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# הגדרות
env = GridWorld()
state_dim = 2
action_dim = 4
lr = 0.001
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 32
memory = []
max_memory = 1000

# יצירת רשת וחיזוק
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = DQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# פונקציה לבחירת פעולה
def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    state = torch.FloatTensor(state).to(device)
    with torch.no_grad():
        return torch.argmax(q_network(state)).item()

# הגדרת רשימות לאיסוף נתונים
rewards_history = []
steps_history = []

# אימון
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False
    episode_steps = 0
    path = [state]

    while not done:
        action = -choose_action(state)
        next_state, reward, done = env.step(action)
        path.append(next_state)
        memory.append((state, action, reward, next_state, done))
        if len(memory) > max_memory:
            memory.pop(0)

        # אימון על דוגמאות רנדומיות
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = q_network(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = loss_fn(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        total_reward += reward
        episode_steps += 1

    rewards_history.append(total_reward)
    steps_history.append(episode_steps)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # הדפסת תוצאות לכל 10 פרקים
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.4f}, Steps: {episode_steps}")
        print("Best path:", path)

    # הצגת גרף בזמן אמת
    if episode % 10 == 0:
        plt.clf()
        plt.plot(rewards_history, label="Reward per Episode")
        plt.plot(steps_history, label="Steps per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Reward / Steps")
        plt.ylim(-150, 150)
        plt.legend(loc="best")
        plt.pause(0.1)

# הצגת הגרף הסופי
plt.show()

print("Training complete!")

# הערכת ביצועים
eval_episodes = 10
successes = 0
total_eval_steps = 0
total_eval_rewards = 0

print("\nEvaluating Performance...\n")

for episode in range(eval_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    path = [state]

    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = torch.argmax(q_network(state_tensor)).item()

        next_state, reward, done = env.step(action)
        path.append(next_state)
        episode_reward += reward
        episode_steps += 1
        state = next_state

    total_eval_rewards += episode_reward
    total_eval_steps += episode_steps
    if state == [env.size - 1, env.size - 1]:
        successes += 1

    print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}, Steps = {episode_steps}, Path: {path}")

# חישוב מדדי ביצוע ממוצעים
avg_reward = total_eval_rewards / eval_episodes
avg_steps = total_eval_steps / eval_episodes
success_rate = successes / eval_episodes * 100

print("\nEvaluation Results:")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Average Steps: {avg_steps:.2f}")
print(f"Success Rate: {success_rate:.2f}%")