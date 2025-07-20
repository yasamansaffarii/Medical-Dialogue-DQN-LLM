import json
import random
from env.dialogue_env import DialogueEnv
from model.dqn import DQNAgent
from model.llm_generator import LLMGenerator
from utils.state_summary import summarize_state

# بارگذاری داده‌ها
with open('data/dialog_data.json') as f:
    all_dialogs = json.load(f)

with open('cluster_actions/cluster_definitions.json') as f:
    cluster_dict = json.load(f)

symptom_list = ["cough", "fever", "shortness of breath", "infection"]

# ابعاد state و action
state_dim = len(symptom_list) + 1 + 384  # (علائم + length + انکدینگ متن)
action_dim = len(cluster_dict)

# تعریف agent و generator
agent = DQNAgent(state_dim, action_dim)
llm_generator = LLMGenerator(cluster_dict)

# حلقه آموزش
n_epochs = 50
target_update_freq = 5

for epoch in range(n_epochs):

    random.shuffle(all_dialogs)
    total_reward = 0

    for dialog in all_dialogs:
        env = DialogueEnv(dialog, cluster_dict, symptom_list)
        state = env.reset()

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action, llm_generator)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

    if epoch % target_update_freq == 0:
        agent.update_target_network()

    print(f"Epoch {epoch}, Total Reward: {total_reward}")

    # ذخیره وزن‌ها هر 5 epoch
    if epoch % 5 == 0:
        torch.save(agent.q_network.state_dict(), "model_weights.pth")
