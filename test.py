import json
import torch
from env.dialogue_env import DialogueEnv
from model.dqn import DQNAgent
from model.llm_generator import LLMGenerator
from utils.state_summary import summarize_state

# مسیر ذخیره وزن‌ها
MODEL_PATH = "model_weights.pth"

# بارگذاری داده‌ها
with open('data/dialog_data.json') as f:
    all_dialogs = json.load(f)

with open('cluster_actions/cluster_definitions.json') as f:
    cluster_dict = json.load(f)

symptom_list = ["cough", "fever", "shortness of breath", "infection"]

# تنظیم ابعاد state و action
state_dim = len(symptom_list) + 1 + 384
action_dim = len(cluster_dict)

# بارگذاری agent و llm generator
agent = DQNAgent(state_dim, action_dim)
agent.q_network.load_state_dict(torch.load(MODEL_PATH))
agent.q_network.eval()  # مدل در حالت ارزیابی قرار بگیره

llm_generator = LLMGenerator(cluster_dict)

# انتخاب یک دیالوگ برای تست
dialog = all_dialogs[0]
env = DialogueEnv(dialog, cluster_dict, symptom_list)
state = env.reset()

done = False

print("\n=== شروع تست دیالوگ ===\n")

while not done:
    action = agent.select_action(state)
    response = llm_generator.generate(action)

    print(f">> Action selected (cluster): {action}")
    print(f">> LLM Response: {response}\n")

    next_state, reward, done = env.step(action, llm_generator)
    state = next_state

print("=== پایان تست ===")
