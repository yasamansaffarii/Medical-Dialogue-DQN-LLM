import random
from utils.state_summary import summarize_state

class DialogueEnv:
    def __init__(self, dialog_data, cluster_dict, symptom_list):
        self.dialog = dialog_data
        self.cluster_dict = cluster_dict
        self.symptom_list = symptom_list
        self.index = 0
        self.history = []

    def reset(self):
        self.index = 0
        self.history = []
        return summarize_state(self.history, self.symptom_list)

    def step(self, cluster_action, llm_generator):
        done = False

        # تولید جمله با LLM بر اساس خوشه
        action_text = llm_generator.generate(cluster_action)

        # اضافه به تاریخچه
        self.history.append({"speaker": "doctor", "text": action_text})

        # گرفتن پاسخ بیمار از دیتای واقعی
        if self.index < len(self.dialog):
            patient_utterance = self.dialog[self.index]
            self.history.append(patient_utterance)
            self.index += 1
        else:
            done = True

        # ساخت state جدید
        next_state = summarize_state(self.history, self.symptom_list)

        # تعریف reward ساده (می‌تونی بعداً پیچیده‌تر کنی)
        reward = self.compute_reward(action_text)

        return next_state, reward, done

    def compute_reward(self, action_text):
        # پاداش ساده برای مثال
        if "diagnosis" in action_text.lower():
            return 1.0
        else:
            return 0.0
