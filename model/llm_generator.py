class LLMGenerator:
    def __init__(self, cluster_dict):
        self.cluster_dict = cluster_dict
        # اینجا می‌تونی مدل LLM واقعی مثل OpenAI یا HuggingFace بارگذاری کنی
        # الان به شکل ساده فقط پرومپت رو برمی‌گردونیم

    def generate(self, cluster_action):
        prompt = self.cluster_dict[str(cluster_action)]["prompt"]
        # اینجا تولید جمله با مدل زبانی انجام میشه (اینجا ساده شده)
        generated_text = f"[Generated response based on prompt: '{prompt}']"
        return generated_text
