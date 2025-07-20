from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # یا مدل پزشکی اگه داری

def summarize_state(dialog_history, symptom_list):
    last_doctor = ""
    last_patient = ""
    symptoms = [0] * len(symptom_list)
    num_turns = len(dialog_history)

    for uttr in dialog_history:
        if uttr["speaker"] == "doctor":
            last_doctor = uttr["text"]
        elif uttr["speaker"] == "patient":
            last_patient = uttr["text"]

            for idx, symptom in enumerate(symptom_list):
                if symptom in uttr["text"].lower():
                    symptoms[idx] = 1

    # تبدیل تاریخچه به متن
    history_text = " ".join([u["text"] for u in dialog_history])

    # انکدینگ تاریخچه با SentenceTransformer
    embedding = model.encode(history_text)

    # خروجی = علائم + طول دیالوگ + انکدینگ متن
    state_vector = symptoms + [num_turns / 100] + list(embedding)

    return state_vector
