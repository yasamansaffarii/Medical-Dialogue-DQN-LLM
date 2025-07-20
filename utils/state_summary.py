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

            # چک کردن علائم توی جمله بیمار
            for idx, symptom in enumerate(symptom_list):
                if symptom in uttr["text"].lower():
                    symptoms[idx] = 1

    # خروجی = علائم + طول دیالوگ (نرمال شده)
    state_vector = symptoms + [num_turns / 100]

    return state_vector
