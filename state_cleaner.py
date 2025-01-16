from main import common_states

def state_cleaner(state):
    return "other" if state not in common_states else state
