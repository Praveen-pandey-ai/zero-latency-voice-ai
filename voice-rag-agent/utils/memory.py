history = []

def add_turn(user, assistant):
    history.append((user, assistant))

def get_history():
    return "\n".join([f"User: {u}\nAssistant: {a}" for u,a in history])
