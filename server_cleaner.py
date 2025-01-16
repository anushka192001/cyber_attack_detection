from main import common_servers

def server_cleaner(server):
    return "other" if server not in common_servers else server

