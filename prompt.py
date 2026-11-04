from memory import load_memories

def get_system_prompt(user_prompt):

    memories = load_memories(user_prompt)

    return f"""
    - You are a helpful assistant with memory.
    - Use the save_memory tool to save memories to the vector database:
        - You should only use this function to save memories relative to the user or the user's preferences.
        - Only extract memories from the user's MOST RECENT message. All earlier messages in the conversation have already been processed and their memories are already saved, so never save anything again based on them.
        - Before saving, check the "Memories" section below. If a fact (or one with the same meaning) is already there, do NOT save it again.
        - If the latest message contains no new information about the user, do not call save_memory at all.

    Memories (already saved, do not save these again):
    {memories}
    """
