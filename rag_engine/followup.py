"""Simple lexical query augmentation for short context-dependent follow-ups."""
import re


def contextualize_query(prompt: str, history: list[dict]) -> str:
    """Use only previous questions from the same authorized conversation."""
    recent_users = [m.get('content','').strip() for m in history if m.get('role') == 'user']
    if not recent_users:
        return prompt
    # Only augment short questions that appear referential. Avoid polluting
    # a complete new question with an unrelated old topic.
    referential = re.search(r'\b(đó|này|vậy|như thế|trường hợp trên|còn|thế thì|nó|ấy)\b', prompt.casefold())
    if (len(prompt) > 110 or not referential):
        return prompt
    return recent_users[-1][:240] + '\nCâu hỏi tiếp theo: ' + prompt
