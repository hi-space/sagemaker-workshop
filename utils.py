import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

def pretty_print(json_data):
    try:
        if isinstance(json_data, str):
            parsed = json.loads(json_data)
        elif isinstance(json_data, dict):
            parsed = json_data
        else:
            print(json_data)
            return
        
        pretty_json = json.dumps(parsed, indent=4, ensure_ascii=False)
        colored_json = highlight(pretty_json, JsonLexer(), TerminalFormatter())
        print(colored_json)
    except Exception:
        print(json_data)