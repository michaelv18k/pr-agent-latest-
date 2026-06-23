import os
from pr_agent.config_loader import get_settings

print("\n--- 🧠 PR-AGENT CONFIGURATION VERIFIER ---")
print(f"System GROQ_API_KEY:    {os.environ.get('GROQ_API_KEY', '❌ MISSING')[:15]}...")
print(f"System OPENAI_API_KEY:  {os.environ.get('OPENAI_API_KEY', '❌ MISSING')[:15]}...")

print("\n--- 🎛️ DYNACONF PARSED SETTINGS ---")
print(f"Target Model:           {get_settings().get('CONFIG', {}).get('model', '❌ NOT FOUND')}")
print(f"GitHub User Token:      {get_settings().get('GITHUB', {}).get('user_token', '❌ NOT FOUND')[:15]}...")
print("------------------------------------------\n")
