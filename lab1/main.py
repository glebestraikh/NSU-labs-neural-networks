import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ("БИБЛИОТЕЧНАЯ РЕАЛИЗАЦИЯ", os.path.join(BASE, "library_impl", "main.py")),
    ("СОБСТВЕННАЯ РЕАЛИЗАЦИЯ",  os.path.join(BASE, "custom_impl", "main.py")),
]

for title, path in scripts:
    print("#" * 70)
    print(f"  {title}")
    print("#" * 70)
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"[ОШИБКА] Скрипт завершился с кодом {result.returncode}")

