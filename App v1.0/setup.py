import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Installing required libraries...")

libraries = [
    "eel",
    "pillow",
    "ultralytics"
]

for lib in libraries:
    print(f"Installing {lib}...")
    install(lib)

print("All required libraries have been installed.")
print("Setup complete!")