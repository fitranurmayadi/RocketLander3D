import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"Files in CWD: {os.listdir('.')}")

try:
    import rocket_lander
    print("Import rocket_lander SUCCESS")
    from rocket_lander import RocketLanderEnv
    print("Import RocketLanderEnv SUCCESS")
except ImportError as e:
    print(f"Import FAILED: {e}")
except Exception as e:
    print(f"Error: {e}")
