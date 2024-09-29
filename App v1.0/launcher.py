import os
import subprocess

def launch_app():
    try:
        # Get the absolute path of the current script
        script_path = os.path.abspath(__file__)
        
        # Get the directory containing the current script
        script_dir = os.path.dirname(script_path)
        
        # Construct the path to the main.py file
        main_path = os.path.join(script_dir, "main.py")
        
        # Launch the main.py script
        subprocess.run(["python", main_path])
    except Exception as e:
        print(f"An error occurred while launching the application: {str(e)}")

if __name__ == "__main__":
    launch_app()
