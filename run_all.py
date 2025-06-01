
import os
import glob
import subprocess

# Procurar ficheiros na pasta datasource
data_files = glob.glob("datasource/*.csv")
print(f"Ficheiros encontrados em datasource/: {data_files}")

# Correr o script principal
print("A correr pipeline_step.py ...")
subprocess.run(["python", "pipeline_step_test.py"], check=True)
