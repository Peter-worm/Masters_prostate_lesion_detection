import pickle
import os
from pathlib import Path

class preprocess_file_manager:
    def __init__(self, main_folder,preprocess_steps):
        self.main_folder = main_folder
        self.steps = preprocess_steps
        self.current_step = preprocess_steps[0]

    def load_file(self, step, patient_id):
        path = os.path.join(self.main_folder, step, f"{patient_id}.pkl")

        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    def save_file(self, step, patient_id, data):
        path = os.path.join(self.main_folder, step, f"{patient_id}.pkl")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def get_file_names(self):
        data_folder = os.path.join(self.main_folder, self.current_step)
        folder = Path(data_folder)
        ids = []
        for patient in folder.iterdir():
            ids.append(patient.stem)   # removes .plk
        return ids