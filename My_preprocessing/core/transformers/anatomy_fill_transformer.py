from core.transformers.preprocessing_transformer import preprocessing_transformer
from core.anatomy_gap_fixer import fix_patient_anatomy

class anatomy_fill_transformer(preprocessing_transformer):
    def __init__(self,outliers = None):
        self.outliers = outliers

    def execute(self, patient_data):
        return fix_patient_anatomy(patient_data)
        