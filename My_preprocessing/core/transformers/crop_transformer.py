from core.cropping_lib import center_crop_shift
from core.stat_calc import find_centroid_non_weighted
from core.transformers.preprocessing_transformer import preprocessing_transformer

class non_weighted_crop_transformer(preprocessing_transformer):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def execute(self, patient_data):
        center = find_centroid_non_weighted(patient_data['anatomy'])
        for channel in patient_data:
            patient_data[channel] = center_crop_shift(patient_data[channel], center, self.crop_size)
        return patient_data
