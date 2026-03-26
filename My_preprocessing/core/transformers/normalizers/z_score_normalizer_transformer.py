from anyio import key
import numpy as np

from core.transformers.preprocessing_transformer import preprocessing_transformer
import SimpleITK as sitk
import numpy as np

class z_score_normalizer_transformer(preprocessing_transformer):
    def __init__(self):
        pass

    def execute(self, patient_data):
        patient_data['t2'] = self.zscore_normalize_global(patient_data['t2'],patient_data['anatomy'])
        patient_data['dwi'] = self.zscore_normalize_global(patient_data['dwi'],patient_data['anatomy'])
        patient_data['adc'] = self.zscore_normalize_global(patient_data['adc'],patient_data['anatomy'])                                                                        
        return patient_data


    def zscore_normalize_global(self, img, mask):
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(mask)

        arr_stat_array = arr[mask_arr > 0]

        mean = arr_stat_array.mean()
        std = arr_stat_array.std()

        if std == 0:
            std = 1.0

        norm_arr = (arr - mean) / std

        norm_img = sitk.GetImageFromArray(norm_arr)

        norm_img.CopyInformation(img)

        for key in img.GetMetaDataKeys():
            norm_img.SetMetaData(key, img.GetMetaData(key))

        return norm_img
    
    