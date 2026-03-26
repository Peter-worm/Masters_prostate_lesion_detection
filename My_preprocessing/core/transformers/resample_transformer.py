from core.transformers.preprocessing_transformer import preprocessing_transformer
import SimpleITK as sitk
import numpy as np

class resample_transformer(preprocessing_transformer):
    def __init__(self, crop_size, target_spacing=(0.8, 0.8, 3.5),is_label=False):
        self.crop_size = crop_size
        self.target_spacing = target_spacing
        self.is_label = is_label

    def execute(self, patient_data):
        patient_data['anatomy'] = self.resample_nii(patient_data['anatomy'], target_spacing=self.target_spacing, is_label=True)
        patient_data['t2'] = self.resample_nii(patient_data['t2'], target_spacing=self.target_spacing, is_label=self.is_label)
        patient_data['dwi'] = self.resample_nii(patient_data['dwi'], target_spacing=self.target_spacing, is_label=self.is_label)
        patient_data['adc'] = self.resample_nii(patient_data['adc'], target_spacing=self.target_spacing, is_label=self.is_label)
        return patient_data


    def resample_nii(
    self,
    image,
    target_spacing=(0.8, 0.8, 3.5),
    is_label=False
):
        # Load image
        image
        
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
    
        # Compute new size
        new_size = [
            int(np.round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]
    
        # Choose interpolation
        if is_label:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkBSpline  # or sitkLinear
    
        # Resample
        resampled = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            interpolator,
            image.GetOrigin(),
            target_spacing,
            image.GetDirection(),
            0,
            image.GetPixelID()
        )
    
        return resampled

