import SimpleITK as sitk
import numpy as np

def register_and_resample(moving, reference, interpolator=sitk.sitkLinear):
    """Register moving image to reference and resample."""
    
    transform = sitk.CenteredTransformInitializer(
        reference,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    resampled = sitk.Resample(
        moving,
        reference,
        transform,
        interpolator,
        0.0,
        moving.GetPixelID()
    )

    return resampled

def sikit_to_just_data(scikit_images):     
        ##transform to just data
        patient_data = {}
        for channel in scikit_images:
            patient_data[channel] = sitk.GetArrayFromImage(scikit_images[channel])
            #sus transpose
            patient_data[channel] = np.transpose(patient_data[channel], (2, 1, 0))
        return patient_data