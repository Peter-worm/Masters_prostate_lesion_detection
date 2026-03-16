import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from file_manager import preprocess_file_manager

def do_nothing_normalizer(file):
    return file

def folder_shower(file_manager: preprocess_file_manager,
    step = 'raw',
    normalizer = do_nothing_normalizer,
    channels = {
        'adc': 'adc',
        'anatomy': 'anatomy',
        'dwi': 'dwi',
        't2': 't2'
}):

    def show_image(patient, channel, slice_idx):
        plt.figure(figsize=(5,5))
        plt.imshow(
            normalizer(file_manager.load_file(step, patient)[channel][:, :, slice_idx]),
            cmap='gray',
            vmin=0,
            vmax=1
        )
        plt.title(f"Patient: {patient} | Channel: {channel} | Slice: {slice_idx}")
        plt.axis('off')
        plt.show()

    patients = file_manager.get_file_names()

    # Patient Dropdown (NEW)
    patient_dropdown = widgets.Dropdown(
        options=sorted(patients),
        value=sorted(patients)[0],
        description='Patient:'
    )

    # Slice Slider
    slice_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=50,  # You may later want to update this dynamically per patient
        step=1,
        description='Slice:',
        continuous_update=False
    )

    # Channel Dropdown
    channel_dropdown = widgets.Dropdown(
        options=list(channels.keys()),
        value='adc',
        description='Channel:'
    )

    # Interactive display
    ui = widgets.interactive(
        show_image,
        patient=patient_dropdown,
        channel=channel_dropdown,
        slice_idx=slice_slider
    )

    display(ui)