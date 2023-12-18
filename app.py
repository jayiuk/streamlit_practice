import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt



st.title('image_convert')


uploaded_file = st.file_uploader("�대�吏� �뚯씪 �낅줈��.....", type=["jpg", "jpeg", "png"])


def process_image(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Y_channel, Cr, Cb = cv2.split(ycrcb_image)
    Y_channel = clahe.apply(Y_channel)
    merged_ycrcb = cv2.merge([Y_channel, Cr, Cb])
    final_image = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    rgb_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    return rgb_image



def convert_image_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def rotate_image(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
    rotated_image = cv2.warpAffine(image, m, (w, h))
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
    return rotated_image




def plot_histograms(original_image, processed_image):
    Y_original, Cr_original, Cb_original = cv2.split(cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb))
    Y_processed, Cr_processed, Cb_processed = cv2.split(cv2.cvtColor(processed_image, cv2.COLOR_BGR2YCrCb))

    channels = ('Y', 'Cr', 'Cb')

    fig, axs = plt.subplots(2, 3, figsize=(16, 6))


    for i, channel in enumerate([Y_original, Cr_original, Cb_original]):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        axs[0, i].plot(histogram)
        axs[0, i].set_xlim([0, 256])
        axs[0, i].set_title(f'Original {channels[i]} Histogram')

    
    for i, channel in enumerate([Y_processed, Cr_processed, Cb_processed]):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        axs[1, i].plot(histogram)
        axs[1, i].set_xlim([0, 256])
        axs[1, i].set_title(f'Convert {channels[i]} Histogram')

    return fig



if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


    st.image(uploaded_file, caption='Original Image', use_column_width=True)

    option = st.selectbox(
        'option:',
        ('None', 'Histogram Equalization', 'gray', 'rotate')
    )

    
    if option == 'Histogram Equalization':
        processed_image = process_image(image)
        st.image(processed_image, caption = 'Histogram Equalized Image', use_column_width=True)
        st.pyplot(plot_histograms(image, processed_image))

    elif option == 'gray':
        grayscale_image = convert_image_to_grayscale(image)
        st.image(grayscale_image, caption='Grayscale Image', use_column_width=True)
    
    elif option == 'rotate':
        rotated_image = rotate_image(image)
        st.image(rotated_image, caption = 'rotated image', use_column_width = True)