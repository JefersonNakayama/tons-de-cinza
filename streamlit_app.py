import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

st.title("Comparação de Conversão para Tons de Cinza")
st.markdown("Confira o [artigo/site](https://e2eml.school/convert_rgb_to_grayscale.html) para mais informações.")
color_img = image.imread('/workspaces/tons-de-cinza/assets/tmnt.jpg')

def gamma_correction(c_srgb):
    c_linear = np.where(
        c_srgb <= 0.04045,
        c_srgb / 12.92,
        ((c_srgb + 0.055) / 1.055) ** 2.4
    )
    return c_linear

weights = [0.2989, 0.5870, 0.1140]
gray_img = np.dot(color_img[..., :3], weights)

gray_img_normalized = gray_img / np.max(gray_img)

linear_img = gamma_correction(color_img[..., :3])

gray_img_corrected = np.dot(linear_img, weights)

gray_img_corrected_normalized = gray_img_corrected / np.max(gray_img_corrected)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(color_img, caption='Imagem RGB Original', use_column_width=True)

with col2:
    st.image(gray_img_normalized, caption='Imagem em Tons de Cinza (Sem Correção de Gama)', use_column_width=True)

with col3:
    st.image(gray_img_corrected_normalized, caption='Imagem em Tons de Cinza (Com Correção de Gama)', use_column_width=True)

plt.figure(figsize=(10, 5))
plt.hist(gray_img_normalized.flatten(), bins=256, color='grey', alpha=0.5, label="Sem Correção de Gama")
plt.hist(gray_img_corrected_normalized.flatten(), bins=256, color='black', alpha=0.5, label="Com Correção de Gama")
plt.title('Comparação de Histogramas das Imagens em Tons de Cinza')
plt.xlabel('Intensidade de Cinza')
plt.ylabel('Número de Pixels')
plt.legend()
plt.grid()
plt.show()
