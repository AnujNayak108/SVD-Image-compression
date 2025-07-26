import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image

# --- Sidebar Explanations ---
st.sidebar.title("About SVD Image Compression")
st.sidebar.markdown("""
**Singular Value Decomposition (SVD)** is a mathematical technique that factorizes a matrix into three components: $A = U \Sigma V^T$. In image compression, SVD allows us to approximate the image using only the most significant singular values, reducing storage while preserving visual quality.

- **Singular values** represent the importance of each component. Retaining more singular values means higher quality but less compression.
- **Compression** is achieved by keeping only the top $k$ singular values and setting the rest to zero, reconstructing an approximation of the original image.
""")

# --- Helper Functions ---
def svd_compress_channel(channel, k):
    U, S, VT = np.linalg.svd(channel, full_matrices=False)
    S_k = np.zeros_like(S)
    S_k[:k] = S[:k]
    compressed = np.dot(U, np.dot(np.diag(S_k), VT))
    return compressed, S

def compress_image(img, k):
    if len(img.shape) == 2:  # Grayscale
        compressed, S = svd_compress_channel(img, k)
        return np.clip(compressed, 0, 255).astype(np.uint8), [S]
    else:  # Color
        channels = cv2.split(img)
        compressed_channels = []
        singular_values = []
        for ch in channels:
            comp, S = svd_compress_channel(ch, k)
            compressed_channels.append(np.clip(comp, 0, 255).astype(np.uint8))
            singular_values.append(S)
        compressed_img = cv2.merge(compressed_channels)
        return compressed_img, singular_values

def estimate_compressed_size(img_shape, k, is_color):
    # SVD storage: U (Mxk), S (k), VT (kxN) per channel
    if is_color:
        M, N = img_shape[0], img_shape[1]
        size = 3 * (M * k + k + k * N)
    else:
        M, N = img_shape
        size = M * k + k + k * N
    return size

def mse(original, compressed):
    return np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)

def psnr(original, compressed):
    mse_val = mse(original, compressed)
    if mse_val == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))

# --- Streamlit UI ---
st.title("SVD Image Compression Demo")
st.write("""
Upload an image and use the slider to select the number of singular values ($k$) to retain for compression. The app will show the original and compressed images, singular value spectrum, and compression statistics.
""")

uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("Could not read the image. Please upload a valid image file.")
    else:
        # Convert to RGB for display
        if len(img.shape) == 2:
            display_img = img
            is_color = False
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            is_color = True
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            is_color = True

        st.subheader("Original Image")
        st.image(display_img, channels="RGB" if is_color else "GRAY", use_container_width=True)

        # SVD k slider
        max_k = min(img.shape[0], img.shape[1], 100)
        k = st.slider(f"Number of singular values to retain (k)", 1, max_k, min(50, max_k))

        # Compress image
        if is_color:
            img_for_svd = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_for_svd = img
        compressed_img, singular_values = compress_image(img_for_svd, k)

        # Display compressed image
        st.subheader("Compressed Image (SVD Reconstruction)")
        st.image(compressed_img, channels="RGB" if is_color else "GRAY", use_container_width=True)

        # Plot singular values
        st.subheader("Singular Values Spectrum")
        fig, ax = plt.subplots()
        if is_color:
            colors = ['r', 'g', 'b']
            for i, S in enumerate(singular_values):
                ax.plot(np.arange(1, len(S)+1), S, color=colors[i], label=f'Channel {colors[i].upper()}')
                ax.axvline(x=k, color=colors[i], linestyle='--', alpha=0.5)
        else:
            S = singular_values[0]
            ax.plot(np.arange(1, len(S)+1), S, color='k', label='Singular Values')
            ax.axvline(x=k, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value')
        ax.set_title('Singular Values of Image Matrix')
        ax.legend()
        st.pyplot(fig)

        # Size estimation
        orig_size = img_for_svd.size
        comp_size = estimate_compressed_size(img_for_svd.shape, k, is_color)
        st.markdown(f"**Original size:** {orig_size:,} values")
        st.markdown(f"**Compressed size (estimate):** {comp_size:,} values")
        st.markdown(f"**Compression ratio:** {orig_size/comp_size:.2f}x")

        # Optional: Reconstruction error
        st.subheader("Reconstruction Error (MSE / PSNR)")
        err = mse(img_for_svd, compressed_img)
        psnr_val = psnr(img_for_svd, compressed_img)
        st.markdown(f"- **MSE:** {err:.2f}")
        st.markdown(f"- **PSNR:** {psnr_val:.2f} dB")

        # Optional: Download button
        st.subheader("Download Compressed Image")
        buf = io.BytesIO()
        if is_color:
            out_img = Image.fromarray(compressed_img)
        else:
            out_img = Image.fromarray(compressed_img, mode='L')
        out_img.save(buf, format='PNG')
        st.download_button(
            label="Download Compressed Image as PNG",
            data=buf.getvalue(),
            file_name="compressed_svd.png",
            mime="image/png"
        )
else:
    st.info("Please upload an image to begin.") 