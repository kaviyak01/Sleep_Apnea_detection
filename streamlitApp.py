# import streamlit as st
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import wfdb
# import os
# import tempfile

# # ===========================
# # Model Definition
# # ===========================
# class ResNetBlock1D(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
#         super().__init__()
#         self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
#         self.bn1 = torch.nn.BatchNorm1d(out_channels)
#         self.relu = torch.nn.ReLU()
#         self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
#         self.bn2 = torch.nn.BatchNorm1d(out_channels)
#         self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()
#     def forward(self, x):
#         identity = self.downsample(x)
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += identity
#         out = self.relu(out)
#         return out

# class SleepApneaNet(torch.nn.Module):
#     def __init__(self, input_channels=1, num_classes_bin=2, num_classes_sev=3):
#         super().__init__()
#         self.res1 = ResNetBlock1D(input_channels, 16)
#         self.res2 = ResNetBlock1D(16, 32)
#         self.res3 = ResNetBlock1D(32, 64)
#         self.gru = torch.nn.GRU(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
#         self.fc_bin = torch.nn.Linear(64, num_classes_bin)
#         self.fc_sev = torch.nn.Linear(64, num_classes_sev)
#     def forward(self, x):
#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.res3(x)
#         x = x.permute(0, 2, 1)
#         out_gru, _ = self.gru(x)
#         out_gru = out_gru[:, -1, :]
#         bin_out = self.fc_bin(out_gru)
#         sev_out = self.fc_sev(out_gru)
#         return bin_out, sev_out

# # ===========================
# # Load Model
# # ===========================
# device = torch.device("cpu")
# model = SleepApneaNet()
# model.load_state_dict(torch.load("D:/apnea/model_demo.pt", map_location=device))
# model.eval()

# st.title("Sleep Apnea Detection with ECG")

# # ===========================
# # File uploader
# # ===========================
# uploaded_files = st.file_uploader(
#     "Upload ECG files (.dat, .hea, .apn)",
#     type=["dat","hea","apn"],
#     accept_multiple_files=True
# )

# if uploaded_files and len(uploaded_files) >= 2:  # need at least .dat and .hea
#     tmp_dir = tempfile.mkdtemp()
#     for f in uploaded_files:
#         with open(os.path.join(tmp_dir, f.name), "wb") as temp_file:
#             temp_file.write(f.getbuffer())

#     base_name = os.path.join(tmp_dir, uploaded_files[0].name.split(".")[0])
#     record = wfdb.rdrecord(base_name)
#     ecg_signal = record.p_signal[:, 0]

#     # Take first 6000 samples for demo
#     segment_np = ecg_signal[:6000]
#     segment_np = (segment_np - np.mean(segment_np)) / np.std(segment_np)
#     segment = torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#     # ===========================
#     # Grad-CAM / Highlighting
#     # ===========================
#     activations = {}
#     gradients = {}
#     def forward_hook(module, input, output):
#         activations['value'] = output
#     def backward_hook(module, grad_in, grad_out):
#         gradients['value'] = grad_out[0]
#     target_layer = model.res3.conv2
#     target_layer.register_forward_hook(forward_hook)
#     target_layer.register_full_backward_hook(backward_hook)

#     out_bin, out_sev = model(segment)
#     pred_class = out_bin.argmax(dim=1).item()
#     severity_class = out_sev.argmax(dim=1).item()
#     model.zero_grad()
#     out_bin[0, pred_class].backward()

#     activation = activations['value'][0]
#     gradient = gradients['value'][0]
#     weights = gradient.mean(dim=1)
#     cam = (weights[:, None] * activation).sum(dim=0)
#     cam = F.relu(cam)
#     cam = cam / cam.max()

#     # ===========================
#     # Visualization with circles
#     # ===========================
#     downsample_factor = 50
#     time_ds = np.arange(0, segment.shape[-1], downsample_factor)
#     ecg_ds = segment.squeeze().numpy()[::downsample_factor]
#     cam_ds = cam.detach().numpy()[::downsample_factor]

#     plt.figure(figsize=(18,6))
#     plt.plot(time_ds, ecg_ds, color='blue', linewidth=0.8, label='ECG Signal')

#     threshold = 0.3
#     highlight_indices = np.where(cam_ds > threshold)[0]
#     plt.scatter(time_ds[highlight_indices], ecg_ds[highlight_indices],
#                 s=200, facecolors='none', edgecolors='red', linewidths=2, label='Apnea Event')

#     plt.title(f"ECG with Apnea Highlighting\nBinary: {pred_class}, Severity: {severity_class}")
#     plt.xlabel("Samples")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     st.pyplot(plt)




# import streamlit as st
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import wfdb
# import os
# import tempfile
# import io

# # ===========================
# # Model Definition
# # ===========================
# class ResNetBlock1D(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
#         super().__init__()
#         self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
#         self.bn1 = torch.nn.BatchNorm1d(out_channels)
#         self.relu = torch.nn.ReLU()
#         self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
#         self.bn2 = torch.nn.BatchNorm1d(out_channels)
#         self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()
#     def forward(self, x):
#         identity = self.downsample(x)
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += identity
#         out = self.relu(out)
#         return out

# class SleepApneaNet(torch.nn.Module):
#     def __init__(self, input_channels=1, num_classes_bin=2, num_classes_sev=3):
#         super().__init__()
#         self.res1 = ResNetBlock1D(input_channels, 16)
#         self.res2 = ResNetBlock1D(16, 32)
#         self.res3 = ResNetBlock1D(32, 64)
#         self.gru = torch.nn.GRU(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
#         self.fc_bin = torch.nn.Linear(64, num_classes_bin)
#         self.fc_sev = torch.nn.Linear(64, num_classes_sev)
#     def forward(self, x):
#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.res3(x)
#         x = x.permute(0, 2, 1)
#         out_gru, _ = self.gru(x)
#         out_gru = out_gru[:, -1, :]
#         bin_out = self.fc_bin(out_gru)
#         sev_out = self.fc_sev(out_gru)
#         return bin_out, sev_out

# # ===========================
# # Load Model
# # ===========================
# device = torch.device("cpu")
# model = SleepApneaNet()
# model.load_state_dict(torch.load("D:/apnea/model_demo.pt", map_location=device))
# model.eval()

# st.title("🫀 Sleep Apnea Detection with Explainable AI (MLG-CAM + Circles)")

# # ===========================
# # File uploader
# # ===========================
# uploaded_files = st.file_uploader(
#     "Upload ECG files (.dat, .hea, .apn)", 
#     type=["dat","hea","apn"], 
#     accept_multiple_files=True
# )

# if uploaded_files and len(uploaded_files) >= 2:  # need at least .dat and .hea
#     tmp_dir = tempfile.mkdtemp()
#     for f in uploaded_files:
#         with open(os.path.join(tmp_dir, f.name), "wb") as temp_file:
#             temp_file.write(f.getbuffer())

#     base_name = os.path.join(tmp_dir, uploaded_files[0].name.split(".")[0])
#     record = wfdb.rdrecord(base_name)
#     ecg_signal = record.p_signal[:, 0]

#     # Take first 6000 samples for demo
#     segment_np = ecg_signal[:6000]
#     segment_np = (segment_np - np.mean(segment_np)) / np.std(segment_np)
#     segment = torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#     # ===========================
# #     # Grad-CAM / Highlighting
# #     # ===========================
#     activations = {}
#     gradients = {}
#     def forward_hook(module, input, output):
#         activations['value'] = output
#     def backward_hook(module, grad_in, grad_out):
#         gradients['value'] = grad_out[0]
#     target_layer = model.res3.conv2
#     target_layer.register_forward_hook(forward_hook)
#     target_layer.register_full_backward_hook(backward_hook)

#     out_bin, out_sev = model(segment)
#     pred_class = out_bin.argmax(dim=1).item()
#     severity_class = out_sev.argmax(dim=1).item()
#     model.zero_grad()
#     out_bin[0, pred_class].backward()

#     activation = activations['value'][0]
#     gradient = gradients['value'][0]
#     weights = gradient.mean(dim=1)
#     cam = (weights[:, None] * activation).sum(dim=0)
#     cam = F.relu(cam)
#     cam = cam / cam.max()

#     # ===========================
#     # Visualization with circles
#     # ===========================
#     downsample_factor = 50
#     time_ds = np.arange(0, segment.shape[-1], downsample_factor)
#     ecg_ds = segment.squeeze().numpy()[::downsample_factor]
#     cam_ds = cam.detach().numpy()[::downsample_factor]

#     plt.figure(figsize=(18,6))
#     plt.plot(time_ds, ecg_ds, color='blue', linewidth=0.8, label='ECG Signal')

#     threshold = 0.3
#     highlight_indices = np.where(cam_ds > threshold)[0]
#     plt.scatter(time_ds[highlight_indices], ecg_ds[highlight_indices],
#                 s=200, facecolors='none', edgecolors='red', linewidths=2, label='Apnea Event')

#     plt.title(f"ECG with Apnea Highlighting\nBinary: {pred_class}, Severity: {severity_class}")
#     plt.xlabel("Samples")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     st.pyplot(plt)

#     # ===========================
#     # Result Section
#     # ===========================
#     st.subheader("Apnea Detection:")
#     if pred_class == 0:
#         st.success("🟢 **Normal** — No apnea detected.")
#     else:
#         st.error("🔴 **Apnea Detected** — Breathing interruptions observed.")

#     st.markdown("---")
#     st.subheader("Severity Classification:")
#     severity_labels = ["Mild Apnea", "Moderate Apnea", "Severe Apnea"]
#     colors = [st.info, st.warning, st.error]
#     colors[severity_class](f"**{severity_labels[severity_class]}** — ({severity_percent:.2f}%)")
#     st.progress(int(severity_percent))

#     st.markdown("---")

#     # ===========================
#     # Lifestyle & Medical Recommendations
#     # ===========================
#     if st.button("View Recommendation"):
#         if severity_class == 0:
#             st.write("""
#             **Mild Sleep Apnea**  
#             - **Causes:** Temporary airway blockage, sleeping posture, or mild obesity.  
#             - **Effects:** Daytime tiredness, light snoring, poor focus.  
#             - **Precautions:** Maintain healthy weight, sleep on side, avoid alcohol & heavy food before bed, and do light exercises.
#             """)
#         elif severity_class == 1:
#             st.write("""
#             **Moderate Sleep Apnea**  
#             - **Causes:** Narrow airways, nasal blockage, or weight gain.  
#             - **Effects:** Morning headaches, irritability, daytime sleepiness.  
#             - **Precautions:** Avoid smoking, keep nasal passages clear, sleep on your side, and seek medical evaluation if symptoms persist.
#             """)
#         else:
#             st.write("""
#             **Severe Sleep Apnea**  
#             - **Causes:** Complete airway obstruction, obesity, or throat structure issues.  
#             - **Effects:** Frequent oxygen drops, high blood pressure, heart strain.  
#             - **Precautions:** Consult doctor for CPAP or oral devices, avoid alcohol and late-night caffeine, maintain good sleep hygiene and weight control.
#             """)

#     # ===========================
#     # Download Option
#     # ===========================
#     st.download_button(
#         label="📥 Download MLG-CAM + Circle Plot",
#         data=buf,
#         file_name="xai_mlgcam_highlight.png",
#         mime="image/png"
#     )



import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
import tempfile

# ===========================
# 1️⃣ Model Definition
# ===========================
class ResNetBlock1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()
    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class SleepApneaNet(torch.nn.Module):
    def __init__(self, input_channels=1, num_classes_bin=2, num_classes_sev=3):
        super().__init__()
        self.res1 = ResNetBlock1D(input_channels, 16)
        self.res2 = ResNetBlock1D(16, 32)
        self.res3 = ResNetBlock1D(32, 64)
        self.gru = torch.nn.GRU(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc_bin = torch.nn.Linear(64, num_classes_bin)
        self.fc_sev = torch.nn.Linear(64, num_classes_sev)
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = x.permute(0, 2, 1)
        out_gru, _ = self.gru(x)
        out_gru = out_gru[:, -1, :]
        bin_out = self.fc_bin(out_gru)
        sev_out = self.fc_sev(out_gru)
        return bin_out, sev_out

# ===========================
# 2️⃣ Load Model
# ===========================
device = torch.device("cpu")
model = SleepApneaNet()
model.load_state_dict(torch.load("D:/apnea/model_demo.pt", map_location=device))
model.eval()

# ===========================
# 3️⃣ Streamlit UI
# ===========================
st.title("🫁 Sleep Apnea Detection from ECG (ResNet + GRU + MLG-CAM)")
st.write("Upload your PhysioNet-style ECG files (.dat, .hea, .apn) to visualize apnea regions.")

uploaded_files = st.file_uploader(
    "Upload ECG Files (.dat, .hea, .apn)",
    type=["dat", "hea", "apn"],
    accept_multiple_files=True
)

# ===========================
# 4️⃣ File Handling
# ===========================
if uploaded_files and len(uploaded_files) >= 2:
    tmp_dir = tempfile.mkdtemp()
    for f in uploaded_files:
        with open(os.path.join(tmp_dir, f.name), "wb") as temp_file:
            temp_file.write(f.getbuffer())

    base_name = os.path.join(tmp_dir, uploaded_files[0].name.split(".")[0])
    record = wfdb.rdrecord(base_name)
    ecg_signal = record.p_signal[:, 0]

    # Normalize & prepare ECG segment
    segment_np = ecg_signal[:6000]
    segment_np = (segment_np - np.mean(segment_np)) / np.std(segment_np)
    segment = torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # ===========================
    # 5️⃣ Grad-CAM setup
    # ===========================
    activations, gradients = {}, {}
    def forward_hook(module, input, output):
        activations['value'] = output
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]
    target_layer = model.res3.conv2
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    out_bin, out_sev = model(segment)
    pred_bin = out_bin.argmax(dim=1).item()
    pred_sev = out_sev.argmax(dim=1).item()

    classes_bin = ['Normal', 'Apnea']
    classes_sev = ['Mild', 'Moderate', 'Severe']

    model.zero_grad()
    out_bin[0, pred_bin].backward()

    activation = activations['value'][0]
    gradient = gradients['value'][0]
    weights = gradient.mean(dim=1)
    cam = (weights[:, None] * activation).sum(dim=0)
    cam = F.relu(cam)
    cam = cam / cam.max()

    # ===========================
    # 6️⃣ Visualization
    # ===========================
    downsample_factor = 50
    time_ds = np.arange(0, segment.shape[-1], downsample_factor)
    ecg_ds = segment.squeeze().numpy()[::downsample_factor]
    cam_ds = cam.detach().numpy()[::downsample_factor]

    # Automatically detect top apnea regions
    num_highlights = 4
    sorted_indices = np.argsort(cam_ds)[::-1]
    selected_indices = []
    min_gap = 30
    for idx in sorted_indices:
        if all(abs(idx - s) > min_gap for s in selected_indices):
            selected_indices.append(idx)
        if len(selected_indices) >= num_highlights:
            break

    plt.figure(figsize=(18, 6))
    plt.plot(time_ds, ecg_ds, color='blue', linewidth=0.8, label='ECG Signal')
    for i, idx in enumerate(selected_indices):
        plt.scatter(time_ds[idx], ecg_ds[idx],
                    s=1000, facecolors='none', edgecolors='red', linewidths=3,
                    label='Detected Apnea' if i == 0 else "")
    plt.title(f"ECG with Detected Apnea Events\nBinary: {classes_bin[pred_bin]} | Severity: {classes_sev[pred_sev]}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    st.pyplot(plt)

    # ===========================
    # 7️⃣ Confidence Graph
    # ===========================
    st.subheader("📊 Classification Confidence")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(classes_bin, F.softmax(out_bin, dim=1).detach().numpy()[0])
    ax[0].set_title("Binary Classification (Normal / Apnea)")
    ax[1].bar(classes_sev, F.softmax(out_sev, dim=1).detach().numpy()[0])
    ax[1].set_title("Severity Classification")
    st.pyplot(fig)

    # ===========================
    # 8️⃣ Summary Output
    # ===========================
    st.success(f"**Binary Prediction:** {classes_bin[pred_bin]}")
    st.info(f"**Severity Prediction:** {classes_sev[pred_sev]}")
    st.write(f"**Detected Apnea Regions:** {len(selected_indices)}")
    st.write("🔴 Red circles mark regions of breathing pause detected by MLG-CAM.")
else:
    st.warning("Please upload at least a `.dat` and `.hea` file to begin.")
