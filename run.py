import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, n_output):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 128 -> 124
        self.conv2 = nn.Conv2d(6, 16, 3)  # 62 -> 60
        self.conv3 = nn.Conv2d(16, 32, 3)  # 30 -> 28
        self.conv4 = nn.Conv2d(32, 64, 3)  # 14 -> 12
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_output)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.conv3,
            self.relu,
            self.pool,
            self.conv4,
            self.relu,
            self.dropout
        )

        self.classifier = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 顔検出
def detect(image):
    classifier = cv2.CascadeClassifier("lbpcascade_animeface.xml")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_image)

    return faces
# 顔認識
def identify(image, faces, model):
    if len(faces) > 0:
        for face in faces:
            x, y, width, height = face
            detect_face = image[y:y + height, x:x + width]
            # if detect_face.shape[0] < 64:
            #     continue
            detect_face = cv2.resize(detect_face, (128, 128))
            detect_face = np.transpose(detect_face, (2, 0, 1))  # HWC -> CHW
            detect_face = torch.tensor(detect_face, dtype=torch.float32) / 255.0

            transform = transforms.Normalize(0.5, 0.5)
            detect_face = transform(detect_face)
            detect_face = detect_face.view(1, 3, 128, 128)

            output = model(detect_face)

            name_label = output.argmax(dim=1, keepdim=True)
            name, color = label_to_name(name_label)

            cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=3)  # 四角形描画
            font_size = width / 150

            cv2.putText(image, name, (x, y + int(7*height/6)), cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), 6) # アウトライン
            cv2.putText(image, name, (x, y + int(7*height/6)), cv2.FONT_HERSHEY_DUPLEX, font_size, color, 2)  # 内文字

    return image

# ラベルから対応するウマ娘の名前を返す
def label_to_name(name_label):
    if name_label == 0:
        name = "ChevalGrand"
        color = (124, 193, 192)
    elif name_label == 1:
        name = "Verxina"
        color = (71, 71, 197)
    elif name_label == 2:
        name = "Vivlos"
        color = (67, 79, 171)

    return name, color

# Grad-CAM
def GradCam(image, face, model):
    x, y, width, height = face
    detect_face = image[y:y + height, x:x + width]
#    if detect_face.shape[0] < 128:
#        pass

    features = model.features.eval()
    classifier = model.classifier.eval()

    detect_face = cv2.resize(detect_face, (128, 128))
    detect_face = np.transpose(detect_face, (2, 0, 1))  # HWC -> CHW
    detect_face = torch.tensor(detect_face, dtype=torch.float32) / 255.0

    transform = transforms.Normalize(0.5, 0.5)
    detect_face = transform(detect_face)
    detect_face = detect_face.view(1, 3, 128, 128)

    features = features(detect_face)
    features = features.clone().detach().requires_grad_(True)

    pred = classifier(features.view(-1, 64 * 12 * 12))
    pred_index = torch.argmax(pred)
    pred[0][pred_index].backward()

    feature_vec = features.grad.view(64, 12 * 12)

    alpha = torch.mean(feature_vec, axis=1)

    feature = features.squeeze(0)

    l = F.relu(torch.sum(feature * alpha.view(-1, 1, 1), 0))
    l = l.detach().numpy()

    l = np.array(l)
    l_min = np.min(l)
    l_max = np.max(l - l_min)
    l = (l - l_min) / l_max

    l = cv2.resize(l, (128, 128))

    img2 = toHeatmap(l)
    img1 = detect_face.squeeze(0).permute(1,2,0).numpy()
    img1 = np.clip(img1, 0, 1)

    alpha = 0.5
    grad_cam_image = img1 * alpha + img2 * (1 - alpha)
    grad_cam_image = np.clip(grad_cam_image, 0, 1)
    return grad_cam_image

# ヒートマップを表示
def toHeatmap(x):
    x = (x * 128).reshape(-1)
    cm = plt.get_cmap('jet')
    x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
    return x.reshape(128, 128, 3)


def main():
    st.set_page_config(layout="wide")
    # タイトルの表示
    st.title("ウマ娘 三姉妹　顔認識")
    # 制作者の表示
    st.text("Created by Re7U6")
    # アプリの説明の表示
    st.markdown("""### シュヴァルグラン・ヴィブロス・ヴィルシーナを識別""")

    # サイドバーの表示
    image = st.sidebar.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])
    # サンプル画像を使用する場合
    use_sample = st.sidebar.checkbox("サンプル画像")
    if use_sample:
        image = "img/sample.jpg"

    # 保存済みのモデルをロード
    model = CNN(3)
    model.load_state_dict(
        torch.load("model/cnn_31.model", map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    # 画像ファイルが読み込まれた後，顔認識を実行
    if image != None:
        # 画像の読み込み
        image = np.array(Image.open(image))
        image1 = np.copy(image)
        # 顔検出
        faces = detect(image1)
        # 分類
        identify_img = identify(image1, faces, model)
        # 分類結果を表示
        st.markdown("""## ・識別結果""")
        st.image(identify_img, use_column_width=True)
        # GradCAM
        if len(faces) > 0:
            st.markdown("""## ・推論根拠""")
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            for i, face in enumerate(faces):
                col = columns[i % 3]
                with col:
                    grad_cam_image = GradCam(image, face, model)
                    st.image(grad_cam_image, width=256)


if __name__ == "__main__":
    main()