# ウマ娘　三姉妹　顔分類
ウマ娘のシュヴァルグラン、ヴィルシーナ、ヴィブロスの顔を認識して分類します

Grad-CAMを用いて推論の根拠となっている可能性の高い場所も表示します

![jpg](/img/demo.jpg)

## 環境
* python 3.9.19
* matplotlib 3.8.4
* numpy 1.26.4
* opencv-python 4.10.0
* Pillow 10.4.0
* pytorch 2.4.0
* torchvision 0.19.0
* streamlit 1.37.1

## 実行
```bash
streamlit run run.py
```

## モデルの構造
```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNN                                      [1, 3]                    --
├─Sequential: 1-9                        --                        (recursive)
│    └─Conv2d: 2-1                       [1, 6, 124, 124]          456
├─Sequential: 1-11                       --                        (recursive)
│    └─ReLU: 2-2                         [1, 6, 124, 124]          --
├─Sequential: 1-9                        --                        (recursive)
│    └─MaxPool2d: 2-3                    [1, 6, 62, 62]            --
│    └─Conv2d: 2-4                       [1, 16, 60, 60]           880
├─Sequential: 1-11                       --                        (recursive)
│    └─ReLU: 2-5                         [1, 16, 60, 60]           --
├─Sequential: 1-9                        --                        (recursive)
│    └─MaxPool2d: 2-6                    [1, 16, 30, 30]           --
│    └─Conv2d: 2-7                       [1, 32, 28, 28]           4,640
├─Sequential: 1-11                       --                        (recursive)
│    └─ReLU: 2-8                         [1, 32, 28, 28]           --
├─Sequential: 1-9                        --                        (recursive)
│    └─MaxPool2d: 2-9                    [1, 32, 14, 14]           --
│    └─Conv2d: 2-10                      [1, 64, 12, 12]           18,496
├─Sequential: 1-11                       --                        (recursive)
│    └─ReLU: 2-11                        [1, 64, 12, 12]           --
├─Sequential: 1-9                        --                        (recursive)
│    └─Dropout: 2-12                     [1, 64, 12, 12]           --
├─Flatten: 1-10                          [1, 9216]                 --
├─Sequential: 1-11                       --                        (recursive)
│    └─Linear: 2-13                      [1, 128]                  1,179,776
│    └─ReLU: 2-14                        [1, 128]                  --
│    └─Linear: 2-15                      [1, 64]                   8,256
│    └─ReLU: 2-16                        [1, 64]                   --
│    └─Linear: 2-17                      [1, 3]                    195
==========================================================================================
Total params: 1,212,699
Trainable params: 1,212,699
Non-trainable params: 0
Total mult-adds (M): 17.67
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 1.47
Params size (MB): 4.85
Estimated Total Size (MB): 6.52
==========================================================================================
```