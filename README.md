<div align="center">

English | [简体中文](./README-zh_CN.md)


<h1>DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception</h1>

Official PyTorch implementation of [DocLayout-YOLO](https://arxiv.org/abs/2410.12628).

[![arXiv](https://img.shields.io/badge/arXiv-2405.14458-b31b1b.svg)](https://arxiv.org/abs/2410.12628) [![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97-Online%20Demo-yellow)](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Models%20and%20Data-yellow)](https://huggingface.co/collections/juliozhao/doclayout-yolo-670cdec674913d9a6f77b542)

</div>
    
## Abstract

> We present DocLayout-YOLO, a real-time and robust layout detection model for diverse documents, based on YOLO-v10. This model is enriched with diversified document pre-training and structural optimization tailored for layout detection. In the pre-training phase, we introduce Mesh-candidate BestFit, viewing document synthesis as a two-dimensional bin packing problem, and create a large-scale diverse synthetic document dataset, DocSynth-300K. In terms of model structural optimization, we propose a module with Global-to-Local Controllability for precise detection of document elements across varying scales. 


<p align="center">
  <img src="assets/comp.png" width=52%>
  <img src="assets/radar.png" width=44%> <br>
</p>

## News 🚀🚀🚀

**2024.10.25** 🎉🎉  **Mesh-candidate Bestfit** code is released. Mesh-candidate Bestfit is an automatic pipeline which can synthesize large-scale, high-quality, and visually appealing document layout detection dataset. Tutorial and example data are available in [here](./mesh-candidate_bestfit).

**2024.10.23** 🎉🎉  **DocSynth300K dataset** is released on [🤗Huggingface](https://huggingface.co/datasets/juliozhao/DocSynth300K), DocSynth300K is a large-scale and diverse document layout analysis pre-training dataset, which can largely boost model performance.

**2024.10.21** 🎉🎉  **Online demo** available on [🤗Huggingface](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO).

**2024.10.18** 🎉🎉  DocLayout-YOLO is implemented in **[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)** for document context extraction.

**2024.10.16** 🎉🎉  **Paper** now available on [ArXiv](https://arxiv.org/abs/2410.12628).   


## Quick Start

[Online Demo](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO) is now available. For local development, follow steps below:

### 1. Environment Setup

Follow these steps to set up your environment:

```bash
conda create -n doclayout_yolo python=3.10
conda activate doclayout_yolo
pip install -e .
```

**Note:** If you only need the package for inference, you can simply install it via pip:

```bash
pip install doclayout-yolo
```

### 2. Prediction

You can make predictions using either a script or the SDK:

- **Script**

  Run the following command to make a prediction using the script:

  ```bash
  python demo.py --model path/to/model --image-path path/to/image
  ```

- **SDK**

  Here is an example of how to use the SDK for prediction:

  ```python
  import cv2
  from doclayout_yolo import YOLOv10

  # Load the pre-trained model
  model = YOLOv10("path/to/provided/model")

  # Perform prediction
  det_res = model.predict(
      "path/to/image",   # Image to predict
      imgsz=1024,        # Prediction image size
      conf=0.2,          # Confidence threshold
      device="cuda:0"    # Device to use (e.g., 'cuda:0' or 'cpu')
  )

  # Annotate and save the result
  annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
  cv2.imwrite("result.jpg", annotated_frame)
  ```


We provide model fine-tuned on **DocStructBench** for prediction, **which is capable of handing various document types**. Model can be downloaded from [here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/tree/main) and example images can be found under ```assets/example```.

<p align="center">
  <img src="assets/showcase.png" width=100%> <br>
</p>


**Note:** For PDF content extraction, please refer to [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit/tree/main) and [MinerU](https://github.com/opendatalab/MinerU).

**Note:** Thanks to [NielsRogge](https://github.com/NielsRogge), DocLayout-YOLO now supports implementation directly from 🤗Huggingface, you can load model as follows:

```python
filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(filepath)
```

or directly load using ```from_pretrained```:

```python
model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
```

more details can be found at [this PR](https://github.com/opendatalab/DocLayout-YOLO/pull/6).

**Note:** Thanks to [luciaganlulu](https://github.com/luciaganlulu), DocLayout-YOLO can perform batch inference and prediction. Instead of passing single image into ```model.predict``` in ```demo.py```, pass a **list of image path**. Besides, due to batch inference is not implemented before ```YOLOv11```, you should manually change ```batch_size``` in [here](doclayout_yolo/engine/model.py#L431).

## DocSynth300K Dataset

<p align="center">
  <img src="assets/docsynth300k.png" width=100%>
</p>

### Data Download

Use following command to download dataset(about 113G):

```python
from huggingface_hub import snapshot_download
# Download DocSynth300K
snapshot_download(repo_id="juliozhao/DocSynth300K", local_dir="./docsynth300k-hf", repo_type="dataset")
# If the download was disrupted and the file is not complete, you can resume the download
snapshot_download(repo_id="juliozhao/DocSynth300K", local_dir="./docsynth300k-hf", repo_type="dataset", resume_download=True)
```

### Data Formatting & Pre-training

If you want to perform DocSynth300K pretraining, using ```format_docsynth300k.py``` to convert original ```.parquet``` format into ```YOLO``` format. The converted data will be stored at ```./layout_data/docsynth300k```.

```bash
python format_docsynth300k.py
```

To perform DocSynth300K pre-training, use this [command](assets/script.sh#L2). We default use 8GPUs to perform pretraining. To reach optimal performance, you can adjust hyper-parameters such as ```imgsz```, ```lr``` according to your downstream fine-tuning data distribution or setting.

**Note:** Due to memory leakage in YOLO original data loading code, the pretraining on large-scale dataset may be interrupted unexpectedly, use ```--pretrain last_checkpoint.pt --resume``` to resume the pretraining process.

## Training and Evaluation on Public DLA Datasets

### Data Preparation

1. specify  the data root path

Find your ultralytics config file (for Linux user in ```$HOME/.config/Ultralytics/settings.yaml)``` and change ```datasets_dir``` to project root path.

2. Download prepared yolo-format D4LA and DocLayNet data from below and put to ```./layout_data```:

| Dataset | Download |
|:--:|:--:|
| D4LA | [link](https://huggingface.co/datasets/juliozhao/doclayout-yolo-D4LA) |
| DocLayNet | [link](https://huggingface.co/datasets/juliozhao/doclayout-yolo-DocLayNet) |

the file structure is as follows:

```bash
./layout_data
├── D4LA
│   ├── images
│   ├── labels
│   ├── test.txt
│   └── train.txt
└── doclaynet
    ├── images
    ├── labels
    ├── val.txt
    └── train.txt
```

### Training and Evaluation

Training is conducted on 8 GPUs with a global batch size of 64 (8 images per device). The detailed settings and checkpoints are as follows:

| Dataset | Model | DocSynth300K Pretrained? | imgsz | Learning rate | Finetune | Evaluation | AP50 | mAP | Checkpoint |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| D4LA | DocLayout-YOLO | &cross; | 1600 | 0.04 | [command](assets/script.sh#L5) | [command](assets/script.sh#L11) | 81.7 | 69.8 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-from_scratch) |
| D4LA | DocLayout-YOLO | &check; | 1600 | 0.04 | [command](assets/script.sh#L8) | [command](assets/script.sh#L11) | 82.4 | 70.3 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained) |
| DocLayNet | DocLayout-YOLO | &cross; | 1120 | 0.02 | [command](assets/script.sh#L14) | [command](assets/script.sh#L20) | 93.0 | 77.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-from_scratch) |
| DocLayNet | DocLayout-YOLO | &check; | 1120 | 0.02 | [command](assets/script.sh#L17) | [command](assets/script.sh#L20) | 93.4 | 79.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained) |

The DocSynth300K pretrained model can be downloaded from [here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocSynth300K-pretrain). Change ```checkpoint.pt``` to the path of model to be evaluated during evaluation.


## 한국 시험지 레이아웃 가이드 (K-Exam)

본 섹션은 한국어 시험지(중/고교 지필평가 등)에 특화된 최소 라벨 스키마, 데이터 준비, 증강과 전이학습, 평가·후처리, 소량 데이터 전략까지 한 번에 실행할 수 있도록 정리된 가이드입니다. 실무에서 빠르게 돌릴 수 있도록 전체 스크립트/예시를 모두 제공합니다.

### 1) 한국 시험지용 라벨 스키마 정의

시험지에 특화된 최소·필수 블록 위주로 시작하고, 세부 분리는 후처리 규칙(라인/번호 탐지)로 해결하는 것이 안정적입니다.

- 10개 권장 클래스
  1. header (학교/학기/과목/점수 영역)
  2. passage (공통 지문/자료)
  3. question_stem (발문)
  4. choice_block (선지 묶음 전체)
  5. choice_item (①~⑤ 등 개별 선지)
  6. diagram (그림)
  7. table (표)
  8. formula_block (수식 블록)
  9. answer_area (서술/단답 답안 칸)
  10. footer (페이지 번호/유의사항)

- 메모
  - 초판에서는 choice_block만 두고, 내부 ①~⑤는 후처리(원 안 숫자/‘ㄱ, ㄴ’ 등 패턴 인식)로 분리해도 됩니다. (클래스 수↓ → 라벨 일관성↑)
  - 필요 시 question_number를 별도 클래스로 두면 문항 단위 클러스터링이 쉬워집니다(번호→가까운 stem/choice 매칭).

### 2) 데이터 준비(라벨링 → YOLO 포맷)

- 라벨링 도구: Label Studio, CVAT, Roboflow 등(박스+클래스)
- 해상도: 300DPI 스캔 원본 또는 PDF→이미지(긴 변 2000~3000px 권장)
- 포맷: YOLO 텍스트 라벨(클래스 ID, 정규화 bbox: xc yc w h)
- 디렉터리 구조(이 리포의 D4LA/DocLayNet 예시와 동일하게 구성 권장)

```bash
./layout_data/kexam
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
```

- data.yaml 예시(파일 경로: `layout_data/kexam/kexam.yaml`)

```yaml
path: ./layout_data/kexam
train: images/train
val: images/val
names:
  0: header
  1: passage
  2: question_stem
  3: choice_block
  4: choice_item
  5: diagram
  6: table
  7: formula_block
  8: answer_area
  9: footer
```

Important: 본 리포의 `train.py`/`val.py`는 `--data` 인자에 `.yaml` 확장자를 자동으로 붙입니다. 실행 시에는 확장자를 제외한 베이스 경로만 넘기세요. 예: `--data layout_data/kexam/kexam` (O), `--data layout_data/kexam/kexam.yaml` (X)

- PDF → 이미지 변환(예시 스크립트)

```bash
python /Users/jiwon/Workspace/DocLayout-YOLO/scripts/pdf_to_images.py \
  "/Users/jiwon/Workspace/DocLayout-YOLO/pdfs" \
  -o "/Users/jiwon/Workspace/DocLayout-YOLO/layout_data/kexam" \
  --fmt png --dpi 300 --max-side 3000 --workers 4 --optimize
```

- Label Studio/CVAT 내보내기 후 train/val 분할(예시 스크립트)

```bash
python /Users/jiwon/Workspace/DocLayout-YOLO/scripts/train-val-splitter.py
```

스크립트 상단 변수(`export_dir`)를 실제 내보내기 경로로 맞춘 뒤 실행하세요. 이미지/라벨 파일명이 1:1로 매칭되도록 관리하는 것이 중요합니다.

### 3) 도메인 증강(한국 시험지에 잘 먹히는 것)

- 복사/스캔 아티팩트: 가우시안/모션 블러, JPEG 압축, 임의 그레이닝
- 휘어짐/그림자: 밝기/대비/섀도우 그라디언트 합성
- 약간의 기울기/컷오프: ±1~2.5° 회전, 테두리 잘림
- 흑백/저채도 전환(학교 프린트 느낌)
- 폰트 다양화: 한글 본문/번호(①②…/ㄱㄴㄷ) 합성(선지 검출 강건성↑)
- 수식/표 밀집 합성(수학/과학 과목 대응)

DocLayout-YOLO는 합성 데이터 사전학습(“Mesh-candidate BestFit” 기반 DocSynth-300K)을 강조합니다. 한국 시험지 템플릿을 흉내 낸 합성 미니셋을 곁들이면 소량 실데이터로도 성능이 빠르게 붙습니다.

### 4) 학습 시작(전이학습)

- 환경/추론 설치(선택): 상단 Quick Start 참고 또는 아래 간단 설치

```bash
conda create -n doclayout_yolo python=3.10 -y
conda activate doclayout_yolo
pip install -e /Users/jiwon/Workspace/DocLayout-YOLO
```

- 제공 체크포인트(예: DocStructBench) 경로 예시: `/Users/jiwon/Workspace/DocLayout-YOLO/models/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt`

- 단일/소수 GPU 전이학습 실행 예시(이 리포의 `train.py` 인터페이스 기준)

```bash
# 사전학습 체크포인트 전이학습
python /Users/jiwon/Workspace/DocLayout-YOLO/train.py \
  --data layout_data/kexam/kexam \
  --model m \
  --epoch 80 \
  --batch-size 8 \
  --image-size 1280 \
  --lr0 0.01 \
  --device 0 \
  --workers 4 \
  --val 1 \
  --val-period 1 \
  --project /Users/jiwon/Workspace/DocLayout-YOLO/dataset/kexam \
  --pretrain /Users/jiwon/Workspace/DocLayout-YOLO/models/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt

# 스크래치(사전학습 없이, 모델 구조: yolov10m)
python /Users/jiwon/Workspace/DocLayout-YOLO/train.py \
  --data layout_data/kexam/kexam \
  --model m \
  --epoch 80 \
  --batch-size 8 \
  --image-size 1600 \
  --lr0 0.02 \
  --device 0 \
  --workers 4 \
  --val 1 \
  --val-period 1 \
  --project /Users/jiwon/Workspace/DocLayout-YOLO/dataset/kexam
```

Hints

- 한국 시험지는 소물체(번호/원형숫자/수식)가 많아 `--image-size 1280~1600`을 권장합니다.
- 여러 GPU 사용 시 `--device 0,1,2,...` 형태로 지정하고 전역 배치를 늘리면 수렴이 안정적입니다.

### 5) 평가/후처리(문항 단위로 묶기)

- 평가(mAP@50/50:95 + 클래스별 F1), 특히 리콜 확인 권장. 이 리포의 `val.py`로 바로 평가 가능합니다.

```bash
python /Users/jiwon/Workspace/DocLayout-YOLO/val.py \
  --data layout_data/kexam/kexam \
  --model /Users/jiwon/Workspace/DocLayout-YOLO/dataset/kexam/<실행_폴더>/weights/best.pt \
  --batch-size 8 \
  --device 0
```

- 문항 재구성(레이아웃 → 문항 단위) 규칙 예시
  1. question_number(또는 stem bbox) 중심으로 가까운 choice_block/diagram/table 매칭(거리/정렬)
  2. 다단(2-column) 시험지는 세로 공백 프로파일로 열 분할 후, 각 열 내부에서 y-좌표 정렬
  3. choice_block만 검출했다면 내부를 OCR/패턴으로 ‘①②…’ 또는 ‘ㄱㄴㄷ’ 슬라이스
  4. NMS는 클래스별로 수행, choice_item은 IoU를 낮춰 과잉 억제 방지

- 간단한 문항 그룹핑 파이썬 예시(개념용)

```python
from typing import List, Tuple

# bbox: (xc, yc, w, h, score, cls)
Box = Tuple[float, float, float, float, float, int]

def iou_xywh(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a[0] - a[2]/2, a[1] - a[3]/2, a[2], a[3]
    bx, by, bw, bh = b[0] - b[2]/2, b[1] - b[3]/2, b[2], b[3]
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = aw*ah + bw*bh - inter + 1e-6
    return inter / union

def group_by_question(stems: List[Box], others: List[Box], x_weight: float = 0.5) -> List[dict]:
    groups = []
    stems_sorted = sorted(stems, key=lambda b: (b[1], b[0]))
    for s in stems_sorted:
        sx, sy = s[0], s[1]
        bucket = {"stem": s, "attached": []}
        for o in others:
            ox, oy = o[0], o[1]
            # 거리 가중(열 정렬 우선) + y 우선 정렬
            d = abs(oy - sy) + x_weight * abs(ox - sx)
            # stem 하단 근처/우측 근처 우선(간단한 조건)
            if oy >= sy - 0.05:
                bucket["attached"].append((d, o))
        bucket["attached"].sort(key=lambda t: (t[0], -t[1][4]))  # 거리→score 순
        groups.append(bucket)
    return groups
```

### 6) 데이터가 적을 때의 팁

- 도메인 합성: 학교명/학기/배점/번호/원형숫자/수식/표 등을 포함한 합성 시험지 수백~수천 장 생성 → 실데이터 50~200페이지와 혼합 미세조정
- 프리징: 백본 일부 고정, 헤드만 미세조정(과적합 방지)
- 타 데이터 워밍업: DocLayNet 일부 클래스를 한국 시험지 라벨셋에 맵핑해 먼저 학습 → 이후 한국 시험지 소량 파인튜닝. DocLayNet은 COCO 포맷으로 8만+ 페이지 제공

### 7) 자주 막히는 포인트

- 이미지 크기(imgsz): 1120~1600 사용 예가 많고, 32의 배수로 맞추는 관행. 너무 작으면 번호/수식 소실
- 클래스 과도 분할: 초기에는 stem/choice_block/diagram/table/passage 위주로 시작 후 필요 시 확장
- 특수기호: 원 안 숫자·ㄱㄴㄷ·수식은 난이도↑ → 증강/합성으로 충분히 노출

### 8) 최소 실행 체크리스트

1. 라벨셋(6~10종) 결정 → 100~300페이지 라벨링
2. YOLO 포맷으로 정리(`layout_data/kexam/{images,labels}/{train,val}`)
3. 공개 체크포인트 로드 후 `--image-size 1280~1600`, 가능한 배치로 50~80 epoch 파인튜닝
4. `val.py`로 클래스별 리콜 확인 → 누락 많은 클래스 위주로 증강/라벨 보강
5. 후처리 규칙으로 문항 단위 그룹핑(번호↔발문↔선지 매칭)


## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [YOLO-v10](https://github.com/lyuwenyu/RT-DETR).

Thanks for their great work!

## Star History

If you find our project useful, please add a "star" to the repo. It's exciting to us when we see your interest, which keep us motivated to continue investing in the project!

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=opendatalab/DocLayout-YOLO&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=opendatalab/DocLayout-YOLO&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=opendatalab/DocLayout-YOLO&type=Date"
  />
</picture>

## Citation

```bibtex
@misc{zhao2024doclayoutyoloenhancingdocumentlayout,
      title={DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception}, 
      author={Zhiyuan Zhao and Hengrui Kang and Bin Wang and Conghui He},
      year={2024},
      eprint={2410.12628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.12628}, 
}

@article{wang2024mineru,
  title={MinerU: An Open-Source Solution for Precise Document Content Extraction},
  author={Wang, Bin and Xu, Chao and Zhao, Xiaomeng and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Xu, Rui and Liu, Kaiwen and Qu, Yuan and Shang, Fukai and others},
  journal={arXiv preprint arXiv:2409.18839},
  year={2024}
}

```
