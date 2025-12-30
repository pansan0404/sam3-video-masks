# SAM3 Video Masks

SAM3 (Segment Anything Model 3) によるビデオインスタンストラッキング結果を保存するリポジトリです。

## Description

このリポジトリには、SAM3を使用してビデオから生成されたインスタンスセグメンテーションとトラッキング結果が含まれています。

### データ構造

```
output/
├── json/          # 各フレームのマスク情報（JSON形式）
│   ├── frame_00000.json
│   ├── frame_00001.json
│   └── ...
├── img/           # 元画像
│   ├── frame_00000.jpg
│   ├── frame_00001.jpg
│   └── ...
└── img_boundbox/  # バウンディングボックス付き画像
    ├── frame_00000.jpg
    ├── frame_00001.jpg
    └── ...
```

### JSON形式

各JSONファイルには以下の情報が含まれています：

```json
{
  "prompt": "person",
  "video_path": "/path/to/video.mp4",
  "frame_index": 0,
  "video_width": 1920,
  "video_height": 1080,
  "timestamp": "2025-12-30T15:55:10.123456",
  "num_objects": 3,
  "objects": [
    {
      "object_id": 0,
      "confidence": 0.9569892287254333,
      "box": [0.432, 0.580, 0.106, 0.348],
      "mask": [[0, 0, 1, ...], ...]
    },
    ...
  ]
}
```

### 使用方法

このデータは、3D Gaussian Splattingなどの下流タスクで使用できます。

### 生成方法

以下のコマンドで生成されました：

```bash
python test_sam3_video_example.py \
  --video /path/to/video.mp4 \
  --prompt "person" \
  --output_frame_dir output
```

## 修論用説明

本リポジトリは、SAM3を用いたビデオインスタンストラッキングの結果を保存しています。各フレームごとに、検出されたオブジェクトのマスク、バウンディングボックス、信頼度スコアなどの情報をJSON形式で保存しています。これらのデータは、3D Gaussian Splattingなどの3D再構成タスクにおけるインスタンス対応の訓練に活用できます。

