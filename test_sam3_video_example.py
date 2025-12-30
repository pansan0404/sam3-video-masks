#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM3 Video Predictor Example - 動画で人を検出するスクリプト
"""

import os
import sys
import argparse
import glob
import json
import time
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

# GPU設定
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# コマンドライン引数の解析
parser = argparse.ArgumentParser(description="SAM3 Video Predictor Example")
parser.add_argument("--video", type=str, 
                    default="/home/setup/sam3/Aoba_001.mp4",
                    help="入力動画のパス（MP4ファイルまたはJPEGフレームのフォルダ）")
parser.add_argument("--prompt", type=str, default="person",
                    help="検出するオブジェクトのテキストプロンプト")
parser.add_argument("--frame_idx", type=int, default=0,
                    help="プロンプトを追加するフレームインデックス")
parser.add_argument("--output_dir", type=str, default="video_output",
                    help="出力ディレクトリ")
parser.add_argument("--vis_stride", type=int, default=30,
                    help="可視化するフレームの間隔")
parser.add_argument("--output_video", type=str, default=None,
                    help="出力動画ファイル名（指定すると動画を生成）")
parser.add_argument("--fps", type=float, default=30.0,
                    help="出力動画のFPS")
parser.add_argument("--output_json", type=str, default=None,
                    help="出力JSONファイル名（指定するとマスク情報をJSONに保存）")
parser.add_argument("--output_frame_dir", type=str, default=None,
                    help="フレームごとの出力ディレクトリ（指定すると各フレームのJSONと画像を保存）")
args = parser.parse_args()

print("=" * 60)
print("SAM3 Video Predictor Example")
print("=" * 60)
print(f"動画: {args.video}")
print(f"プロンプト: {args.prompt}")
print(f"フレームインデックス: {args.frame_idx}")

# 出力ディレクトリを作成
os.makedirs(args.output_dir, exist_ok=True)

# 全体の開始時間
total_start_time = time.time()

# モデルをビルド
print("\n1. 動画用モデルをビルドしています...")
model_start_time = time.time()
gpus_to_use = range(torch.cuda.device_count())
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
model_time = time.time() - model_start_time
print(f"✓ モデルのビルドが完了しました（{model_time:.2f}秒）")

# 動画フレームをロード（可視化用）
print("\n2. 動画フレームをロードしています...")
video_path = args.video
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"✓ MP4ファイルから {len(video_frames_for_vis)} フレームをロードしました")
else:
    # JPEGフォルダの場合
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        video_frames_for_vis.sort()
    print(f"✓ JPEGフォルダから {len(video_frames_for_vis)} フレームをロードしました")

# セッションを開始
print("\n3. 動画セッションを開始しています...")
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]
print(f"✓ セッションID: {session_id}")

# テキストプロンプトを追加
print(f"\n4. テキストプロンプト '{args.prompt}' をフレーム {args.frame_idx} に追加...")
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=args.frame_idx,
        text=args.prompt,
    )
)
out = response["outputs"]
# 出力構造から検出数を取得
if 'out_obj_ids' in out:
    num_objects = len(out['out_obj_ids'])
elif 'scores' in out:
    num_objects = len(out['scores'])
else:
    num_objects = 0
print(f"✓ プロンプト追加完了。検出されたオブジェクト数: {num_objects}")
if num_objects > 0 and 'out_probs' in out:
    print(f"  確度例: {out['out_probs'][:min(3, num_objects)]}")

# 最初のフレームを可視化
print("\n5. 最初のフレームを可視化しています...")
plt.close("all")
visualize_formatted_frame_output(
    args.frame_idx,
    video_frames_for_vis,
    outputs_list=[prepare_masks_for_visualization({args.frame_idx: out})],
    titles=[f"SAM 3 - Frame {args.frame_idx}"],
    figsize=(12, 8),
)
first_frame_path = os.path.join(args.output_dir, f"frame_{args.frame_idx:05d}_detection.png")
plt.savefig(first_frame_path, bbox_inches='tight', dpi=150)
print(f"✓ 最初のフレームを保存しました: {first_frame_path}")

# 動画全体に伝播
print("\n6. 動画全体に検出結果を伝播しています...")
def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    frame_count = 0
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = response["frame_index"]
        outputs_per_frame[frame_idx] = response["outputs"]
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  処理中: {frame_count} フレーム完了...")
    return outputs_per_frame

sam_start_time = time.time()
outputs_per_frame_raw = propagate_in_video(predictor, session_id)
sam_time = time.time() - sam_start_time
print(f"✓ 全 {len(outputs_per_frame_raw)} フレームの処理が完了しました（{sam_time:.2f}秒）")

# 動画のサイズを取得（JSON保存と画像保存の両方で使用）
if args.output_json or args.output_frame_dir:
    if isinstance(video_frames_for_vis[0], str):
        sample_img = Image.open(video_frames_for_vis[0])
    else:
        sample_img = Image.fromarray(video_frames_for_vis[0])
    video_width, video_height = sample_img.size

# JSONに保存（単一ファイル、可視化前の生データを使用）
if args.output_json:
    print(f"\n7. マスク情報をJSONに保存しています...")
    json_start_time = time.time()
    
    json_data = {
        "prompt": args.prompt,
        "video_path": args.video,
        "num_frames": len(outputs_per_frame_raw),
        "video_width": video_width,
        "video_height": video_height,
        "timestamp": datetime.now().isoformat(),
        "frames": {}
    }
    
    for frame_idx, frame_outputs in outputs_per_frame_raw.items():
        frame_data = {
            "frame_index": frame_idx,
            "num_objects": 0,
            "objects": []
        }
        
        if 'out_obj_ids' in frame_outputs:
            obj_ids = frame_outputs['out_obj_ids']
            probs = frame_outputs.get('out_probs', [])
            boxes = frame_outputs.get('out_boxes_xywh', [])
            masks = frame_outputs.get('out_binary_masks', [])
            
            frame_data["num_objects"] = len(obj_ids)
            
            for i, obj_id in enumerate(obj_ids):
                # マスクデータを取得（存在する場合）
                mask_data = None
                if i < len(masks) and masks[i] is not None:
                    # マスクをnumpy配列に変換して、小さなサイズにリサイズして保存
                    mask = masks[i]
                    if isinstance(mask, np.ndarray):
                        # マスクを圧縮して保存（サイズを1/4に縮小）
                        mask_small = cv2.resize(mask.astype(np.uint8), 
                                                (mask.shape[1]//4, mask.shape[0]//4))
                        mask_data = mask_small.tolist()
                
                obj_data = {
                    "object_id": int(obj_id),
                    "confidence": float(probs[i]) if i < len(probs) else 0.0,
                    "box": boxes[i].tolist() if i < len(boxes) else None,
                    "mask": mask_data,  # マスクデータを追加
                }
                frame_data["objects"].append(obj_data)
        
        json_data["frames"][str(frame_idx)] = frame_data
    
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    json_time = time.time() - json_start_time
    print(f"✓ JSONを保存しました: {args.output_json}（{json_time:.2f}秒）")
else:
    json_time = 0.0

# フレームごとにJSONと画像を保存
if args.output_frame_dir:
    # 出力ディレクトリを作成
    output_base_dir = args.output_frame_dir
    json_dir = os.path.join(output_base_dir, "json")
    img_dir = os.path.join(output_base_dir, "img")
    img_boundbox_dir = os.path.join(output_base_dir, "img_boundbox")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(img_boundbox_dir, exist_ok=True)
    
    # ステップ1: JSONファイルを保存
    print(f"\n7.5. フレームごとのJSONファイルを保存しています...")
    json_save_start_time = time.time()
    
    total_frames = len(outputs_per_frame_raw)
    json_saved_count = 0
    
    for frame_idx, frame_outputs in outputs_per_frame_raw.items():
        # フレームデータを準備
        frame_data = {
            "prompt": args.prompt,
            "video_path": args.video,
            "frame_index": frame_idx,
            "video_width": video_width,
            "video_height": video_height,
            "timestamp": datetime.now().isoformat(),
            "num_objects": 0,
            "objects": []
        }
        
        if 'out_obj_ids' in frame_outputs:
            obj_ids = frame_outputs['out_obj_ids']
            probs = frame_outputs.get('out_probs', [])
            boxes = frame_outputs.get('out_boxes_xywh', [])
            masks = frame_outputs.get('out_binary_masks', [])
            
            frame_data["num_objects"] = len(obj_ids)
            
            for i, obj_id in enumerate(obj_ids):
                # マスクデータを取得（存在する場合）
                mask_data = None
                if i < len(masks) and masks[i] is not None:
                    # マスクをnumpy配列に変換して、小さなサイズにリサイズして保存
                    mask = masks[i]
                    if isinstance(mask, np.ndarray):
                        # マスクを圧縮して保存（サイズを1/4に縮小）
                        mask_small = cv2.resize(mask.astype(np.uint8), 
                                                (mask.shape[1]//4, mask.shape[0]//4))
                        mask_data = mask_small.tolist()
                
                obj_data = {
                    "object_id": int(obj_id),
                    "confidence": float(probs[i]) if i < len(probs) else 0.0,
                    "box": boxes[i].tolist() if i < len(boxes) else None,
                    "mask": mask_data,  # マスクデータを追加
                }
                frame_data["objects"].append(obj_data)
        
        # JSONファイルを保存
        json_filename = f"frame_{frame_idx:05d}.json"
        json_path = os.path.join(json_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)
        
        json_saved_count += 1
        if json_saved_count % 50 == 0:
            print(f"  処理中: {json_saved_count}/{total_frames} JSONファイル完了...")
    
    json_save_time = time.time() - json_save_start_time
    print(f"✓ JSONファイルを保存しました: {json_dir} ({json_saved_count}ファイル, {json_save_time:.2f}秒)")
    
    # ステップ2: JSONファイルを読み込んで画像を生成
    print(f"\n7.6. JSONファイルから画像を生成しています...")
    img_gen_start_time = time.time()
    
    img_saved_count = 0
    
    # 色のリスト（各オブジェクトに異なる色を割り当て）
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for frame_idx in range(total_frames):
        # JSONファイルを読み込む
        json_filename = f"frame_{frame_idx:05d}.json"
        json_path = os.path.join(json_dir, json_filename)
        
        if not os.path.exists(json_path):
            print(f"  警告: JSONファイルが見つかりません: {json_path}")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            frame_data = json.load(f)
        
        # フレーム画像を取得
        if isinstance(video_frames_for_vis[frame_idx], str):
            frame_img = np.array(Image.open(video_frames_for_vis[frame_idx]))
        else:
            frame_img = video_frames_for_vis[frame_idx].copy()
        
        # 元画像を保存（imgディレクトリ）
        img_filename = f"frame_{frame_idx:05d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        # RGBからBGRに変換（OpenCV用）
        frame_img_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, frame_img_bgr)
        
        # バウンディングボックスを描画した画像を作成
        frame_img_with_boxes = frame_img.copy()
        
        # JSONからバウンディングボックス情報を取得して描画
        for i, obj_data in enumerate(frame_data["objects"]):
            box = obj_data.get("box")
            if box is not None:
                x, y, w, h = box
                # 正規化座標からピクセル座標に変換
                x1 = int(x * video_width)
                y1 = int(y * video_height)
                x2 = int((x + w) * video_width)
                y2 = int((y + h) * video_height)
                
                # バウンディングボックスを描画
                color = colors[i % len(colors)]
                cv2.rectangle(frame_img_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # オブジェクトIDと確度を表示
                obj_id = obj_data.get("object_id", i)
                confidence = obj_data.get("confidence", 0.0)
                label = f"ID:{obj_id} ({confidence:.2f})"
                cv2.putText(frame_img_with_boxes, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # バウンディングボックス付き画像を保存（img_boundboxディレクトリ）
        img_boundbox_filename = f"frame_{frame_idx:05d}.jpg"
        img_boundbox_path = os.path.join(img_boundbox_dir, img_boundbox_filename)
        # RGBからBGRに変換（OpenCV用）
        frame_img_with_boxes_bgr = cv2.cvtColor(frame_img_with_boxes, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_boundbox_path, frame_img_with_boxes_bgr)
        
        img_saved_count += 1
        if img_saved_count % 50 == 0:
            print(f"  処理中: {img_saved_count}/{total_frames} 画像生成完了...")
    
    img_gen_time = time.time() - img_gen_start_time
    frame_save_time = json_save_time + img_gen_time
    print(f"✓ 画像を生成しました:")
    print(f"  元画像: {img_dir} ({img_saved_count}ファイル)")
    print(f"  バウンディングボックス付き: {img_boundbox_dir} ({img_saved_count}ファイル)")
    print(f"  画像生成時間: {img_gen_time:.2f}秒")
    print(f"  合計保存時間: {frame_save_time:.2f}秒")
if not args.output_frame_dir:
    frame_save_time = 0.0

# 結果を可視化
print(f"\n8. 結果を可視化しています（{args.vis_stride}フレームごと）...")
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame_raw)

vis_frame_stride = args.vis_stride
saved_frames = []
for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
    plt.close("all")
    visualize_formatted_frame_output(
        frame_idx,
        video_frames_for_vis,
        outputs_list=[outputs_per_frame],
        titles=[f"SAM 3 Dense Tracking - Frame {frame_idx}"],
        figsize=(12, 8),
    )
    frame_path = os.path.join(args.output_dir, f"frame_{frame_idx:05d}_tracking.png")
    plt.savefig(frame_path, bbox_inches='tight', dpi=150)
    saved_frames.append(frame_path)
    print(f"  ✓ フレーム {frame_idx} を保存: {frame_path}")

print("\n" + "=" * 60)
print("動画処理が完了しました！")
print("=" * 60)
print(f"出力ディレクトリ: {args.output_dir}")
print(f"保存されたフレーム数: {len(saved_frames)}")
if 'out_obj_ids' in out:
    print(f"検出されたオブジェクト数: {len(out['out_obj_ids'])}")
elif 'scores' in out:
    print(f"検出されたオブジェクト数: {len(out['scores'])}")
else:
    print(f"検出されたオブジェクト数: 0")
if args.output_json:
    print(f"JSONファイル: {args.output_json}")
if args.output_frame_dir:
    print(f"フレームごとの出力: {args.output_frame_dir}")

# 動画レンダリング時間とJSON保存時間の初期化
render_time = 0.0
json_time = 0.0
frame_save_time = 0.0

# 動画を生成
if args.output_video:
    print(f"\n9. 動画を生成しています...")
    print(f"   FPS: {args.fps}")
    
    render_start_time = time.time()
    
    # 動画のサイズを取得
    if isinstance(video_frames_for_vis[0], str):
        sample_img = Image.open(video_frames_for_vis[0])
    else:
        sample_img = Image.fromarray(video_frames_for_vis[0])
    video_width, video_height = sample_img.size
    
    # 動画ライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        args.output_video,
        fourcc,
        args.fps,
        (video_width, video_height)
    )
    
    if not video_writer.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {args.output_video}")
    else:
        frame_count = 0
        total_frames = len(outputs_per_frame)
        
        for frame_idx in range(total_frames):
            # フレームを可視化（既存の関数を使用）
            plt.close("all")
            visualize_formatted_frame_output(
                frame_idx,
                video_frames_for_vis,
                outputs_list=[outputs_per_frame],
                titles=[f"Frame {frame_idx}"],
                figsize=(video_width/100, video_height/100),
            )
            
            # matplotlibの図を画像配列に変換
            fig = plt.gcf()
            fig.canvas.draw()
            
            # matplotlibのバージョンに応じたAPIを使用
            try:
                # 新しいAPI
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # RGBAからRGBに変換
                buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
            except AttributeError:
                # 古いAPI
                try:
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                except AttributeError:
                    # 別の方法
                    from io import BytesIO
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.read(), dtype=np.uint8)
                    buf = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            
            # サイズを調整（必要に応じてリサイズ）
            if len(buf.shape) == 3 and buf.shape[:2] != (video_height, video_width):
                buf = cv2.resize(buf, (video_width, video_height))
            
            # BGRに変換（OpenCV用）
            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            plt.close(fig)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  処理中: {frame_count}/{total_frames} フレーム...")
        
        video_writer.release()
        render_time = time.time() - render_start_time
        print(f"✓ 動画を保存しました: {args.output_video}")
        print(f"  総フレーム数: {frame_count}")
        print(f"  動画サイズ: {video_width}x{video_height}")
        print(f"  動画レンダリング時間: {render_time:.2f}秒")

# 全体の処理時間を表示
total_time = time.time() - total_start_time
print("\n" + "=" * 60)
print("処理時間のサマリー")
print("=" * 60)
print(f"モデルロード時間: {model_time:.2f}秒")
print(f"SAM3処理時間（トラッキング）: {sam_time:.2f}秒")
if args.output_json:
    print(f"JSON保存時間: {json_time:.2f}秒")
if args.output_frame_dir:
    print(f"フレームごとの保存時間: {frame_save_time:.2f}秒")
if args.output_video:
    print(f"動画レンダリング時間: {render_time:.2f}秒")
print(f"合計処理時間: {total_time:.2f}秒")
print("=" * 60)

