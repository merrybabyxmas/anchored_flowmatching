import os
import subprocess
import argparse
import pandas as pd
import json
from pathlib import Path

def setup_openvid_500(output_directory, limit=500):
    video_folder = os.path.join(output_directory, "video")
    data_folder = os.path.join(output_directory, "data", "train")
    csv_path = os.path.join(data_folder, "OpenVid-1M.csv")
    
    if not os.path.exists(csv_path):
        print("❌ CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 1. CSV 읽기 및 컬럼 확인
    print("📋 Reading CSV and matching captions...")
    df = pd.read_csv(csv_path)
    
    # 가능한 컬럼 이름 후보들 중 실제로 존재하는 것 찾기
    possible_columns = ['text', 'content', 'caption', 'label']
    target_col = None
    for col in possible_columns:
        if col in df.columns:
            target_col = col
            break
            
    if not target_col:
        print(f"❌ 오류: 캡션 컬럼을 찾을 수 없습니다. 존재하는 컬럼: {df.columns.tolist()}")
        return
    else:
        print(f"🔍 Found caption column: '{target_col}'")

    # 2. JSON 생성
    dataset_json = []
    extracted_videos = os.listdir(video_folder)
    
    success_count = 0
    for vid in extracted_videos:
        # OpenVid CSV는 비디오 파일명에 경로가 포함되어 있을 수 있으므로 str.contains 등으로 검색
        # 또는 파일명만 매칭
        match = df[df['video'].str.contains(vid, na=False)]
        
        if not match.empty:
            caption = match.iloc[0][target_col]
            dataset_json.append({
                "video_path": f"video/{vid}",
                "caption": str(caption)
            })
            success_count += 1

    # 3. 최종 저장
    output_json_path = os.path.join(output_directory, "dataset.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 완성! {success_count}개의 데이터가 매칭되었습니다.")
    print(f"📂 파일 위치: {output_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, default="./openvid_data")
    args = parser.parse_args()
    
    setup_openvid_500(args.output_directory)