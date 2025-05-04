import os
import re
import glob
import argparse
import cv2

def extract_max_value(filename):
    """파일 이름에서 max_ 다음에 오는 숫자 값을 추출합니다."""
    match = re.search(r'max_(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def create_yolo_label(output_path, class_id, bbox):
    """YOLO 형식의 라벨 파일을 생성합니다."""
    with open(output_path, 'w') as f:
        x_center, y_center, width, height = bbox
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def get_temp_from_bgr(bgr):
    i = lut_map.get(tuple(int(v) for v in bgr), None)
    if i is not None:
        return i / 255.0 * (60 - 10) + 10
    else:
        return None  # LUT에 없으면 None 반환
        
def auto_labeling(image_dir, output_dir, threshold, bbox, class_id=0, class_name='A'):
    # JET 컬러맵 LUT 생성
    gray = np.arange(256, dtype=np.uint8)
    jet_lut = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    lut_map = {tuple(jet_lut[i,0]): i for i in range(256)}

    
    """이미지 파일을 자동으로 라벨링합니다."""
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 검색 (jpg, png 확장자)
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    abnormal_count = 0
    normal_count = 0
    
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        #max_value = extract_max_value(filename)
        
        #opencv로 이미지 로드 후 최고 온도 추출
        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv_image[:,:,2]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(v_channel)
        x, y = min_loc
        bgr = img[y, x]
        max_value = get_temp_from_bgr(bgr)
        
        if max_value is not None:
            # 라벨 파일 경로 생성 (이미지와 같은 이름, 확장자만 .txt로)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(output_dir, label_filename)
            
            # 임계값 이상이면 지정된 클래스로 라벨링
            if max_value >= threshold:
                create_yolo_label(label_path, 0, bbox)
                abnormal_count += 1
                
            else:
                create_yolo_label(label_path, 1, bbox)
                normal_count +=1    
    
    print(f"라벨링 완료: 총 {len(image_files)}개 이미지 중 {abnormal_count}개가 '비정상' 클래스로, {normal_count}개가 '정상' 클래스로 라벨링되었습니다.")
    return len(image_files)



# 메인 실행 부분
if __name__ == "__main__":
    # 명령행 인자를 먼저 확인
    parser = argparse.ArgumentParser(description='YOLO 오토라벨링 도구')

    parser.add_argument('--image_dir', type=str, help='이미지가 있는 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, help='라벨 파일을 저장할 디렉토리 경로')
    parser.add_argument('--threshold', type=float, help='온도값 판별 임계값')
    parser.add_argument('--class_id', type=int, default=0, help='클래스 ID (기본값: 0)')
    parser.add_argument('--class_name', type=str, default='A', help='클래스 이름 (기본값: A)')
    parser.add_argument('--x_center', type=float, default=0.5, help='바운딩 박스 중심 x 좌표 (0~1)')
    parser.add_argument('--y_center', type=float, default=0.5, help='바운딩 박스 중심 y 좌표 (0~1)')
    parser.add_argument('--width', type=float, default=0.8, help='바운딩 박스 너비 (0~1)')
    parser.add_argument('--height', type=float, default=0.8, help='바운딩 박스 높이 (0~1)')
    
    args = parser.parse_args()
    


    bbox = (args.x_center, args.y_center, args.width, args.height)
    auto_labeling(args.image_dir, args.output_dir, args.threshold, bbox, args.class_id, args.class_name)

