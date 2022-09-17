import cv2
import math
from gtts import gTTS
import os
import playsound
from PIL import Image
import pytesseract
import numpy as np
import json #종우-종연 코드에 이미 있으니 합칠 땐 지울 것
import requests #HTTP 호출 모듈
from urllib.parse import urlencode #URL에 대한 모듈
import xmltodict #xml파일을 dic으로 변환
import config #구성파일구문분석기를 위한 모듈 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import re
import RPi_I2C_driver #LCD전광판 제어를 위한 라이브러리
import multiprocessing #사용자가 탑승을 원하는 버스 번호를 휴대폰으로 받고 라즈베리 파이로 전송하는 작업과
#카메라로 이미지를 처리하는 작업을 동시에 진행하기 위해 필요한 라이브러리
import torch
import torchvision
import pandas as pd
#from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

kernel=np.ones((3,3), dtype=np.uint8)
#pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#mask rcnn 기본 가중치불러오기

#mask rcnn 학습 가중치 불러오기
def get_Yolov5_weight():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 모델 가중치불러오기
    #model = torch.load('D:\\gong\\mrnnside3.9modelmade1 (1).pth',map_location=torch.device('cpu')) #가중치위치로 경로수정
    model = torch.hub.load('/home/pi/Desktop/file/yolov5-master', 'custom', path='/home/pi/Desktop/file/best (1).pt', source='local')
    model.eval()

# move model to the right device
    model.to(device)
    
    return model

#실제 이미지 세그맨테이션
def img_segmentation(img, model,xxxx):
    
    img_list = [] #반환할 리스트이미지

    with torch.no_grad(): #학습모드로설정
        prediction = model([img])

    results = model(img)

    img_list = []

    for a,i in enumerate(results.xyxy):
        for j in i:
            [x1,y1,x2,y2,conf,classes] = j
            x1 = round(x1.item())
            y1 = round(y1.item())
            x2 = round(x2.item())
            y2 = round(y2.item())

            crop = img[y1:y2,x1:x2]
            img_list.append(crop)

    return img_list

# 이미지 리스트 받아서 크기 순으로 정렬후 이미지 리스트 반환
def distance_rearrange(img_list):
    global img_list_rearranged #정렬된 이미지 목록을 저장할 전역 변수 리스트
    global distance_list_rearranged #정렬된 거리 값 목록을 저장할 전역 변수 리스트
    img_list_rearranged=[] #각 리스트를 빈 리스트로 초기화
    distance_list_rearranged=[]
    distance_constant=45000 #거리 상수 값은 카메라 위치에 따라 선정, 예시: 10
    cross_img={}# 딕셔너리 선언
    for img in img_list: # 함수에서 인자로 받은 이미지 리스트를 for문을 돌린다
        img_height,img_width,_= img.shape # 각 이미지 파일의 높이와 가로 길이를 받는다
        cross_img[img_width]=img
    sorted_img_dict = sorted(cross_img.items(), key = lambda item:item[0],reverse=1)# 딕셔너리 키(이미지 넓이) 기준으로 내림차순 정렬하고 딕셔너리로 저장
    for (key, value) in sorted_img_dict: #정렬된 이미지 딕셔너리의 (키, 값)을 반복
        distance_list_rearranged.append(round(distance_constant/key)/10) #넓이에 거리 상수값을 곱한 후 거리 값 리스트에 저장
        img_list_rearranged.append(value) #넓이에 거리 상수값을 곱한 후 거리 값 리스트에 저장
    
def mode(arr):#최빈값구하기
  max_count = 0
  mod_num = 0
  counter = set(arr)
  for c in counter:
    if max_count < arr.count(c):
      max_count = arr.count(c)
      mod_num = c
  return mod_num


def img2string(img_file):
        
    try:
    
        img_resize = cv2.resize(img_file,(300, 200),interpolation=cv2.INTER_AREA)

        img_resize = cv2.copyMakeBorder(img_resize, top=10, bottom=10, left=10, right=10, borderType= cv2.BORDER_CONSTANT, value=(255,255,255))

        img_resized_width = img_resize.shape[1]

        img_resized_height = img_resize.shape[0]

        img_resized_size = img_resized_width * img_resized_height

        target_img = img_resize.copy()

        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

        img_gaus = cv2.GaussianBlur(img_gray, (3,3), 0)

        ret, otsu = cv2.threshold(img_gaus, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contours_dict= []

        for idx,cnt in enumerate(contours):

            x, y, width, height = cv2.boundingRect(cnt) # 윤곽선을 감싸는 박스에서 x, y , 가로, 세로 정보 가져옴

            contours_dict.append({
                'idx':idx,
                'contour': cnt,
                'contours_size': cv2.contourArea(cnt),
                'x': x,
                'y': y,
                'w': width,
                'h': height,
                'hir':hierarchy[0][idx]
            }) # 순서, 윤곽선, 윤곽선 넓이, 왼쪽 위점(x,y), 세로, 가로, 계층을 딕셔너리 형태로 리스트에 저장 

        condition_list_1 = [] # 번호 가까운 윤곽선을 찾아 인덱스값 받음, 1조건 (넓이 + 세로 / 가로 크기 비교) 

        for i in contours_dict:
            
            i_size = i['w'] * i['h']

            if i_size < img_resized_size * 0.01 or i_size > img_resized_size * 0.3 or i['w'] > i['h']  or i['h']/i['w'] > 5.5: 
                # 크기가 전체 크기보다 너무 작거나/ 너무 클 경우, 또는 가로보다 세로가 큰 경우 제외
                continue

            condition_list_1.append(i['idx'])
            
        condition_set_1 = set(condition_list_1) # 중복 제외

        parent_hir = [] # 계층 성분 중 부모계층 정보를 얻어줌

        condition_list_2 = [] # 두번째 조건 만족시키는 인덱스 (조건: 계층 정보가 같은 것 끼리)

        for i in condition_set_1:

            hir = contours_dict[i]['hir']

            parent_hir.append(hir[-1])

        many_time = mode(parent_hir) # 최빈값을 얻어옴 # 최빈값을 얻어옴

        cnt_size = [] # 윤곽선 넓이 얻어줌

        for i in condition_set_1:

            if contours_dict[i]['hir'][-1] != many_time: # 최빈값이 아니면 무시
                continue

            cnt_size.append(contours_dict[i]['contours_size']) # 해당 계층의 윤곽선의 넓이를 찾음

            condition_list_2.append(i) 
            
        last_condtion = {} # 마지막 조건 만족시키는 것( 윤곽선 넓이 / 박스 넓이, 박스안에 숫자가 아닌 다른 오차 무시)

        for i in condition_list_2:

            if contours_dict[i]['contours_size'] / (contours_dict[i]['w']*contours_dict[i]['h']) < 0.3: 
                continue # 윤곽선을 감싸는 박스와 윤곽선의 비율이 너무 작으면 무시
            last_condtion[contours_dict[i]['idx']] = contours_dict[i]['x'] # 인덱스 : 좌표로 얻기

        sorted_idx_by_x = list(dict(sorted(last_condtion.items(), key = lambda item: item[1])).keys()) # x좌표 오름차순으로 정렬후 인덱스 얻어옴

        white_board = np.zeros((img_resized_height, img_resized_width, 3), dtype=np.uint8)

        white_board[:] = (255, 255, 255) # 전체 공간을 흰 색으로 채우기

        for i in sorted_idx_by_x:
            x, dx = contours_dict[i]['x'] , contours_dict[i]['x']+ contours_dict[i]['w']
        
            y, dy = contours_dict[i]['y'] , contours_dict[i]['y']+ contours_dict[i]['h']

            crop = target_img[y:dy, x:dx]   

            white_board[y:dy, x:dx] = crop

        first = sorted_idx_by_x[0]

        end = sorted_idx_by_x[-1]

        if contours_dict[first]['h'] > contours_dict[end]['h']:
            a = 10
            b = 0
        elif contours_dict[first]['h'] < contours_dict[end]['h']: 
            a = 0
            b = 10

        else:
            a = 0
            b = 0

        num = 25

        x1, y1 = contours_dict[first]['x'] - num, contours_dict[first]['y'] - num + a

        x2, y2 = contours_dict[end]['x'] + contours_dict[end]['w'] + num, contours_dict[end]['y'] - num + b

        x3, y3 = contours_dict[end]['x'] + contours_dict[end]['w'] + num, contours_dict[end]['y'] + contours_dict[end]['h'] + num + b

        x4, y4 = contours_dict[first]['x'] - num, contours_dict[first]['y']  + contours_dict[first]['h'] + num + a

        src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32) # Input 4개 지점

        dst = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float32) # Output 4개 지점

        matrix = cv2.getPerspectiveTransform(src, dst) # src -> dst 변환 행렬 얻어옴 

        warp = cv2.warpPerspective(white_board, matrix, (200, 100)) # 이미지 변환

        border = cv2.copyMakeBorder(warp, top=10, bottom=10, left=10, right=10, borderType= cv2.BORDER_CONSTANT, value=(255,255,255)) #

        new_gray_img = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)

        new_gaus_img = cv2.GaussianBlur(new_gray_img, (3, 3), 0) 

        ret, otsu1 = cv2.threshold(new_gaus_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        erode = cv2.erode(otsu1, kernel, iterations=1) # 이미지 인식을 위해 이미지 수정

        final_img = cv2.dilate(erode, kernel, iterations=1) # 이미지 인식을 위해 이미지 팽창

        text = pytesseract.image_to_string(final_img, config='--psm 7 --oem 3  -c tessedit_char_whitelist=0123456789').split('\n')[0:-1][0] 

        return text

    except Exception as e:
        
        print(e)
        
        return -1

# 버스 리스트와 숫자텍스트를 받아와서 포함하는지 확안 후 사실일 경우 tts 함수 호출


#TTS 음성출력 프로그램 텍스트를 받아서 음성실행
def string2voice(desire_num,finalbusnum): #원하는 버스의 번호와 거리 정보를 입력받아 음성으로 출력해주는 함수
    
    image_sequence=finalbusnum.index(desire_num)

    string=str(desire_num)

    sText = string + "번 버스가 도착하였습니다" #도착 문구

    img_distance_wanted_str=str(distance_list_rearranged[image_sequence])

    sText = string + "번 버스가" + img_distance_wanted_str + "미터에 도착했습니다" #도착 문구

    tts=gTTS(text=sText, lang='ko', slow=False) #gTTS 함수 사용
        
    sSaveFile ="text_intro2.mp3"

    tts.save(sSaveFile)  #음성파일저장

    playsound.playsound("text_intro2.mp3") #음성 출력

    os.remove(r"text_intro2.mp3")

    desire_num_list[4]=time.time()

def find_selenium():
    tag="redlaptop"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver=webdriver.Chrome("/lib/chromium-browser/chromedriver",chrome_options=chrome_options)
    driver.get("http://tinywebdb.appinventor.mit.edu/getvalue")

    driver.find_element(by=By.XPATH,value='/html/body/form/p/input').send_keys(tag)

    driver.find_element(by=By.XPATH,value='/html/body/form/input[2]').click()
    pList=driver.find_elements(by=By.TAG_NAME,value='p')
       

    number = re.findall("\d+", pList[1].text)
    driver.close()
    return number

def clear_selenium():
    tag="redlaptop"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver=webdriver.Chrome("/lib/chromium-browser/chromedriver",chrome_options=chrome_options)
    
    driver.get("http://tinywebdb.appinventor.mit.edu/storeavalue")
            



    driver.find_element(by=By.XPATH,value='/html/body/form/p[1]/input').send_keys(tag)
    driver.find_element(by=By.XPATH,value='/html/body/form/p[2]/input').send_keys("0")

    driver.find_element(by=By.XPATH,value='/html/body/form/input[2]').click()
    pList=driver.find_elements(by=By.TAG_NAME,value='p')
       

    numbers = re.findall("\d+", pList[1].text)
    driver.close()


def bus_list_LCD_ON(desire_num): #탑승을 원하는 버스 번호을 받고 전광판에 출력해주는 함수
    station_lcd = RPi_I2C_driver.lcd()

    sText2=''#전광판에 출력할 버스 번호를 저장할 문자열 변수
    
    if (desire_num != '0'):
        
        bus_num_str=str(desire_num)
        sText2=bus_num_str
        #전광판 첫 줄에 출력할 버스 번호를 텍스트로 저장할 변수
        station_lcd.backlight(1) #LCD 전광판을 킴
        station_lcd.lcd_clear() #LCD 전광판의 문구 삭제
        station_lcd.lcd_display_string(sText2,1)
        #LCD 전광판에 기사님께 양해를 드리는 문구 출력
        desire_num_list[3] = desire_num
        station_lcd.lcd_display_string("please wait",2)

def bus_list_LCD_OFF(): #탑승을 원하는 버스 번호을 받고 전광판에 출력해주는 함수
    station_lcd = RPi_I2C_driver.lcd()
    station_lcd.lcd_clear() #LCD 전광판의 문구 삭제


def cam_capture(): #캠의 아이디를 인자로 받아 캠의 이미지를 파일로 저장하는 함수
    
    cap = cv2.VideoCapture(0) #영상을 입력받을 카메라 디바이스 객체 생성
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3804) # 가로
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    ret,frame=cap.read() #카메라가 찍은 사진의 크기 정보를 변수에 저장
    cv2.imwrite("qwer.jpg",frame)
    cap.release()

#cpu1
def cpu1(desire_num_list):#공유 리스트 desire_num_list는 [셀레니움에서 받은 값, 처리 가능 여부 숫자1, 처리 가능 여부 숫자2] 형태
    
    start_time2 = time.time() #걸리는 시간 측정 임시 코드
    desire_num_list[1]=0 #공유 리스트에 저장된 실행 가능 숫자를 0으로 저장, 프로세스 진행중
    #bus_station_num =14170 # 서강대학교 정문 -> 신촌로터리 방향
    

    #img = get_img() # 파이카메라로 사진 받아오는함수 추후 추가예정
    #img = cv2.imread('D:\\gong\\image11.jpg')
    #img = cv2.imread('KakaoTalk_20220817_144936796_09.jpg')
    #img = cv2.imread('WIN_20220826_17_27_44_Pro.jpg')
    
    img = cv2.imread('qwer.jpg')
    
    #img = cv2.imread(cam_capture(0)) #지정된 카메라가 저장한 이미지 파일을 변수로 저장
    
    xxxx = img.copy()

    img_list = img_segmentation(img, model, xxxx)
    

    distance_rearrange(img_list) #이미지 리스트를 번호판 이미지의 크기별로 정렬
    img_listed = img_list_rearranged #정렬된 이미지 리스트로 대체

    busnum_text = []
    
    for img_file in img_listed:
        
        text1=img2string(img_file)
        busnum_text.append(text1)
        print(text1)
         
    if desire_num_list[0] in busnum_text:
        string2voice(desire_num_list[0],busnum_text)#버스 번호와 최종 버스 목록을 인자로 받음
        clear_selenium()
        
    desire_num_list[1]=1 #공유 리스트에 저장된 실행 가능 숫자를 1으로 저장, 프로세스 다시 실행 가능
    end_time2 = time.time() #걸리는 시간 측정 임시 코드
    print('none')
    
    
# #cpu2
def cpu2(desire_num_list):
    
    start_555 = time.time()
    desire_num_list[2]=0 #공유 리스트에 저장된 실행 가능 숫자를 0으로 저장, 프로세스 진행중
    desire_num_list[0] = find_selenium()[0] #버스 번호 변수에 셀레니움에서 받아온 값을 공유리스트에 저장
    if desire_num_list[3] != desire_num_list[0]:
        bus_list_LCD_ON(desire_num_list[0])
        
    time_gap=15 #음성이 켜지고 LCD가 꺼질 때까지의 시간 간격 설정
    if ((desire_num_list[4] != 0) and (time.time()-desire_num_list[4]>=time_gap)): #음성이 켜졌을 때 시간과 비교하여 일정 시간이 지나면 LCD를 끔
        bus_list_LCD_OFF() #LCD를 끄는 함수
        desire_num_list[3] = 0
        desire_num_list[4] = 0 # LCD를 끄면 다시 0으로 초기화
    desire_num_list[2]=1 #공유 리스트에 저장된 실행 가능 숫자를 1으로 저장, 프로세스 다시 실행 가능
    
    
    #버스 번호와 그 버스가 카메라로부터 떨어진 대략적인 거리를 음성으로 출력
    #공유 리스트 desire_num_list는 [셀레니움에서 받은 값] 형태
    
    
#멀티프로세스
#사용자가 탑승을 원하는 버스 번호를 휴대폰으로 받고 라즈베리 파이로 전송하는 작업과
#카메라로 이미지를 처리하는 작업을 동시에 진행
if __name__ == '__main__':
    manager = multiprocessing.Manager()
    desire_num_list = manager.list(range(5)) #공유 리스트 desire_num_list를 선언
    desire_num_list[0]=0 #첫번째에는 사용자가 탑승을 원하는 버스 번호 저장, 초기값 0
    desire_num_list[1]=1 #두번째, 세번째에는 각 프로세스 종료 여부 저장
    desire_num_list[2]=1 #시작 가능 : 1, 진행중 : 0 초기값: 시작 가능(1)
    desire_num_list[3]=0  
    desire_num_list[4]=0 # LCD 시간
    model = get_Yolov5_weight()
    bus_list_LCD_OFF()
    if os.path.isfile("qwer.jpg"):
        os.remove("qwer.jpg")

    while True: #무한으로 이 과정을 일정 시간마다 반복
        #cpu1, cpu2 함수를 각 프로세스에서
        #공유 리스트 desire_num_list 를 인자로 받아 실행
        
        proc1 = multiprocessing.Process(target=cam_capture(),)
        proc2 = multiprocessing.Process(target=cpu1,args=(desire_num_list,))
        proc3 = multiprocessing.Process(target=cpu2,args=(desire_num_list,))
        
        start_444 = time.time()
        proc1.start()
        
        if(desire_num_list[1]==1): #공유 리스트에 시작 가능 변수가 1일때 프로세스 실행
            proc2.start() #각 프로세스를 실행
        if(desire_num_list[2]==1):
            proc3.start()
        
        proc2.join()
        print("proc 1 end: " + str(time.time()- start_444))
        
