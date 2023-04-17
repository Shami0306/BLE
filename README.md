# BLE for RSSI Matching
## Introduction  
該專案用於使用低功耗藍芽(BLE)進行Fingerprinting定位法的實作，包含資料前處理、模型訓練以及測試結果。

### 檔案說明  
before&after目錄 : 存放未經處理&處理過後的藍芽RSSI資料  

ouput目錄 : model訓練完的參數檔及log檔  

init_paths.py : 用來設置系統路徑的檔案  

filter_csv.py : 用於資料前處理的程式。  

train.py : 用於訓練深度模型的程式。  

test.py : 用於測試模型的程式，可選擇查看數據或是影像。

### 資料前處理  

為每個訓練或測試用的資料各自建立yaml檔案。  

#### 產生訓練用的資料  
```
python filter_csv.py --cfg yaml路徑 --mode train
```

在目標資料下生成8個Blocks的csv檔。  
#### 產生測試用的資料
```
python filter_csv.py --cfg yaml路徑 --mode test
```
**注意** 根據yaml中TEST_FOR_VIDEO設定，會決定測試時是計算Loss還是以影片方式呈現。
  
如果TEST_FOR_VIDEO為**False**，則在目標資料下生成一個test.csv檔案，除了各個sniffer的rssi數值，**也保留了label**。  

如果TEST_FOR_VIDEO為**True**，則在目標資料下生成一個test.csv檔案，僅有各個sniffer的rssi數值，**沒有label**。  

### 訓練

```
python train.py --cfg  .\config\U19e_outdoor0103.yaml
```
訓練完的權重會儲存於yaml中OUTPUT_DIR的路徑下(對應OUTPUT_NAME的目錄中)。  

### 測試

```
python test.py --cfg  .\config\U19e_outdoor0103test1.yaml
```
**每次測試不同項目(Loss or video)前皆需重新執行filter_csv.py，確保label欄位正常。**
### 測試(For Accuracy)
```
# 設置為false
TEST_FOR_VIDEO : false
```

#### Accuracy and Error  
![image](https://user-images.githubusercontent.com/57833742/232435412-7e722af3-88a8-457b-a6f3-a81beb195944.png)

Top-1 accuracy表示是否ground truth與prediction完全match的準確率。  
In range accuracy表示是否ground truth落在prediction**前後1格** 範圍內的準確率。  
Mean Distance Error : 平均距離差。(每個Reference Point距離2m，Size為2x2) 



#### 測試生成的混淆矩陣
![image](https://user-images.githubusercontent.com/57833742/226887734-5b4f97d7-f44a-4753-95bd-fdb0ad2ee281.png)


### 測試(For Video)
yaml設定中的時間僅需填寫**開始與結束時間**，中間可以省略不看。
```
START_TIME : ['15:48:10', 'xxx', 'xxx', 'xxx', 'xxx', 'xxx', 'xxx', 'xxx']
END_TIME : ['xxx', 'xxx', 'xxx', 'xxx', 'xxx', 'xxx', 'xxx', '15:48:28']
# 設置為true
TEST_FOR_VIDEO : true
```

### RSSI Matching
![image](https://user-images.githubusercontent.com/57833742/226885060-567b7cb6-0f53-4989-bfc5-cff3c2ab20fa.png)

畫面中紅色框框是 **無線訊號(BLE)** 定位後，轉換到影像上的相對位置。  
黑色框框範圍是導盲磚的監控感測範圍，人物進入範圍內才會被偵測並判斷畫面中哪個人是用戶。


