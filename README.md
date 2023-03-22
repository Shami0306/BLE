# BLE for RSSI Matching

before&after目錄 : 存放未經處理&處理過後的藍芽RSSI資料  

ouput目錄 : model訓練完的參數檔及log檔  

init_paths.py : 用來設置系統路徑的檔案  


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

### 測試

```
python test.py --cfg  .\config\U19e_outdoor0103test1.yaml
```
**每次測試不同項目(Loss or video)前皆需重新執行filter_csv.py，確保label欄位正常。


