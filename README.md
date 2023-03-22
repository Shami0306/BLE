# BLE for RSSI Matching

before&after目錄 : 存放未經處理&處理過後的藍芽RSSI資料  

ouput目錄 : model訓練完的參數檔及log檔  

init_paths.py : 用來設置系統路徑的檔案  


### 資料前處理  
#### 產生訓練用的資料  
```
python filter_csv.py --cfg yaml路徑 --mode train
```

在目標資料下生成8個Blocks的csv檔。  
#### 產生測試用的資料
```
python filter_csv.py --cfg yaml路徑 --mode test
```
**注意** 根據yaml中TEST_FOR_VIDEO設定  
  
如果TEST_FOR_VIDEO為**False**，則在目標資料下生成一個test.csv檔案，除了各個sniffer的rssi數值，**也保留了label**。  

如果TEST_FOR_VIDEO為**True**，則在目標資料下生成一個test.csv檔案，僅有各個sniffer的rssi數值，**沒有label**。  





