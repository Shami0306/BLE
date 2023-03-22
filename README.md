# BLE for RSSI Matching

before&after目錄 : 存放未經處理&處理過後的藍芽RSSI資料  

ouput目錄 : model訓練完的參數檔及log檔  

init_paths.py : 用來設置系統路徑的檔案  


### 資料前處理  
#### 產生訓練用的資料，在目標資料下生成8個Blocks的csv檔。
```
python filter_csv.py --cfg yaml路徑 --mode train
```



