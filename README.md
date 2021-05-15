# DSAI_HW3_Green Energy Trading
NCKU DSAI course homework

## 前置工作
### 作業說明
* 說明連結與交易表單
  - [Slide](https://docs.google.com/presentation/d/1JW27_5HXYZhqWmgvDhtXBaFTOfksO_dS/edit#slide=id.p1)
  - [Dashboard](https://docs.google.com/spreadsheets/d/1cjhQewnXT2IbmYkGXRYNC5PlGRafcbVprCjgSFyDAaU/edit?pli=1#gid=0)

* 目標\
以過去七日生產或消耗電量為基準，\
決定明日需要跟市場買電或賣電，用以最小化用電支出。

### 環境
* python 3.8
* Win 10

### 使用方式
1. 進入專案資料夾\
`cd /d [path/to/this/project]` 

2. 使用`pipenv`自動安裝所需套件與環境

3. 執行程式訓練模型\
`python main_T.py --train True`
訓練完成後會自動儲存模型為`model`至根目錄。

4. 執行程式生成交易資訊\
`python main.py`

## 資料來源
訓練資料為電力公司提供之50戶家庭的每日24小時生產及用電量。\

## 分析
取訓練資料平均一天的電量資訊進行分析。\
![消耗](https://i.imgur.com/gVxjDme.png)
![生產](https://i.imgur.com/q9DN3WG.png)

可以發現儘管每筆資料的差異甚大，\
但仍遵循一定的規律，考量到生產的電力可以自用的規則，\
我們決定預測未來24小時的**生產與消耗電量差**，\
並用差值決定買賣量，詳見後續`規則`條目。

## 方法說明
### 概述
使用sliding windows的方式切割資料，以過去7天24小時的生產與消耗，\
預測隔天24小時的生產與消耗電量差值當作買賣量。\
\
價格部分以預設買2.2元/賣2.0元出發，\
當**生產大於消耗**時即決定賣電，\
並以前日交易結果之**賣價與賣電**量計算**加權平均**決定定價。\
同理，當**消耗大於生產**時則電，\
以前日的**賣價與賣電**量計算**加權平均**決定買價。\
\
另外由於平台上的交易價起伏不大，\
故最終會再賣價部分扣除0.25元，避免漲價過高；\
且當**賣價小於2.0元**或**買價大於2.4元**時，則維持該價格。

### 讀取資料
```py
dpath = './training_data'
data_list = os.listdir(dpath)
Gdata, Cdata, Glabel, Clabel = [], [], [], []
# 使用迴圈讀取50戶電力資訊
for i in range(len(data_list)):
   data_ag = np.array(pd.read_csv(os.path.join(dpath, data_list[i]), header=None))[1:,1:]
   ag = np.stack(data_ag).astype(None)
   gen, con = ag[:,0], ag[:,1]
   # sliding windows切割天數，window=7
   gen_data, con_data = func._pack(gen)[:-24, :], func._pack(con)[:-24, :]
   # 取隔天 (第8天)之24小時用電資訊
   gen_label, con_label = func._pack(gen[7*24:], win=24), func._pack(con[7*24:], win=24)

   Gdata.append(gen_data)
   Cdata.append(con_data)
   Glabel.append(gen_label)
   Clabel.append(con_label)

Gndata = np.vstack(Gdata)
Cndata = np.vstack(Cdata)
# 訓練資料為生產與消耗之串接，大小為 (n, 7*24*2)
# 標籤為生產與消耗之差值，大小為 (n, 24)
ndata = np.concatenate((Gndata, Cndata), axis=-1)
label = (np.vstack(Glabel) - np.vstack(Clabel))
```

### 訓練模型
```py
model = linear_model.Lasso(alpha=1e-2)
model.fit(ndata, label)
# 使用joblib儲存模型
joblib.dump(model, 'model')
```
使用的模型為Lasso，由於在計算的過程中會同時進行進行特徵選擇和正則化，\
故此次對資料並無進行正規化。\
\
模型分別預測生產、消耗與差值的軌跡如下：
![預測](https://i.imgur.com/fbvlAAt.png)
可以看出雖與實際值存在差異，但仍體現了其趨勢。

### 預測數值
```py
# 讀取模型
model = joblib.load('model')
# 讀取資料
G_path = args.generation
C_path = args.consumption 
GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
GVal = np.stack(GVal).astype(None)[:,0]
CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
CVal = np.stack(CVal).astype(None)[:,0] 
# 堆疊資料後進行預測，大小為 (1, 7*24)
date_pre = np.array(pd.read_csv(C_path, header=None))[-1,0]
GVdata, CVdata = GVal[np.newaxis, :], CVal[np.newaxis, :]
nVdata = np.concatenate((GVdata, CVdata), axis=1)

pred = model.predict(nVdata) 
```

### 讀取交易紀錄
```py
path = args.bidresult
out_buy, out_sell = func._bid(path, 'cost.npy')
act = func._comp2(pred)
D = func._output3(pred, act, date_pre, out_buy, out_sell)
# 將交易價格儲存，供下次預測時修正用
with open('cost.npy', 'wb') as f:
    np.save(f, out_buy)
    np.save(f, out_sell)

df = pd.DataFrame(D, columns=["time", "action", "target_price", "target_volume"])
df.to_csv(args.output, index=False)
```




