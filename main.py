import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


class MarketMakerStrategy:
    def __init__(self, file_path):
        #讀取資料
        print("載入資料中")
        self.df = pd.read_csv(
            file_path, 
            header=0, 
            names=['Symbol', 'Date', 'Time', 'Price', 'Volume']
        )
        self.inventory_limit = 10 #設定最大庫存限制 避免單向曝險過高
        print(self.df.head())  

    def prepare_data(self):
        print("資料前處理中")
        #calculate mid price
        self.df['mid_price'] = self.df['Price'] 
        #calculate volatility
        self.df['volatility'] = self.df['Price'].rolling(window=60).std()
        #calculate liquidity
        self.df['liquidity'] = self.df['Volume'].rolling(window=60).mean()
        #calculate order imbalance
        self.df['price_diff'] = self.df['Price'].diff() #記算價差
        self.df['trade_direction'] = np.sign(self.df['price_diff']).replace(0, np.nan).ffill().fillna(0) #根據價差計算交易方向，0的部分用前一個forward fill填充，並將剩餘的NA值填充為0
        self.df['signed_volume'] = self.df['trade_direction'] * self.df['Volume'] #計算帶符號的成交量，買單為正，賣單為負
        self.df['order_imbalance'] = self.df['signed_volume'].rolling(window=20).sum().fillna(0) #計算訂單不平衡，使用帶符號的成交量在20筆資料的滾動窗口內求和，並將NA值填充為0

        ## AI 給出來的半價差代理變數計算方法
        # --- Half-Spread Proxy (動態推估半價差)  ---
        # 1. 計算每一次成交價的絕對跳動幅度
        price_jumps = self.df['Price'].diff().abs()
        # 2. 忽略平盤 (0)，並將前一次的有效跳動幅度遞延 (ffill)
        price_jumps = price_jumps.replace(0, np.nan).ffill()
        # 3. 取過去 60 筆有效跳動的平均值，除以 2 作為「動態半價差」
        self.df['half_spread_proxy'] = price_jumps.rolling(window=60).mean() / 2
        # 4. 處理最前面的空值，給予預設的最小半價差 (假設為 0.5)
        self.df['half_spread_proxy'] = self.df['half_spread_proxy'].fillna(0.5)
        

    def calculate_signals(self):
        print("計算買賣價格 (Signals)")
        #variable
        w_risk = 0.5    #波動度權重
        w_liq = 5       #流動性權重
        base_spread = 2 #台指期基本滑價/跳動點位假設 (對應你的 8_base) 
        '''
        如果要用動態半價差的話就把base_spread改成delta_base 
        '''

        delta_base = (self.df['half_spread_proxy']*5.0).clip(lower=2.5) #使用動態推估的半價差 
        delta_risk = self.df['volatility'] * w_risk
        delta_liq = w_liq / (self.df['liquidity'] + 1) #避免當完全無交易量時除以零
        self.df['delta'] = delta_base + delta_risk + delta_liq

        self.df['my_bid'] = (self.df['mid_price'] - self.df['delta']).shift(1)
        self.df['my_ask'] = (self.df['mid_price'] + self.df['delta']).shift(1) #要把算出來的價格掛到下一筆資料，才不會有未來函數問題

    def run_backtest(self):
        print("執行撮合")
        inventory = 0
        realized_pnl = 0
        pnl_record = []

        prices = self.df['Price'].values
        my_bids = self.df['my_bid'].values
        my_asks = self.df['my_ask'].values

        transaction_cost = 1.0 #假設單筆手續費與滑價總和為1元

        for i in range(len(self.df)):
            # 略過最前面因 shift 產生的空值
            if np.isnan(my_bids[i]):
                pnl_record.append(0)
                continue
                
            curr_price = prices[i]
            
            #--- 撮合邏輯 ---
            #買單成交：市場價低於我們的買價，且多單庫存未達上限
            if curr_price <= my_bids[i] and inventory < self.inventory_limit:
                inventory += 1
                realized_pnl -= my_bids[i] #支付現金買入
                realized_pnl -= transaction_cost #扣除交易成本
                
            #賣單成交：市場價超過我們的賣價，且空單庫存未達上限
            elif curr_price >= my_asks[i] and inventory > -self.inventory_limit:
                inventory -= 1
                realized_pnl += my_asks[i] #取得現金賣出
                realized_pnl -= transaction_cost #扣除交易成本
                
            #這裡的 mtm_pnl 是未實現損益 + 已實現損益，反映當前庫存的市值變化
            mtm_pnl = realized_pnl + (inventory * curr_price)
            pnl_record.append(mtm_pnl)

        
        self.df['cumulative_pnl'] = pnl_record #要記錄的並不是每筆資料的損益變化，而是每筆資料的累積損益，這樣才能反映整體策略的績效走勢

    def evaluate_performance(self):
        print("評估策略績效")
        self.df['pnl_change'] = self.df['cumulative_pnl'].diff().fillna(0) #計算每筆資料的損益變化，第一筆資料的變化設為0
        total_pnl = self.df['cumulative_pnl'].iloc[-1] #取最後一筆資料的累積損益作為總損益

        ###算最大回撤(MDD)
        rolling_max = self.df['cumulative_pnl'].cummax() #cummax() 函數會返回到目前為止的最大值，這樣我們就可以知道每個時間點的最高損益水平
        drawdown = self.df['cumulative_pnl'] - rolling_max #每次從高點摔下來的深度
        mdd = drawdown.min()

        ###計算 Sharpe Ratio 與 Sortino Ratio
        #這裡的運算是以每 1000 筆資料為一個群組來計算報酬率的平均值和標準差，然後再年化處理。這樣做的好處是可以減少單筆資料的噪音對績效指標的影響，讓評估更穩健。
        period_returns = self.df['pnl_change'].groupby(self.df.index // 1000).sum()
        mean_return = period_returns.mean() #計算每個群組的平均報酬率
        std_return = period_returns.std() #計算每個群組的報酬率標準差

        annualization_factor = np.sqrt(25200) 

        '''
        因為時間拉長的時候，『報酬』是會線性累積的，但『波動風險』常常會上下互相抵銷。在統計上，風險的增長速度只有時間的開根號。
        因為 Sharpe Ratio 是報酬除以風險，分子乘上 N 倍，分母乘上根號 N 倍，約分之後，我們只需要把算出來的 Sharpe Ratio 乘上時間的開根號  
        N 就可以了。這裡的 N 大約是一年會出現的資料區塊數量（約 25,200)。
        '''
        sharpe_ratio = (mean_return / std_return) * annualization_factor if std_return != 0 else 0
        
        downside_std = period_returns[period_returns < 0].std()
        sortino_ratio = (mean_return / downside_std) * annualization_factor if downside_std != 0 else 0

        print("回測報告")
        print("-" * 30)
        print(f"總損益(PnL): {total_pnl:.2f}")
        print(f"最大回撤 (MDD): {mdd:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")  


#==========================================
#執行區塊
#==========================================      
if __name__ == "__main__":
    file_name = 'TXF1-Tick-Trade.txt'
    strategy = MarketMakerStrategy(file_name)
    strategy.prepare_data()
    strategy.calculate_signals()
    strategy.run_backtest()
    strategy.evaluate_performance()