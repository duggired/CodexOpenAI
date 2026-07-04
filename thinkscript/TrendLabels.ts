# Trend labels — prints short-term and long-term trend verdicts
# in the corner of the chart. Use on a daily chart.
#
# Short-term: price vs 8/21 EMA stack
# Long-term:  price vs rising 200-day SMA

def ema8  = ExpAverage(close, 8);
def ema21 = ExpAverage(close, 21);
def sma200 = Average(close, 200);

def stUp = close > ema8 and ema8 > ema21;
def stDn = close < ema8 and ema8 < ema21;
AddLabel(yes, if stUp then "Short-term: UP" else if stDn then "Short-term: DOWN" else "Short-term: MIXED",
         if stUp then Color.GREEN else if stDn then Color.RED else Color.GRAY);

def ltUp = close > sma200 and sma200 > sma200[20];
AddLabel(yes, if ltUp then "Long-term: UP" else "Long-term: DOWN/FLAT",
         if ltUp then Color.GREEN else Color.RED);
