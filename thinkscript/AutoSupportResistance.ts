# Auto Support/Resistance — pivot-based, self-updating
# Draws the 3 most recent swing-high resistance and 3 most recent
# swing-low support levels, plus the 52-week high/low.
# Use on a DAILY chart with 1 year+ of history.
# Works on thinkorswim desktop and mobile (basic functions only).
#
# pivotStrength: bars on each side that must not exceed a swing for
# it to count as a pivot. 10 (~2 weeks) is a good daily default;
# 15-20 = fewer/stronger levels, 5 = more/minor levels.
# Note: a swing confirms pivotStrength bars after it forms — the
# newest level appears with that delay by construction.
declare hide_on_intraday;

input pivotStrength = 10;  # bars each side that must not exceed the pivot
input showBubbles = yes;   # price bubbles at the right edge

def bn = BarNumber();
def lastBar = HighestAll(if !IsNaN(close) then bn else 0);
def w = pivotStrength;

# --- pivot detection (confirmed w bars after the swing) ---
def pivotH = high >= Highest(high, w + 1) and high >= Highest(high[-w], w);
def pivotL = low  <= Lowest(low, w + 1)  and low  <= Lowest(low[-w], w);

# --- 3 most recent resistance levels ---
def rB1 = HighestAll(if pivotH then bn else 0);
def rB2 = HighestAll(if pivotH and bn < rB1 then bn else 0);
def rB3 = HighestAll(if pivotH and bn < rB2 then bn else 0);
def R1v = HighestAll(if bn == rB1 then high else 0);
def R2v = HighestAll(if bn == rB2 then high else 0);
def R3v = HighestAll(if bn == rB3 then high else 0);

plot R1 = if R1v > 0 then R1v else Double.NaN;
plot R2 = if R2v > 0 then R2v else Double.NaN;
plot R3 = if R3v > 0 then R3v else Double.NaN;
R1.SetDefaultColor(Color.RED);        R1.SetLineWeight(2);
R2.SetDefaultColor(Color.DARK_RED);   R2.SetStyle(Curve.SHORT_DASH);
R3.SetDefaultColor(Color.DARK_RED);   R3.SetStyle(Curve.SHORT_DASH);

# --- 3 most recent support levels ---
def sB1 = HighestAll(if pivotL then bn else 0);
def sB2 = HighestAll(if pivotL and bn < sB1 then bn else 0);
def sB3 = HighestAll(if pivotL and bn < sB2 then bn else 0);
def S1v = HighestAll(if bn == sB1 then low else 0);
def S2v = HighestAll(if bn == sB2 then low else 0);
def S3v = HighestAll(if bn == sB3 then low else 0);

plot S1 = if S1v > 0 then S1v else Double.NaN;
plot S2 = if S2v > 0 then S2v else Double.NaN;
plot S3 = if S3v > 0 then S3v else Double.NaN;
S1.SetDefaultColor(Color.GREEN);       S1.SetLineWeight(2);
S2.SetDefaultColor(Color.DARK_GREEN);  S2.SetStyle(Curve.SHORT_DASH);
S3.SetDefaultColor(Color.DARK_GREEN);  S3.SetStyle(Curve.SHORT_DASH);

# --- 52-week high / low (aggregation-aware: 252 daily / 52 weekly bars) ---
def wkly = GetAggregationPeriod() >= AggregationPeriod.WEEK;
def yh252 = if wkly then Highest(high, 52) else Highest(high, 252);
def yl252 = if wkly then Lowest(low, 52) else Lowest(low, 252);
def yhV = HighestAll(if bn == lastBar then yh252 else 0);
def ylV = HighestAll(if bn == lastBar then yl252 else 0);
plot YrHigh = if yhV > 0 then yhV else Double.NaN;
plot YrLow  = if ylV > 0 then ylV else Double.NaN;
YrHigh.SetDefaultColor(Color.GRAY);  YrHigh.SetStyle(Curve.LONG_DASH);
YrLow.SetDefaultColor(Color.GRAY);   YrLow.SetStyle(Curve.LONG_DASH);

# --- right-edge price bubbles ---
AddChartBubble(showBubbles and bn == lastBar, R1, "R " + Round(R1, 2), Color.RED, yes);
AddChartBubble(showBubbles and bn == lastBar, S1, "S " + Round(S1, 2), Color.GREEN, no);
