# EarnAVWAP TRIX Alert — chart study for thinkorswim desktop
# Alerts when a candle closes above the Earnings AVWAP with TRIX
# above zero — fires when EITHER condition completes the pair.
# Also paints a green arrow under each historical signal bar.
# Alert() only fires while the desktop platform is open with the
# chart loaded; for phone push alerts use the server-side condition
# in EarnAVWAP_TRIX_StudyAlert.ts instead.
declare hide_on_intraday;

input trixLength = 9;
input startNextDay = yes;

def tp = (high + low + close) / 3;
def bn = BarNumber();
def lastBar = HighestAll(if !IsNaN(close) then bn else 0);

# --- Earnings AVWAP (same auto logic as EarningsAVWAP_Auto) ---
def rawE = HasEarnings();
def e0 = if IsNaN(rawE) then 0 else if rawE then 1 else 0;
def e = if startNextDay then e0[1] else e0;
def hasEventData = HighestAll(e0) > 0;
def cnt = CompoundValue(1, cnt[1] + e, 0);
def total = HighestAll(cnt);
def inQ1 = bn > lastBar - 63 and bn <= lastBar;
def maxV1 = HighestAll(if inQ1 then volume else 0);
def a1bn = HighestAll(if inQ1 and volume == maxV1 then bn else 0);
def A1 = if hasEventData then e and cnt == total else bn == a1bn;
def pv1 = if A1 then tp * volume else pv1[1] + tp * volume;
def v1  = if A1 then volume else v1[1] + volume;
def on1 = CompoundValue(1, on1[1] or A1, no);
def avwap = if on1 and v1 > 0 then pv1 / v1 else Double.NaN;

# --- TRIX ---
def e3 = ExpAverage(ExpAverage(ExpAverage(Log(close), trixLength), trixLength), trixLength);
def trix = (e3 - e3[1]) * 10000;

# --- signal: both true, triggered by whichever turns true last ---
def aboveV = close > avwap;
def tPos = trix > 0;
def signal = (aboveV and tPos and !tPos[1]) or (tPos and aboveV and !aboveV[1]);

plot Arrow = if signal then low * 0.99 else Double.NaN;
Arrow.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
Arrow.SetDefaultColor(Color.GREEN);
Arrow.SetLineWeight(3);

Alert(signal, "Close > EarnAVWAP + TRIX above zero", Alert.BAR, Sound.Ding);
