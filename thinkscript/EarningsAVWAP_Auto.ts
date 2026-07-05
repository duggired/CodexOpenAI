# Earnings AVWAP — last TWO earnings, anchored to the NEXT session
# after the earnings event (the first day that trades on the news)
#
# Works on thinkorswim desktop AND mobile:
#   Desktop: anchors on real earnings events (HasEarnings)
#   Mobile (no event data): anchors on the two biggest quarterly
#     volume days, which are the earnings reaction days
#
# Usage: daily chart, 1 year+ of history. The corner label shows
# which anchor mode the device is using.
declare hide_on_intraday;

input startNextDay = yes;  # yes = anchor the day after the earnings flag
input showLabel = yes;     # set to no on mobile to hide the anchor-mode label
                           # set to no for stocks that report before the open

def tp = (high + low + close) / 3;
def bn = BarNumber();
def lastBar = HighestAll(if !IsNaN(close) then bn else 0);

# ---------- desktop mode: real earnings events ----------
def rawE = HasEarnings();
def e0 = if IsNaN(rawE) then 0 else if rawE then 1 else 0;
def e  = if startNextDay then e0[1] else e0;   # shift anchor to next bar
def hasEventData = HighestAll(e0) > 0;
def cnt = CompoundValue(1, cnt[1] + e, 0);
def total = HighestAll(cnt);

# ---------- mobile mode: volume-based detection ----------
# (volume peak already occurs on the post-earnings reaction day,
#  so no shift is applied here)
# window sizes adapt to the chart aggregation: one quarter is
# ~63 daily bars or ~13 weekly bars
def wkly = GetAggregationPeriod() >= AggregationPeriod.WEEK;
def qBars = if wkly then 13 else 63;
def q2Win = if wkly then 15 else 68;
def q2Gap = if wkly then 1 else 5;

def inQ1 = bn > lastBar - qBars and bn <= lastBar;
def maxV1 = HighestAll(if inQ1 then volume else 0);
def a1bn = HighestAll(if inQ1 and volume == maxV1 then bn else 0);

def inQ2 = bn > a1bn - q2Win and bn < a1bn - q2Gap;
def maxV2 = HighestAll(if inQ2 then volume else 0);
def a2bn = HighestAll(if inQ2 and volume == maxV2 then bn else 0);

# ---------- unified anchors ----------
def A1 = if hasEventData then e and cnt == total     else bn == a1bn;
def A2 = if hasEventData then e and cnt == total - 1 else bn == a2bn;

# ---- AVWAP from most recent earnings (cyan) ----
def pv1 = if A1 then tp * volume else pv1[1] + tp * volume;
def v1  = if A1 then volume else v1[1] + volume;
def on1 = CompoundValue(1, on1[1] or A1, no);
plot CurrentEarnVWAP = if on1 and v1 > 0 then pv1 / v1 else Double.NaN;
CurrentEarnVWAP.SetDefaultColor(Color.CYAN);
CurrentEarnVWAP.SetLineWeight(2);

# ---- AVWAP from prior earnings (orange dashed) ----
def pv2 = if A2 then tp * volume else pv2[1] + tp * volume;
def v2  = if A2 then volume else v2[1] + volume;
def on2 = CompoundValue(1, on2[1] or A2, no);
plot PriorEarnVWAP = if on2 and v2 > 0 then pv2 / v2 else Double.NaN;
PriorEarnVWAP.SetDefaultColor(Color.ORANGE);
PriorEarnVWAP.SetStyle(Curve.MEDIUM_DASH);

AddLabel(showLabel, if hasEventData then "Anchors: earnings events (next-day start)"
              else "Anchors: quarterly volume peaks",
         if hasEventData then Color.GREEN else Color.YELLOW);
