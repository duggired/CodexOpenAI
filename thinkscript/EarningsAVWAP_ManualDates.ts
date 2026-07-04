# Earnings AVWAP — manual anchor dates (works on mobile + intraday)
# Backup version: anchors to dates you type in. Use when you want an
# exact anchor on a specific date (e.g. intraday charts, or a stock
# where a non-earnings event out-traded earnings that quarter).
# Update anchor1/anchor2 in the study settings after each report.

input anchor1 = 20260528; # most recent earnings date (yyyymmdd)
input anchor2 = 20260225; # prior quarter earnings date (yyyymmdd)
# tip: if the report came out after the close, use the NEXT trading day

def tp = (high + low + close) / 3;
def d = GetYYYYMMDD();

def in1 = d >= anchor1;
def s1 = in1 and !in1[1];
def pv1 = if s1 then tp * volume else if in1 then pv1[1] + tp * volume else 0;
def v1 = if s1 then volume else if in1 then v1[1] + volume else 0;
plot CurrentEarnVWAP = if in1 and v1 > 0 then pv1 / v1 else Double.NaN;
CurrentEarnVWAP.SetDefaultColor(Color.CYAN);
CurrentEarnVWAP.SetLineWeight(2);

def in2 = d >= anchor2;
def s2 = in2 and !in2[1];
def pv2 = if s2 then tp * volume else if in2 then pv2[1] + tp * volume else 0;
def v2 = if s2 then volume else if in2 then v2[1] + volume else 0;
plot PriorEarnVWAP = if in2 and v2 > 0 then pv2 / v2 else Double.NaN;
PriorEarnVWAP.SetDefaultColor(Color.ORANGE);
PriorEarnVWAP.SetStyle(Curve.MEDIUM_DASH);
