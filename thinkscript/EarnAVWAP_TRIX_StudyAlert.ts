# EarnAVWAP TRIX — server-side Study Alert condition
# Runs on Schwab servers, so it pushes to the thinkorswim mobile app
# or email even when the desktop platform is closed.
#
# Setup: MarketWatch tab -> Alerts -> enter symbol -> Study Alert ->
# thinkScript Editor tab -> paste this -> aggregation: Day.
# Check "Recreate alert" so it re-arms after firing.
# Notifications: Setup -> Application Settings -> Notifications.
#
# Uses a hardcoded anchor date (earnings-event functions are not
# reliable server-side) — update anchorDate each quarter to the
# earnings reaction date.

def anchorDate = 20260528; # last earnings reaction date, yyyymmdd
def inA = GetYYYYMMDD() >= anchorDate;
def s = inA and !inA[1];
def tp = (high + low + close) / 3;
def pv = if s then tp * volume else if inA then pv[1] + tp * volume else 0;
def vv = if s then volume else if inA then vv[1] + volume else 0;
def avwap = if vv > 0 then pv / vv else Double.NaN;
def e3 = ExpAverage(ExpAverage(ExpAverage(Log(close), 9), 9), 9);
def trix = (e3 - e3[1]) * 10000;
def aboveV = close > avwap;
def tPos = trix > 0;
plot signal = (aboveV and tPos and !tPos[1]) or (tPos and aboveV and !aboveV[1]);
