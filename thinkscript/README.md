# Earnings AVWAP — thinkorswim studies

Custom thinkScript studies for charting Anchored VWAP from earnings dates,
plus trend labels. Developed for use on thinkorswim desktop and mobile.

## Files

| File | What it does | Where it works |
|---|---|---|
| `EarningsAVWAP_Auto.ts` | Main study. Plots AVWAP from the last TWO earnings dates, self-updating every quarter — no maintenance. Cyan solid = most recent earnings, orange dashed = prior quarter. | Desktop (exact earnings events) and mobile (volume-peak detection). Daily charts, 1yr+ history. |
| `EarningsAVWAP_ManualDates.ts` | Backup study with typed-in anchor dates (yyyymmdd inputs). | Everywhere, including intraday charts. Update dates each quarter. |
| `TrendLabels.ts` | Corner labels: short-term trend (8/21 EMA) and long-term trend (200 SMA). | Daily charts. |

## How to install

1. thinkorswim **desktop** → Charts tab → Studies (flask icon) → **Create Study…**
2. Delete the placeholder code, paste the file contents, name the study, click OK.
3. Add it to the chart like any built-in study.
4. **Mobile:** log out and back in to the thinkorswim mobile app, then
   chart → studies → Add Study → find it under your custom/My Studies section.

## Notes

- `EarningsAVWAP_Auto` anchors to the **first session after** the earnings
  flag (the day that trades on the news). For companies that report before
  the open, set the `startNextDay` input to `no`.
- Mobile has no earnings-event data, so the auto study falls back to
  anchoring on the quarter's biggest volume day — which is the earnings
  reaction day for virtually all normally traded stocks. The corner label
  shows which mode is active.
- Companion built-in studies for the full chart setup: MovAvgExponential
  (8, 21), SimpleMovingAvg (50, 200), RSI or MACD, VolumeAvg (20), ATR (14),
  RelativeStrength vs SPX.
