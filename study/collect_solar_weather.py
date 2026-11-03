"""남동발전 태양광 발전량 + 기상청 ASOS 날씨를 모아서 표로 합치는 수집 스크립트.

필요: study/.env 에 G2B_KEY=<공공데이터포털 인증키>
     (활용신청: 한국남동발전 시간대별 태양광 발전실적, 기상청 ASOS 시간자료/일자료)

실행:
    source ~/venvs/tabpfn/bin/activate
    python study/collect_solar_weather.py            # 2023-01 ~ 2024-12
    python study/collect_solar_weather.py 2022 2024  # 연도 범위 지정

이미 받은 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 받는다.

결과 (study/data/solar/):
    raw/solar_YYYYMM.csv          태양광 원본 (발전기 x 날짜, 01~24시 컬럼)
    raw/wx_hourly_<지점>_<연도>.csv  기상청 시간자료
    raw/wx_daily_<지점>_<연도>.csv   기상청 일자료
    solar_daily.csv               하루 단위로 합친 최종 표  <- 실험에 쓸 파일
    solar_hourly.csv              시간 단위로 합친 최종 표
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "solar"
RAW = OUT / "raw"
KEY = re.search(r"^G2B_KEY=(.*)$", (ROOT / ".env").read_text(), re.M).group(1).strip()

SOLAR_URL = "https://apis.data.go.kr/B551893/solar-power-by-hour/list"
WX_HOURLY_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
WX_DAILY_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

# 발전소명에 들어간 지명 -> 가장 가까운 ASOS 관측소 번호
# 일사량을 측정하는 관측소만 사용 (응답의 stnNm 으로 이름을 로그에 출력)
STATION_BY_KEYWORD = {
    "영동": 105,     # 강릉 (영동에코발전본부)
    "삼천포": 192,   # 진주 (남해 관측소는 일사량 미측정)
    "광양": 266,     # 광양시
    "두산엔진": 155,  # 창원
    "구미": 143,     # 대구 (구미 관측소는 일사량 미측정)
    "영흥": 112,     # 인천
    "예천": 136,     # 안동
    "경상대": 192,   # 진주
    "고흥": 262,     # 고흥
    "여수": 168,     # 여수
}

# 일자료에서 쓸 항목 (기상청 항목명 -> 우리 컬럼명)
DAILY_COLS = {
    "sumGsr": "일사량합",       # MJ/m2
    "sumSsHr": "일조시간",      # hr
    "avgTca": "평균운량",       # 0~10
    "avgTa": "평균기온",
    "maxTa": "최고기온",
    "minTa": "최저기온",
    "sumRn": "강수량",
    "avgRhm": "평균습도",
    "avgWs": "평균풍속",
    "ddMes": "적설",
}
HOURLY_COLS = {
    "icsr": "일사량",     # MJ/m2
    "ss": "일조",         # hr
    "dc10Tca": "운량",
    "ta": "기온",
    "rn": "강수량",
    "hm": "습도",
    "ws": "풍속",
}


def call(url, params, retries=3):
    q = urllib.parse.urlencode({"serviceKey": KEY, **params})
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # 네트워크 오류 등은 잠깐 쉬고 재시도
            msg = str(e).replace(KEY, "<KEY>")
            print(f"    재시도 {i + 1}/{retries}: {msg}")
            time.sleep(3)
    raise RuntimeError(f"요청 실패: {url}")


def fetch_solar_month(year, month):
    path = RAW / f"solar_{year}{month:02d}.csv"
    if path.exists():
        return
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    rows, page = [], 1
    while True:
        d = call(SOLAR_URL, {"page": page, "size": 100,
                             "startD": f"{start:%Y%m%d}", "endD": f"{end:%Y%m%d}"})
        body = d["reponse"]["body"]  # 원본 API의 오타(reponse) 그대로
        rows += body.get("content") or []
        if body.get("last") or not body.get("content"):
            break
        page += 1
    df = pd.DataFrame(rows).drop(columns=["eaidml", "eaiflag", "eaimsg"], errors="ignore")
    df.to_csv(path, index=False)
    print(f"  태양광 {year}-{month:02d}: {len(df)}행")


def fetch_weather(kind, stn, year):
    path = RAW / f"wx_{kind}_{stn}_{year}.csv"
    if path.exists():
        return
    params = {"numOfRows": 999, "dataType": "JSON", "dataCd": "ASOS", "stnIds": stn,
              "startDt": f"{year}0101", "endDt": f"{year}1231"}
    if kind == "hourly":
        url = WX_HOURLY_URL
        params.update(dateCd="HR", startHh="00", endHh="23")
    else:
        url = WX_DAILY_URL
        params.update(dateCd="DAY")
    rows, page = [], 1
    while True:
        d = call(url, {**params, "pageNo": page})
        if d["response"]["header"]["resultCode"] != "00":
            print(f"    {stn} {year} {kind}: {d['response']['header']}")
            break
        body = d["response"]["body"]
        items = body["items"]["item"] if body.get("items") else []
        rows += items
        if page * 999 >= int(body["totalCount"]):
            break
        page += 1
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    name = df["stnNm"].iloc[0] if len(df) else "?"
    print(f"  날씨 {kind} {stn}({name}) {year}: {len(df)}행")


def station_for(plant_name):
    for kw, stn in STATION_BY_KEYWORD.items():
        if kw in plant_name:
            return stn
    return None


def build_tables(years):
    solar = pd.concat(
        [pd.read_csv(p, dtype={"ippt": str, "hogi": str})
         for p in sorted(RAW.glob("solar_*.csv"))
         if int(p.stem[6:10]) in years],
        ignore_index=True,
    )
    solar["ipptNam"] = solar["ipptNam"].str.strip()
    solar["stn"] = solar["ipptNam"].map(station_for)
    unmapped = solar.loc[solar["stn"].isna(), "ipptNam"].unique()
    if len(unmapped):
        print("  관측소 미지정(제외):", list(unmapped))
    solar = solar.dropna(subset=["stn"]).astype({"stn": int})
    solar["date"] = pd.to_datetime(solar["dgenYmd"])
    solar["unit"] = solar["ipptNam"] + "#" + solar["hogi"]

    # ---- 하루 단위 ----
    wx = []
    for p in RAW.glob("wx_daily_*.csv"):
        w = pd.read_csv(p)
        if w.empty:
            continue
        w = w[["stnId", "stnNm", "tm", *[c for c in DAILY_COLS if c in w]]]
        wx.append(w.rename(columns={"stnId": "stn", "stnNm": "관측소", **DAILY_COLS}))
    wx = pd.concat(wx, ignore_index=True)
    wx["date"] = pd.to_datetime(wx["tm"])
    daily = solar[["unit", "ipptNam", "hogi", "stn", "date", "qsum"]].merge(
        wx.drop(columns="tm"), on=["stn", "date"], how="left"
    ).rename(columns={"ipptNam": "발전소", "hogi": "호기", "qsum": "발전량"})
    daily = daily.sort_values(["unit", "date"])
    daily.to_csv(OUT / "solar_daily.csv", index=False)
    print(f"  -> solar_daily.csv: {len(daily)}행, 발전기 {daily['unit'].nunique()}대")

    # ---- 시간 단위 (qhorGenHH = HH시 발전량, 기상청 tm 'HH:00'과 맞춤) ----
    hcols = [f"qhorGen{h:02d}" for h in range(1, 25)]
    long = solar.melt(id_vars=["unit", "ipptNam", "hogi", "stn", "date"], value_vars=hcols,
                      var_name="h", value_name="발전량")
    long["time"] = long["date"] + pd.to_timedelta(long["h"].str[-2:].astype(int), unit="h")
    wxh = []
    for p in RAW.glob("wx_hourly_*.csv"):
        w = pd.read_csv(p)
        if w.empty:
            continue
        w = w[["stnId", "tm", *[c for c in HOURLY_COLS if c in w]]]
        wxh.append(w.rename(columns={"stnId": "stn", **HOURLY_COLS}))
    wxh = pd.concat(wxh, ignore_index=True)
    wxh["time"] = pd.to_datetime(wxh["tm"])
    hourly = long.drop(columns=["h", "date"]).merge(
        wxh.drop(columns="tm"), on=["stn", "time"], how="left"
    ).rename(columns={"ipptNam": "발전소", "hogi": "호기"})
    hourly = hourly.sort_values(["unit", "time"])
    hourly.to_csv(OUT / "solar_hourly.csv", index=False)
    print(f"  -> solar_hourly.csv: {len(hourly)}행")


def main():
    y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (2023, 2024)
    years = range(y0, y1 + 1)
    RAW.mkdir(parents=True, exist_ok=True)

    print("[1/3] 태양광 발전량")
    for y in years:
        for m in range(1, 13):
            fetch_solar_month(y, m)

    print("[2/3] 기상청 날씨")
    for stn in sorted(set(STATION_BY_KEYWORD.values())):
        for y in years:
            fetch_weather("daily", stn, y)
            fetch_weather("hourly", stn, y)

    print("[3/3] 표 합치기")
    build_tables(set(years))


if __name__ == "__main__":
    main()
