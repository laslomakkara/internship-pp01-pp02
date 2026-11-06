"""Анализ A/B-теста системы достижений из командной строки."""

import argparse
from pathlib import Path
from math import asin, erfc, sqrt, sin
from statistics import NormalDist
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


START = pd.Timestamp("2026-05-04")
END = START + pd.Timedelta(days=21)
ACTIVITY = ["login", "homework_started", "homework_completed"]
RETURNS = {"D7": "return_visit_d7", "D14": "return_visit_d14"}
ALPHA = 0.05
BOT_LIMIT = 100
METRICS = {
    "D7": "Удержание D7", "D14": "Удержание D14", "f_done": "Выполнил ДЗ в воронке",
    "done3": "≥3 событий выполнения", "done5": "≥5 событий выполнения"
}
normal = NormalDist()


def print_table(title, table):
    print(f"\n{title}\n{table.to_string()}")


def split_check(frame):
    a, b = frame.group.value_counts().reindex(["A", "B"])
    expected = (a + b) / 2
    chi2 = ((a - expected)**2 + (b - expected)**2) / expected
    return int(a), int(b), chi2, erfc(sqrt(chi2 / 2))


def student_metrics(assignments, activity):
    counts = (activity.groupby(["user_id", "event_type"]).size().unstack(fill_value=0)
              .reindex(index=assignments.index, columns=ACTIVITY + list(RETURNS.values()),
                       fill_value=0))
    result = assignments.join(counts)
    result["assigned"] = True
    result["D7"] = result[RETURNS["D7"]].gt(0)
    result["D14"] = result[RETURNS["D14"]].gt(0)
    result["done3"] = result.homework_completed.ge(3)
    result["done5"] = result.homework_completed.ge(5)

    previous = None
    for step, event_type in zip(["f_login", "f_start", "f_done"], ACTIVITY):
        candidates = activity.loc[activity.event_type.eq(event_type), ["user_id", "timestamp"]]
        if previous is not None:
            candidates = candidates.join(previous.rename("prior"), on="user_id")
            candidates = candidates.loc[candidates.timestamp.ge(candidates.prior)]
        previous = candidates.groupby("user_id").timestamp.min()
        result[step] = result.index.isin(previous.index)
    return result


def font(size):
    for candidate in ("arial.ttf", "DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def grouped_bars(table, title, limit, suffix):
    image = Image.new("RGB", (1040, 140 + 82 * len(table)), "white")
    draw = ImageDraw.Draw(image)
    ink, colors = "#132339", {"A": "#47709C", "B": "#087F71"}
    draw.text((32, 21), title, fill=ink, font=font(27))
    for j, group in enumerate(("A", "B")):
        draw.text((34 + 166*j, 65), f"{group}   {'Контроль' if j == 0 else 'Тест'}",
                  fill=colors[group], font=font(19))
    for i, (label, values) in enumerate(table.iterrows()):
        y = 111 + 82*i
        draw.text((32, y + 13), str(label), fill=ink, font=font(18))
        for j, group in enumerate(("A", "B")):
            value = float(values[group])
            x0, y0 = 313, y + 27*j
            x1 = x0 + round(610 * value / limit)
            if x1 > x0:
                draw.rectangle((x0, y0, x1, y0 + 17), fill=colors[group])
            draw.text((x1 + 9, y0 - 2), f"{value:.1f}{suffix}", fill=ink, font=font(17))
    return image


def wilson(p, n, alpha=ALPHA):
    z = normal.inv_cdf(1 - alpha / 2)
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return center - radius, center + radius


def power(h, n_a, n_b, alpha):
    shift = h / sqrt(1/n_a + 1/n_b)
    critical = normal.inv_cdf(1 - alpha/2)
    return normal.cdf(-critical - shift) + normal.cdf(shift - critical)


def mde(p_a, n_a, n_b, alpha):
    base = asin(sqrt(p_a))
    lo, hi = 0., 2*(asin(1) - base)
    if power(hi, n_a, n_b, alpha) < .80:
        return np.nan
    for _ in range(45):
        mid = (lo + hi) / 2
        if power(mid, n_a, n_b, alpha) < .80:
            lo = mid
        else:
            hi = mid
    return sin(base + hi/2)**2 - p_a


def difference(frame, metric):
    grouped = frame.groupby("group")[metric]
    n_a, n_b = grouped.size().reindex(["A", "B"])
    s_a, s_b = grouped.sum().reindex(["A", "B"])
    a, b = s_a/n_a, s_b/n_b
    pooled = (s_a + s_b)/(n_a + n_b)
    se = sqrt(pooled*(1-pooled)*(1/n_a + 1/n_b))
    delta = b - a
    p = erfc(abs(delta/se)/sqrt(2)) if se else 1.
    la, ua = wilson(a, n_a)
    lb, ub = wilson(b, n_b)
    return {
        "n_A": int(n_a), "n_B": int(n_b), "s_A": int(s_a), "s_B": int(s_b),
        "A, %": 100*a, "B, %": 100*b, "Δ, п.п.": 100*delta,
        "ДИ от": 100*(delta - sqrt((b-lb)**2 + (ua-a)**2)),
        "ДИ до": 100*(delta + sqrt((ub-b)**2 + (a-la)**2)),
        "p": p, "MDE, п.п.": 100*mde(a, n_a, n_b, ALPHA/2 if metric in RETURNS else ALPHA)
    }


def retention_segments(groups):
    rows = []
    for label, frame in groups:
        for metric in RETURNS:
            effect = difference(frame, metric)
            rows.append({"Класс": str(label), "Метрика": metric,
                         **{k: effect[k] for k in ["n_A", "n_B", "A, %", "B, %",
                                                    "Δ, п.п.", "ДИ от", "ДИ до"]}})
    return pd.DataFrame(rows)


def analyze(data_dir, charts_dir):
    # Загрузка CSV
    users_raw = pd.read_csv(data_dir / "ab_test_users_groups.csv", dtype={"user_id": "string"})
    events_raw = pd.read_csv(data_dir / "ab_test_events.csv", dtype={"user_id": "string"})
    print(f"Строки с назначениями: {len(users_raw):,}. Строки событий: {len(events_raw):,}.")

    # Очистка и проверка сплита
    users = users_raw.drop_duplicates().copy()
    events = events_raw.drop_duplicates().copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    signup = pd.to_datetime(users["signup_date"], errors="coerce")
    assert users["user_id"].notna().all() and users["group"].isin(["A", "B"]).all()
    assert users["grade"].isin(range(5, 12)).all() and signup.lt(START).all()
    overlap = users.groupby("user_id")["group"].nunique().gt(1)
    ambiguous_ids = overlap[overlap].index
    eligible = users.loc[~users.user_id.isin(ambiguous_ids)].set_index("user_id")
    assert eligible.index.is_unique, "Ожидается одно назначение на пользователя"
    valid_time = events.timestamp.ge(START) & events.timestamp.lt(END)
    valid_type = events.event_type.isin(ACTIVITY + list(RETURNS.values()))
    known_id = events.user_id.isin(users.user_id)
    usable = events.loc[valid_time & valid_type & known_id].copy()
    linked = usable.merge(eligible[["group"]], left_on="user_id", right_index=True,
                          how="inner", validate="many_to_one")
    volume = (linked.loc[linked.event_type.isin(ACTIVITY)].groupby("user_id").size()
              .reindex(eligible.index, fill_value=0))
    bot_flag = volume.gt(BOT_LIMIT)
    clean = eligible.loc[~bot_flag]
    logs = linked.loc[linked.user_id.isin(clean.index)]
    for day, event_type in [(7, RETURNS["D7"]), (14, RETURNS["D14"])]:
        days = logs.loc[logs.event_type.eq(event_type), "timestamp"].dt.normalize()
        assert days.eq(START + pd.Timedelta(days=day)).all()
    audit = pd.Series({
        "Точные дубликаты назначений": len(users_raw) - len(users),
        "Точные дубликаты событий": len(events_raw) - len(events),
        "В обеих группах, учеников": len(ambiguous_ids),
        "Нераспознанное время или вне окна, событий": int((~valid_time).sum()),
        "Неизвестный тип, событий": int((~valid_type).sum()),
        "Пользователь без назначения, событий": int((~known_id).sum()),
        "События исключённых из-за сплита": len(usable) - len(linked),
        "Аномальные аккаунты": int(bot_flag.sum()),
        "События аномальных аккаунтов": len(linked) - len(logs),
        "Максимум событий ниже порога": int(volume[~bot_flag].max()),
        "Минимум событий выше порога": int(volume[bot_flag].min()),
        "События для анализа": len(logs),
    })
    print_table("Качество данных", audit.to_frame("Количество"))
    srm = pd.DataFrame([split_check(eligible), split_check(clean)],
                       index=["До фильтра активности", "После фильтра активности"],
                       columns=["A", "B", "χ²", "p-value"])
    print_table("Проверка SRM", srm.round(4))
    print_table("Состав групп по классам", pd.crosstab(clean.grade, clean.group))

    # Метрики и воронка
    students = student_metrics(clean, logs)
    with_outliers = student_metrics(eligible, linked)
    steps = {
        "assigned": "Назначены", "f_login": "Вошли",
        "f_start": "Начали ДЗ после входа", "f_done": "Выполнили ДЗ после старта"
    }
    funnel_counts = students.groupby("group")[list(steps)].sum().T.rename(index=steps)
    funnel_rates = funnel_counts.div(funnel_counts.iloc[0]) * 100
    funnel_previous = funnel_counts.div(funnel_counts.shift()) * 100
    print_table("Воронка", pd.concat({"Ученики": funnel_counts, "% от назначенных": funnel_rates.round(2),
                       "% от предыдущего шага": funnel_previous.round(2)}, axis=1))
    assert (np.diff(funnel_counts.to_numpy(), axis=0) <= 0).all()
    assert len(students) == students.index.nunique()

    # Графики
    charts_dir.mkdir(parents=True, exist_ok=True)
    grouped_bars(funnel_rates, "Воронка за 21 день", 100, "%").save(charts_dir / "funnel.png")
    bins = pd.cut(students.homework_completed, [-1, 0, 2, 4, 7, float("inf")],
                  labels=["0", "1–2", "3–4", "5–7", "8 и более"])
    distribution = pd.crosstab(bins, students.group, normalize="columns") * 100
    print_table("Распределение выполнения ДЗ, %", distribution.round(2))
    grouped_bars(distribution, "Число событий выполнения ДЗ на ученика", 100, "%").save(
        charts_dir / "homework_distribution.png")
    print(f"Графики сохранены в {charts_dir}")

    # Проверка гипотез
    results = pd.DataFrame({metric: difference(students, metric) for metric in METRICS}).T
    p_sorted = results.loc[list(RETURNS), "p"].sort_values()
    results["p Холма"] = (p_sorted * np.arange(len(p_sorted), 0, -1)).cummax().clip(upper=1)
    result_table = results.rename(index=METRICS).round(2)
    for column in ["p", "p Холма"]:
        result_table[column] = results.rename(index=METRICS)[column].map(
            lambda value: f"{value:.3g}" if pd.notna(value) else "—")
    for column in ["n_A", "n_B", "s_A", "s_B"]:
        result_table[column] = result_table[column].astype(int)
    print_table("Эффекты и доверительные интервалы", result_table)
    sensitivity = pd.DataFrame({
        "После очистки, п.п.": results.loc[list(RETURNS), "Δ, п.п."],
        "С аномальными аккаунтами, п.п.": {
            m: difference(with_outliers, m)["Δ, п.п."] for m in RETURNS
        }
    })
    print_table("Чувствительность к фильтру активности", sensitivity.round(2))

    # Разбивка по классам
    bands = pd.cut(students.grade, [4, 7, 9, 11], labels=["5–7", "8–9", "10–11"])
    by_grade = retention_segments(students.groupby("grade"))
    by_band = retention_segments(students.groupby(bands, observed=True))
    print_table("Разбивка по классам", by_grade.round(2))
    print_table("Разбивка по группам классов", by_band.round(2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Папка с исходными CSV (по умолчанию data/).")
    parser.add_argument("--charts-dir", type=Path, default=Path("data/analysis_charts"),
                        help="Папка для PNG-графиков.")
    args = parser.parse_args()
    try:
        analyze(args.data_dir, args.charts_dir)
    except FileNotFoundError as error:
        parser.exit(2, f"Файл не найден: {error.filename}\n")


if __name__ == "__main__":
    main()


