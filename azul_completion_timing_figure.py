"""Render the seeding-harvest completion timing figure as vector PDF."""

import csv
import math
import os

from reportlab.lib.colors import Color, HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


INPUT = "azul_completion_timing.csv"
OUTPUT = "azul_completion_timing.pdf"
EPISODES = 100_000
STEPS = 45

INK = HexColor("#243247")
MUTED = HexColor("#667085")
GRID = HexColor("#E5E9F0")
BLUE = HexColor("#5167C9")
BLUE_FILL = Color(0.32, 0.40, 0.79, alpha=0.20)
ORANGE = HexColor("#D97706")
PANEL = HexColor("#F8FAFC")


def load_counts(path):
    values = [[[0] * STEPS for _ in range(8)] for _ in range(2)]
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[int(row["side"]) - 1][int(row["window"]) - 1][
                int(row["step"])
            ] = int(row["count"])
    return values


def smooth(hist):
    kernel = [0.027, 0.066, 0.123, 0.180, 0.208,
              0.180, 0.123, 0.066, 0.027]
    result = []
    for index in range(len(hist)):
        total = 0.0
        for offset, weight in enumerate(kernel):
            j = index + offset - 4
            if 0 <= j < len(hist):
                total += weight * hist[j]
        result.append(total)
    return result


def quantile(hist, probability):
    total = sum(hist)
    target = probability * (total - 1) + 1
    cumulative = 0
    for index, count in enumerate(hist):
        cumulative += count
        if cumulative >= target:
            return index
    return len(hist) - 1


def summarize(hist):
    total = sum(hist)
    mean = sum(step * count for step, count in enumerate(hist)) / total
    return {
        "probability": total / EPISODES,
        "mean": mean,
        "q10": quantile(hist, 0.10),
        "q25": quantile(hist, 0.25),
        "q75": quantile(hist, 0.75),
        "q90": quantile(hist, 0.90),
    }


def draw_panel(c, counts, x0, y0, width, height, title, side):
    left = 38
    right = 12
    bottom = 44
    top = 30
    plot_x = x0 + left
    plot_y = y0 + bottom
    plot_w = width - left - right
    plot_h = height - bottom - top

    def x(window):
        return plot_x + (window - 0.5) * plot_w / 8

    def y(step):
        return plot_y + step / STEPS * plot_h

    c.setFillColor(PANEL)
    c.roundRect(x0, y0, width, height, 7, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("NotoSansSC", 12.5)
    c.drawCentredString(x0 + width / 2, y0 + height - 18, title)

    c.setStrokeColor(GRID)
    c.setLineWidth(0.55)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.5)
    for step in range(0, 46, 5):
        yy = y(step)
        c.line(plot_x, yy, plot_x + plot_w, yy)
        c.drawRightString(plot_x - 6, yy - 2.5, str(step))

    c.setStrokeColor(INK)
    c.setLineWidth(0.8)
    c.line(plot_x, plot_y, plot_x, plot_y + plot_h)
    c.line(plot_x, plot_y, plot_x + plot_w, plot_y)
    for window in range(1, 9):
        xx = x(window)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(xx, plot_y - 13, str(window))

    c.setFont("NotoSansSC", 8.5)
    c.drawCentredString(plot_x + plot_w / 2, y0 + 10, "窗户编号")
    c.saveState()
    c.translate(x0 + 10, plot_y + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "行动步")
    c.restoreState()

    summaries = [summarize(hist) for hist in counts]
    if side == 0:
        order = sorted(range(8), key=lambda index: summaries[index]["mean"])
        rank = {window: position + 1 for position, window in enumerate(order)}
        c.setStrokeColor(ORANGE)
        c.setLineWidth(1.2)
        c.setDash(4, 3)
        path = c.beginPath()
        first = True
        for window in order:
            xx = x(window + 1)
            yy = y(summaries[window]["mean"])
            if first:
                path.moveTo(xx, yy)
                first = False
            else:
                path.lineTo(xx, yy)
        c.drawPath(path, fill=0, stroke=1)
        c.setDash()
    else:
        rank = {}

    for index, hist in enumerate(counts):
        window = index + 1
        xx = x(window)
        summary = summaries[index]
        density = smooth(hist)
        maximum = max(density)
        half_max = 10.8 * math.sqrt(summary["probability"])
        left_points = []
        right_points = []
        visible = [
            step for step, value in enumerate(density)
            if maximum and value / maximum >= 0.003
        ]
        start = max(0, min(visible) - 1)
        end = min(STEPS - 1, max(visible) + 1)
        for step in range(start, end + 1):
            value = density[step]
            half = half_max * value / maximum if maximum else 0
            left_points.append((xx - half, y(step)))
            right_points.append((xx + half, y(step)))

        path = c.beginPath()
        path.moveTo(*left_points[0])
        for point in left_points[1:]:
            path.lineTo(*point)
        for point in reversed(right_points):
            path.lineTo(*point)
        path.close()
        c.setFillColor(BLUE_FILL)
        c.setStrokeColor(BLUE)
        c.setLineWidth(0.75)
        c.drawPath(path, fill=1, stroke=1)

        c.setStrokeColor(BLUE)
        c.setLineWidth(0.75)
        c.line(xx, y(summary["q10"]), xx, y(summary["q90"]))
        c.setLineWidth(3.1)
        c.setLineCap(1)
        c.line(xx, y(summary["q25"]), xx, y(summary["q75"]))
        c.setLineCap(0)

        radius = 2.8 if side == 0 else max(0.7, 4.8 * math.sqrt(summary["probability"]))
        c.setFillColor(ORANGE)
        c.setStrokeColor(ORANGE)
        c.circle(xx, y(summary["mean"]), radius, fill=1, stroke=0)
        if side == 0:
            c.setFont("Helvetica-Bold", 7)
            c.drawString(xx + 4.2, y(summary["mean"]) + 3.2, f"#{rank[index]}")
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6.5)
        digits = 0 if side == 0 else 1
        c.drawCentredString(
            xx, plot_y - 25,
            f"{100 * summary['probability']:.{digits}f}%"
        )

    c.setFillColor(MUTED)
    c.setFont("NotoSansSC", 6.8)
    c.drawRightString(plot_x - 5, plot_y - 25, "完成率")


def main():
    font_path = os.environ.get(
        "AZUL_CJK_FONT", "tmp/fonts/NotoSansSC-Regular.ttf"
    )
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            "Chinese font not found. Set AZUL_CJK_FONT to a Noto Sans SC-compatible "
            "TTF/OTF path before running this script."
        )
    pdfmetrics.registerFont(TTFont("NotoSansSC", font_path))
    values = load_counts(INPUT)
    page_w, page_h = 700, 350
    c = canvas.Canvas(OUTPUT, pagesize=(page_w, page_h), pageCompression=1)
    c.setTitle("Azul completion timing distributions")
    margin = 18
    gap = 14
    panel_w = (page_w - 2 * margin - gap) / 2
    panel_h = page_h - 36
    draw_panel(
        c, values[0], margin, 25, panel_w, panel_h,
        "第一面：右端播种", 0
    )
    draw_panel(
        c, values[1], margin + panel_w + gap, 25, panel_w, panel_h,
        "第二面：左侧收割", 1
    )
    c.setFillColor(MUTED)
    c.setFont("NotoSansSC", 6.8)
    c.drawCentredString(
        page_w / 2, 8,
        "小提琴宽度按完成概率平方根缩放；细线为10%-90%，粗线为25%-75%，橙点为条件均值。"
    )
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
