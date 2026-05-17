from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Iterable

from .utils import OLLAMA_CHAT_URL, OLLAMA_MODEL, fmt

FEEDBACK_SYSTEM = (
    "Ты вокальный педагог, который пишет только финальный отчёт ученику. "
    "Запрещено выводить внутренний анализ, черновик, самопроверку, рассуждения о том, что нужно сделать, "
    "и фразы вроде 'проверю', 'теперь структурирую', 'нужно объяснить', 'пишем'. "
    "Начинай строго с заголовка '### 1. Общая картина'. "
    "Пиши по-русски, поддерживающим тоном, подробно, но без медицинских диагнозов. "
    "Технические признаки объясняй простым языком и связывай только с педагогическими наблюдениями."
)

FEEDBACK_USER_TEMPLATE = """\
Сформируй развёрнутый педагогический фидбэк для ученика вокала по методике Estill Voice Training.

У тебя есть два источника:
1. Педагогический профиль — уже отобранные выводы, которые нужно использовать как основу.
2. Технический отчёт — численные метрики для точности формулировок.

Жёсткие требования к ответу:
- Верни только финальный Markdown-ответ, без черновика и без рассуждений о процессе.
- Не повторяй заголовки и не пересказывай инструкцию.
- Не делай медицинских выводов; формулируй как вокально-педагогические наблюдения.
- Если используешь Estill-признаки, объясняй их без жаргона:
  CPP — насколько голос звучит собранно и чисто;
  H1-H2 — насколько смыкание голосовых складок открытое или плотное;
  alpha ratio — яркость/звонкость тембра;
  spectral tilt — баланс яркости и мягкости в спектре.
- Не перечисляй все метрики подряд: выбирай только те, которые помогают ученику понять действие.
- В зонах роста укажи, как это может быть слышно, и что именно тренировать.

Обязательная структура и формат:
### 1. Общая картина
2–3 связных предложения: общий уровень, главный сильный навык, главный приоритет.

### 2. Что получилось хорошо
2–3 пункта. Каждый пункт: **навык** — что получилось и почему это важно.

### 3. Главные зоны роста
3–4 пункта. Каждый пункт строго в формате:
- **Зона**: наблюдение → как это может звучать → что улучшать.

### 4. Что делать
3 упражнения. Для каждого упражнения укажи:
- **Цель**
- **Как делать**
- **Самопроверка**

### 5. План на неделю
3 шага с длительностью/частотой занятий и фокусом.

### 6. Короткое резюме ученику
1–2 поддерживающих предложения без новых технических терминов.

Педагогический профиль:
{pedagogical_brief}

Технический отчёт:
{report}
"""

_META_LINE_RE = re.compile(
    r"^\s*(?:"
    r"проверю|теперь\s+структурирую|нужно\s+объяснить|нужно\s+избегать|"
    r"пишем|возможно,?\s+ученик|как-будто|llm feedback|"
    r"структурирую\s+ответ|сейчас\s+сформирую|думаю|рассуждение"
    r")\b.*$",
    flags=re.IGNORECASE,
)

_EXPECTED_HEADINGS = [
    "### 1. Общая картина",
    "### 2. Что получилось хорошо",
    "### 3. Главные зоны роста",
    "### 4. Что делать",
    "### 5. План на неделю",
    "### 6. Короткое резюме ученику",
]


def _finite(value) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _num(value, digits: int = 1, fallback: str = "—") -> str:
    return fmt(value, digits) if _finite(value) else fallback


def _pct(value, digits: int = 1) -> str:
    return f"{_num(value, digits)}%" if _finite(value) else "—"


def _score_label(score: float) -> str:
    if not _finite(score):
        return "нет достаточных данных"
    if score >= 85:
        return "сильная сторона"
    if score >= 75:
        return "хорошая база"
    if score >= 60:
        return "рабочая зона"
    return "приоритетная зона роста"


def _metric_line(title: str, score_key: str, details: str, metrics: dict) -> str:
    score = metrics.get(score_key)
    return f"- {title}: {_num(score)}/100 ({_score_label(score)}). {details}"


def _ranked_scores(metrics: dict) -> list[tuple[str, str, float, str]]:
    rows = [
        (
            "Интонация",
            "intonation_score",
            metrics.get("intonation_score", math.nan),
            f"средняя ошибка высоты {_num(metrics.get('pitch_mean_abs_cents'))} центов, попадание в ±50 центов — {_pct(metrics.get('in_tune_50_pct'))}",
        ),
        (
            "Ритм",
            "rhythm_score",
            metrics.get("rhythm_score", math.nan),
            f"расхождение вступлений {_num(metrics.get('onset_mae_ms'), 0)} мс, длительностей — {_num(metrics.get('duration_mae_ms'), 0)} мс",
        ),
        (
            "Атака звука",
            "attack_score",
            metrics.get("attack_score", math.nan),
            f"начало звука: эталон {_num(metrics.get('attack_rise_teacher_ms'), 0)} мс, ученик {_num(metrics.get('attack_rise_student_ms'), 0)} мс",
        ),
        (
            "Дыхание и поддержка",
            "breath_score",
            metrics.get("breath_score", math.nan),
            f"разница voiced ratio {_num(metrics.get('voiced_ratio_diff'), 3)}, средние паузы ученика {_num(metrics.get('silent_gap_mean_student_s'))} с",
        ),
        (
            "Смыкание и голосовой контроль",
            "voice_closure_score",
            metrics.get("voice_closure_score", math.nan),
            f"HNR ученика отличается от эталона на {_num(metrics.get('hnr_diff_db'))} дБ, H1-H2 ученика {_num(metrics.get('h1h2_mean_student_db'))} дБ",
        ),
    ]
    return rows


def _has_closure_warning(metrics: dict) -> bool:
    return bool(
        (_finite(metrics.get("hnr_diff_db")) and metrics["hnr_diff_db"] < -2)
        or (
            _finite(metrics.get("h1h2_mean_student_db"))
            and _finite(metrics.get("h1h2_mean_teacher_db"))
            and abs(metrics["h1h2_mean_student_db"] - metrics["h1h2_mean_teacher_db"])
            > 4
        )
        or (
            _finite(metrics.get("cpp_mean_student"))
            and _finite(metrics.get("cpp_mean_teacher"))
            and metrics["cpp_mean_student"] - metrics["cpp_mean_teacher"] < -0.15
        )
    )


def _select_strengths(metrics: dict) -> list[str]:
    rows = [row for row in _ranked_scores(metrics) if _finite(row[2])]
    rows.sort(key=lambda item: item[2], reverse=True)
    strengths = []
    for title, key, score, details in rows:
        if title == "Смыкание и голосовой контроль" and _has_closure_warning(metrics):
            continue
        if score >= 75:
            strengths.append(_metric_line(title, key, details, metrics))
        if len(strengths) == 3:
            break
    if len(strengths) < 2:
        for title, key, _score, details in rows:
            if title == "Смыкание и голосовой контроль" and _has_closure_warning(
                metrics
            ):
                continue
            candidate = _metric_line(title, key, details, metrics)
            if candidate not in strengths:
                strengths.append(candidate)
            if len(strengths) == 2:
                break
    return strengths or [
        "- Стабильность выполнения: данных недостаточно для уверенного вывода, поэтому фокус лучше держать на аккуратном повторении эталона."
    ]


def _select_growth_zones(metrics: dict, flags: Iterable[str]) -> list[str]:
    zones: list[tuple[int, str]] = []
    flag_text = "\n".join(flags).lower()

    if (
        metrics.get("rhythm_score", 100) < 75
        or "ритм" in flag_text
        or "длительности" in flag_text
    ):
        zones.append(
            (
                10,
                "- **Ритм и длительности**: вступления и удержание нот расходятся с эталоном "
                f"(onset MAE {_num(metrics.get('onset_mae_ms'), 0)} мс, duration MAE {_num(metrics.get('duration_mae_ms'), 0)} мс) "
                "→ фраза может звучать собранной по высоте, но не совпадать с пульсом упражнения → тренировать входы под счёт и одинаковую длину нот.",
            )
        )

    if (
        metrics.get("intonation_score", 100) < 80
        or metrics.get("pitch_mean_abs_cents", 0) > 35
    ):
        direction = "выше" if metrics.get("pitch_bias_cents", 0) > 0 else "ниже"
        zones.append(
            (
                9,
                "- **Интонация**: среднее отклонение высоты "
                f"{_num(metrics.get('pitch_mean_abs_cents'))} центов, общий сдвиг {direction} эталона "
                "→ отдельные ноты могут звучать слегка не в центре → тренировать медленное попадание в целевую высоту и удержание без сползания.",
            )
        )

    if metrics.get("attack_score", 100) < 78 or "атака" in flag_text:
        zones.append(
            (
                8,
                "- **Атака звука**: начало нот отличается от эталона "
                f"(у ученика подъём громкости {_num(metrics.get('attack_rise_student_ms'), 0)} мс против {_num(metrics.get('attack_rise_teacher_ms'), 0)} мс у эталона) "
                "→ вход может восприниматься слишком резким или, наоборот, запаздывающим → тренировать одинаковое мягкое начало каждой ноты.",
            )
        )

    closure_reasons = []
    if _finite(metrics.get("hnr_diff_db")) and metrics["hnr_diff_db"] < -2:
        closure_reasons.append(
            f"HNR ниже эталона на {_num(abs(metrics['hnr_diff_db']))} дБ"
        )
    if _finite(metrics.get("cpp_mean_student")) and _finite(
        metrics.get("cpp_mean_teacher")
    ):
        cpp_delta = metrics["cpp_mean_student"] - metrics["cpp_mean_teacher"]
        if cpp_delta < -0.15:
            closure_reasons.append(f"CPP ниже эталона на {_num(abs(cpp_delta))} дБ")
    if _finite(metrics.get("h1h2_mean_student_db")) and _finite(
        metrics.get("h1h2_mean_teacher_db")
    ):
        h_delta = metrics["h1h2_mean_student_db"] - metrics["h1h2_mean_teacher_db"]
        if abs(h_delta) > 4:
            closure_reasons.append(
                f"H1-H2 отличается от эталона на {_num(abs(h_delta))} дБ"
            )
    if metrics.get("voice_closure_score", 100) < 82 or closure_reasons:
        zones.append(
            (
                7,
                "- **Чистота тона и смыкание**: "
                + (
                    ", ".join(closure_reasons)
                    if closure_reasons
                    else f"оценка смыкания {_num(metrics.get('voice_closure_score'))}/100"
                )
                + " → звук может становиться менее собранным, с лишним воздухом или лишним нажимом → искать более ровное, экономное смыкание без давления.",
            )
        )

    if metrics.get("breath_score", 100) < 78:
        zones.append(
            (
                6,
                "- **Дыхание и поддержка**: доля звучащих участков и паузы отличаются от эталона "
                f"(voiced ratio diff {_num(metrics.get('voiced_ratio_diff'), 3)}) "
                "→ фраза может терять непрерывность или опору → тренировать распределение воздуха на всю длительность звука.",
            )
        )

    if (
        _finite(metrics.get("alpha_ratio_diff_db"))
        and abs(metrics["alpha_ratio_diff_db"]) > 3.5
    ):
        tone = "ярче" if metrics["alpha_ratio_diff_db"] > 0 else "темнее"
        zones.append(
            (
                5,
                "- **Тембровая яркость**: alpha ratio показывает, что голос ученика "
                f"{tone} эталона на {_num(abs(metrics['alpha_ratio_diff_db']))} дБ "
                "→ тембр может отличаться по звонкости и собранности → подбирать яркость постепенно, не меняя высоту и ритм.",
            )
        )

    zones.sort(key=lambda item: item[0], reverse=True)
    selected = [text for _, text in zones[:4]]
    if len(selected) < 3:
        selected.append(
            "- **Стабильность повторения**: даже при хороших отдельных метриках важно закрепить одинаковое качество на каждом повторе → слушатель должен слышать не случайное попадание, а устойчивый навык → повторять упражнение короткими сериями и сравнивать с эталоном."
        )
    return selected[:4]


def _select_exercises(metrics: dict) -> list[str]:
    exercises: list[tuple[int, str]] = []

    if metrics.get("rhythm_score", 100) < 80:
        exercises.append(
            (
                10,
                "1. **Ритм под счёт**\n"
                "   - **Цель:** выровнять вступления и длительности нот.\n"
                "   - **Как делать:** включи метроном в темпе эталона, сначала проговори ритм на 'та' без пения, затем спой упражнение на одной удобной ноте, сохраняя те же входы. Делай 4–5 коротких повторов, не ускоряясь.\n"
                "   - **Самопроверка:** начало каждой ноты должно совпадать с долей; если запись наложить на эталон, основные входы не должны заметно уезжать.",
            )
        )

    if metrics.get("attack_score", 100) < 85:
        exercises.append(
            (
                8,
                "2. **Единая мягкая атака**\n"
                "   - **Цель:** сделать начало каждой ноты предсказуемым и одинаковым.\n"
                "   - **Как делать:** на удобной высоте пой короткие слоги 'ма-ма-ма' или 'на-на-на': перед каждым звуком ощущай спокойный вдох, затем начинай звук без толчка горлом и без лишнего воздуха.\n"
                "   - **Самопроверка:** первые доли нот не должны звучать как хлопок, скрип или запоздалое 'подъезжание'; громкость набирается ровно.",
            )
        )

    closure_needed = metrics.get(
        "voice_closure_score", 100
    ) < 86 or _has_closure_warning(metrics)
    if closure_needed:
        exercises.append(
            (
                7,
                "3. **Чистый тон без лишнего воздуха**\n"
                "   - **Цель:** найти более собранное смыкание без зажима.\n"
                "   - **Как делать:** спой тихое 'ммм' 2–3 секунды, затем открой в гласную 'ма', сохраняя то же ощущение собранного, спокойного звука. Повтори 5 раз, каждый раз останавливаясь до усталости.\n"
                "   - **Самопроверка:** звук не должен становиться шёпотным, но и не должен давить; ощущение — устойчивый тон на небольшом расходе воздуха.",
            )
        )

    if metrics.get("breath_score", 100) < 82:
        exercises.append(
            (
                6,
                "4. **Распределение воздуха на фразу**\n"
                "   - **Цель:** удерживать звук ровно до конца ноты и не терять поддержку.\n"
                "   - **Как делать:** на одном выдохе спой 4 одинаковые короткие ноты, затем одну длинную. Следи, чтобы последняя нота не проваливалась по громкости и не становилась воздушной.\n"
                "   - **Самопроверка:** в конце фразы качество звука остаётся таким же, как в начале.",
            )
        )

    if metrics.get("intonation_score", 100) < 85:
        exercises.append(
            (
                5,
                "5. **Медленное попадание в высоту**\n"
                "   - **Цель:** точнее попадать в центр каждой ноты.\n"
                "   - **Как делать:** выбери 2–3 проблемных перехода, спой их в два раза медленнее, сначала на 'у', затем на исходной гласной. После каждого повтора сравни с эталоном.\n"
                "   - **Самопроверка:** нота начинается сразу в нужной высоте, без заметного подъезда снизу или сверху.",
            )
        )

    if len(exercises) < 3:
        exercises.append(
            (
                1,
                "6. **Контрольное повторение с эталоном**\n"
                "   - **Цель:** связать интонацию, ритм и качество звука в одном исполнении.\n"
                "   - **Как делать:** послушай эталон один раз, затем спой вместе с ним вполголоса, затем отдельно запиши самостоятельный дубль.\n"
                "   - **Самопроверка:** самостоятельный дубль должен сохранять те же входы, длительности и общий характер звука.",
            )
        )

    exercises.sort(key=lambda item: item[0], reverse=True)
    selected = [text for _, text in exercises[:3]]
    renumbered = []
    for index, item in enumerate(selected, 1):
        renumbered.append(re.sub(r"^\d+\.", f"{index}.", item, count=1))
    return renumbered


def build_pedagogical_brief(metrics: dict, flags: Iterable[str] | None = None) -> str:
    """Build deterministic teaching priorities for the LLM.

    The LLM should phrase these points for a student, not decide priorities from
    scratch. This keeps feedback stable and easier to justify in the thesis.
    """
    flags = list(flags or [])
    lines = [
        "Сводка оценок:",
        _metric_line(
            "Интонация",
            "intonation_score",
            f"mean abs pitch error {_num(metrics.get('pitch_mean_abs_cents'))} cents; in tune ±50c {_pct(metrics.get('in_tune_50_pct'))}",
            metrics,
        ),
        _metric_line(
            "Ритм",
            "rhythm_score",
            f"onset MAE {_num(metrics.get('onset_mae_ms'), 0)} ms; duration MAE {_num(metrics.get('duration_mae_ms'), 0)} ms",
            metrics,
        ),
        _metric_line(
            "Атака",
            "attack_score",
            f"student rise {_num(metrics.get('attack_rise_student_ms'), 0)} ms vs teacher {_num(metrics.get('attack_rise_teacher_ms'), 0)} ms",
            metrics,
        ),
        _metric_line(
            "Дыхание/поддержка",
            "breath_score",
            f"voiced ratio diff {_num(metrics.get('voiced_ratio_diff'), 3)}; silent gap student {_num(metrics.get('silent_gap_mean_student_s'))} s",
            metrics,
        ),
        _metric_line(
            "Смыкание/контроль",
            "voice_closure_score",
            f"HNR diff {_num(metrics.get('hnr_diff_db'))} dB; CPP teacher/student {_num(metrics.get('cpp_mean_teacher'))}/{_num(metrics.get('cpp_mean_student'))}; H1-H2 teacher/student {_num(metrics.get('h1h2_mean_teacher_db'))}/{_num(metrics.get('h1h2_mean_student_db'))}",
            metrics,
        ),
        "",
        "Сильные стороны, которые нужно отметить:",
        *_select_strengths(metrics),
        "",
        "Главные зоны роста по приоритету:",
        *_select_growth_zones(metrics, flags),
        "",
        "Рекомендуемые упражнения:",
        *_select_exercises(metrics),
        "",
        "Приоритетные технические флаги:",
    ]
    if flags:
        lines.extend(f"- {flag}" for flag in flags[:6])
    else:
        lines.append(
            "- Критичных отклонений не найдено; акцент на закреплении стабильности."
        )
    return "\n".join(lines)


def build_rule_based_feedback(metrics: dict, flags: Iterable[str] | None = None) -> str:
    """Generate a complete deterministic feedback report without an LLM."""
    flags = list(flags or [])
    strengths = _select_strengths(metrics)
    growth = _select_growth_zones(metrics, flags)
    exercises = _select_exercises(metrics)

    strength_names = {
        "Интонация": "точную интонацию",
        "Ритм": "ритмическую организацию",
        "Атака звука": "достаточно управляемое начало звука",
        "Дыхание и поддержка": "дыхание и поддержку",
        "Смыкание и голосовой контроль": "голосовой контроль",
    }
    raw_strength = strengths[0].split(":", 1)[0].lstrip("- ") if strengths else ""
    top_strength = strength_names.get(raw_strength, "отдельные сильные навыки")
    main_growth = (
        growth[0].split("**", 2)[1]
        if growth and "**" in growth[0]
        else "стабильность выполнения"
    )

    lines = [
        "### 1. Общая картина",
        f"В исполнении уже есть хорошая база: особенно заметна опора на {top_strength}. Главный приоритет на ближайшую работу — {main_growth.lower()}, потому что именно эта зона сильнее всего влияет на совпадение с эталоном и ощущение уверенного упражнения.",
        "Метрики стоит воспринимать не как оценку личности или голоса, а как подсказку, куда направить внимание на следующей тренировке.",
        "",
        "### 2. Что получилось хорошо",
        *strengths[:3],
        "",
        "### 3. Главные зоны роста",
        *growth,
        "",
        "### 4. Что делать",
        *exercises,
        "",
        "### 5. План на неделю",
        "1. **Дни 1–2 — ритм и карта упражнения:** 10–12 минут в день работай с метрономом и эталоном: сначала проговори ритм, затем спой на одной ноте, затем верни исходные высоты.",
        "2. **Дни 3–5 — качество начала и тона:** 12–15 минут в день чередуй упражнение на атаку и упражнение на чистый тон; записывай по одному короткому дублю и отмечай, где звук начинается ровнее.",
        "3. **Дни 6–7 — контрольная сборка:** сделай 2–3 полных исполнения рядом с эталоном, выбери лучший дубль и сравни по трём критериям: входы вовремя, ноты держатся до конца, звук остаётся собранным.",
        "",
        "### 6. Короткое резюме ученику",
        "У тебя уже есть материал, на который можно опереться; теперь задача — сделать исполнение более предсказуемым и ровным от повтора к повтору. Работай короткими сериями и сравнивай себя с эталоном: так прогресс будет заметнее и стабильнее.",
    ]
    return "\n".join(lines)


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    cleaned_lines = []
    for line in text.splitlines():
        if _META_LINE_RE.match(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    hit = re.search(r"(#{1,3}\s*)?1\.\s*\**\s*Общая картина", text, flags=re.IGNORECASE)
    if hit:
        text = text[hit.start() :]

    # If the model repeats the first heading several times while drafting, keep
    # the last occurrence because it is usually the completed answer.
    starts = list(
        re.finditer(r"(?im)^\s*(?:#{1,3}\s*)?1\.\s*\**\s*Общая картина.*$", text)
    )
    if len(starts) > 1:
        text = text[starts[-1].start() :]

    # Normalize top-level section headings to the expected form.
    for heading in _EXPECTED_HEADINGS:
        number, title = heading.split(". ", 1)
        n = number.replace("### ", "")
        text = re.sub(
            rf"(?im)^\s*(?:#{{1,4}}\s*)?{re.escape(n)}\.\s*\**\s*{re.escape(title)}\**\s*$",
            heading,
            text,
        )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_feedback(
    text_report: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_CHAT_URL,
    max_tokens: int = 1800,
    pedagogical_brief: str | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": FEEDBACK_SYSTEM},
            {
                "role": "user",
                "content": FEEDBACK_USER_TEMPLATE.format(
                    pedagogical_brief=pedagogical_brief
                    or "Педагогический профиль не передан; используй технический отчёт осторожно.",
                    report=text_report,
                ),
            },
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.12,
            "top_p": 0.85,
            "num_ctx": 8192,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Не удалось подключиться к Ollama ({url}). "
            f"Убедись, что сервер запущен и модель установлена: ollama pull {model}"
        ) from exc

    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama вернул пустой ответ: {data}")
    return _strip_thinking(content)
