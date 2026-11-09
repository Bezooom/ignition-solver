<a id="readme-top"></a>
<!-- best-readme-template -->
<div align="center">

<h1 align="center">ИГНИЦИЯ — солвер жёсткого тепломассопереноса (BDF2 + BiCGSTAB + инварианты 10⁻¹⁶)</h1>

<p align="center">
  Солвер жёсткого тепломассопереноса на чистом NumPy: BDF2 + BiCGSTAB с предобуславливателем, адаптивный шаг и контроль инвариантов.
  <br />
  <br />
  <a href="https://github.com/Bezooom/ignition-solver/tree/HEAD/docs"><strong>Документация »</strong></a> · <a href="https://github.com/Bezooom/ignition-solver/issues/new?labels=bug">Сообщить об ошибке</a> · <a href="https://github.com/Bezooom/ignition-solver/issues/new?labels=enhancement">Предложить улучшение</a>
</p>

<p align="center">
  <a href="https://github.com/Bezooom/ignition-solver/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Bezooom/ignition-solver/ci.yml?style=for-the-badge&label=CI&logo=githubactions&logoColor=white" alt="CI" /></a>
  <a href="https://github.com/Bezooom/ignition-solver/blob/HEAD/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License" /></a>
  <a href="https://github.com/Bezooom/ignition-solver/stargazers"><img src="https://img.shields.io/github/stars/Bezooom/ignition-solver?style=for-the-badge" alt="Stars" /></a>
  <a href="https://github.com/Bezooom/ignition-solver/commits"><img src="https://img.shields.io/github/last-commit/Bezooom/ignition-solver?style=for-the-badge" alt="Last commit" /></a>
  <a href="https://github.com/Bezooom/ignition-solver/issues"><img src="https://img.shields.io/github/issues/Bezooom/ignition-solver?style=for-the-badge" alt="Issues" /></a>
</p>

</div>

<details open>
<summary><b>📑 Содержание</b></summary>

- [📖 О проекте](#-о-проекте)
- [Результаты (полный лог — в docs/VERIFICATION.md)](#результаты-полный-лог--в-docsverificationmd)
- [Мотивация](#мотивация)
- [🧱 Технологии (Built With)](#-технологии-built-with)
- [Быстрый старт](#быстрый-старт)
- [Задача](#задача)
- [Архитектура солвера](#архитектура-солвера)
- [Как это работает (выжимка)](#как-это-работает-выжимка)
- [Производительность](#производительность)
- [Документация](#документация)
- [Структура](#структура)
- [Честные ограничения](#честные-ограничения)
- [Воспроизводимость](#воспроизводимость)
- [Происхождение](#происхождение)
- [References](#references)
- [🗺️ Дорожная карта](#-дорожная-карта)
- [🤝 Участие в разработке](#-участие-в-разработке)
- [Лицензия](#лицензия)
- [🌍 English summary](#-english-summary)
- [📬 Контакты](#-контакты)
- [🙏 Благодарности](#-благодарности)

</details>

## 📖 О проекте

> **Серия «Матчасть»** — ответ на «Матчасти нет / это программно невозможно».
> Две половины доказательства: **[ТАКТ](https://github.com/Bezooom/takt-rt)** —
> системное ядро (C++20, hard real-time), **ИГНИЦИЯ** — численное ядро (жёсткая
> МФЕ, закон сохранения на уровне round-off). Оба написаны с нуля локальной
> моделью; каждый — с матрицей приёмки и честными ограничениями.

**Написан с нуля полностью локально, без облачных API: кодинг-агент с моделью
Strata (квантованный Qwen, IQ3) на не самом дорогом железе — Xeon E5-2699 v3,
64 GB RAM, RTX 3090 24 GB.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![deps](https://img.shields.io/badge/deps-NumPy%20(basic)-informational)
![self-checks](https://img.shields.io/badge/self--checks-ALL%20PASS-green)
![series](https://img.shields.io/badge/%D0%A1%D0%B5%D1%80%D0%B8%D1%8F-%D0%9C%D0%B0%D1%82%D1%87%D0%B0%D1%81%D1%82%D1%8C-orange)
![MIT](https://img.shields.io/badge/license-MIT-yellow)

Самодостаточный скрипт на чистом NumPy (без SciPy и научных библиотек),
демонстрирующий полный математический цикл «модель = математик + архитектор
ядра»: вывод схемы и аналитического Якоби — *до* кода; решатель СЛАУ,
адаптивный шаг, контроль инвариантов — в одном файле. Ключевое число:
инвариант энергии держится на уровне **7.7·10⁻¹⁶** при требовании 10⁻⁹.

---

## Результаты (полный лог — в [docs/VERIFICATION.md](docs/VERIFICATION.md))

| Проверка | Метрика | Вердикт |
|---|---|---|
| Аналитический Якоби vs комплексно-шаговый | `1.4e-16` (машинная точность) | PASS |
| BiCGSTAB + точный блок-трёхдиагональный предобуславливатель | **1 итерация**, невязка `4.3e+03 → 2.9e-27`; решение точно до `9.8e-15`; без предобуславливателя — 40 итераций | PASS |
| Порядок BDF2 по времени (точное спектральное решение) | `2.076 / 2.037 / 2.018` | PASS |
| Инвариант E = ∫(θ+y)dx | сырой дрейф **7.7e-16** при 20 589 шагах (требование < 1e-9) | PASS |
| Критический режим (тепловой взрыв) | t₅₀(s): `0.0232 → 0.0155`; полное выгорание y→5e-22; финальная θ = предсказанию E₀−M с точностью `2.2e-16` | PASS |
| Жёсткость | явному методу нужно ≥ 5.1·10⁵ шагов; BDF: 2.06·10⁴ при rtol=1e-11 (527 шагов при rtol=1e-6) | PASS |

Все строки таблицы даёт один прогон: **~200 секунд чистого счёта на CPU**.
Это время вычислений, а не написания кода — разработка, прогоны и рецензия
шли отдельным рабочим циклом; облака при этом не было ни на одном шаге.

## Мотивация

Спор из инженерных чатов: «ИИ не тянет настоящую вычислительную матчасть».
Проверяемая версия спора — задача, где сама модель выступает и математиком,
и архитектором ядра: жёсткая система тепломассопереноса с кинетикой Аррениуса,
в которой ошибка округления на критическом режиме меняет сценарий (взрыв /
гашение). Ответ строится не словами, а матрицей приёмки: каждая строка —
проверяемый факт из логов.

## 🧱 Технологии (Built With)

<p>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/NumPy-2-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
</p>

## Быстрый старт

```bash
git clone https://github.com/Bezooom/ignition-solver.git
cd ignition-solver
python3 stiff_thermal_solver.py    # Python 3.12, numpy>=2; ~185 с
```

Ожидаемый финал: `ИТОГ: ALL PASS`, exit code 0.

## Задача

Нестационарный тепломассоперенос в 1D слое реактора с кинетикой Аррениуса:

$$
\partial_t\theta = \partial_x^2\theta + Da\,y\,e^{\theta/(1+\varepsilon\theta)},
\qquad
\partial_t y = \partial_x^2 y - Da\,y\,e^{\theta/(1+\varepsilon\theta)},
$$

адиабатические стенки (Нейман), начальное горячее пятно ширины `s` в холодном
реагенте. Параметры: `Da=20`, `ε=0.1`, сетка N=160. Жёсткость: диффузионные
собственные числа ~4/h² ≈ 10⁵, явный метод нестабилен при шаге > 2·10⁻⁵.

## Архитектура солвера

```mermaid
flowchart TD
    PDE["Уравнения θ, y (Аррениус)"] --> FD["Конечные разности\nцентральные, Нейман"]
    FD --> BDF["BDF2 переменного шага\nНьютон, аналитический Якоби"]
    BDF --> LIN["BiCGSTAB\n+ точный блок-Томаса"]
    LIN --> ERR["Локальная ошибка\ne = h²(h+r)²/(6(2h+r))·y‴"]
    ERR --> PI["PI Густафссона–Зёдерлинда\nсмена шага ≤ 2.5×"]
    PI -->|"новый шаг"| BDF
    BDF --> INV["Монитор E = ∫(θ+y)dx"]
    INV -->|"дрейф больше порога"| CORR["Консервативная коррекция"]
```

| Компонент | Реализация |
| Вывод схемы и Якоби | documentation-first: п.2–5 шапки `.py`, полные выводы в [docs/THEORY.md](docs/THEORY.md) |
| Временная интеграция | BDF1 старт + BDF2 переменного шага (a₀=(2h+r)/(h(h+r)) и пр., a₀+a₁+a₂=0), Ньютон с точным Якоби |
| Решение СЛАУ | BiCGSTAB (van der Vorst), matvec O(N) без хранения матрицы; предобуславливатель — точная блок-Томаса на 2×2-блоках |
| Управление шагом | локальная ошибка e = h²(h+r)²/(6(2h+r))·y‴ через 3-ю разделённую разность; PI-регулятор Густафссона–Зёдерлинда (1/z)^0.7/3·(1/z_prev)^(−0.4/3) |
| Контроль инвариантов | доказательство инвариантности E на полуразностном уровне; мониторинг сырого дрейфа + консервативная коррекция |
| Самопроверки | Якоби против комплексно-шагового; BiCGSTAB против эталона; спектральная верификация порядка; скан критичности; уточнение допуска |

Ключевые инварианты:

1. E = ∫(θ+y)dx сохраняется **точно** в полуразностной системе: реакции
   сокращаются алгебраически, сумма дискретного лапласиана с зеркальными
   призраками телескопируется в 0; BDF точен на константах (a₀+a₁+a₂=0).
2. Матрица Ньютона блок-трёхдиагональна (чередование переменных по ячейкам)
   → предобуславливатель точен → BiCGSTAB сходится за одну итерацию
   (доказуемо и наблюдаемо).
3. Константа ведущего члена локальной ошибки h²(h+r)²/(6(2h+r)) выведена для
   переменного шага; при r=h сворачивается в классическую (2/9)h³y‴.

## Как это работает (выжимка)

1. **Инвариант.** 7.7·10⁻¹⁶ — не «магическая проекция», а следствие: E точен
   по построению полуразностной схемы, а BDF его не портит.
2. **Якоби.** Все блоки 2×2 выписаны аналитически (шапка п.4); структура
   матрицы выбрана чередованием переменных так, чтобы предобуславливатель
   был точным.
3. **Шаг.** PI-регулятор + единственная в своём роде честная константа оценки
   для переменного шага: при r/h≥3 наивная оценка занижала ошибку до 4×.

## Производительность

527 шагов при rtol=1e-6, 2.06·10⁴ при rtol=1e-11 — против ≥5.1·10⁵ шагов
явного метода (жёсткость ~10⁵ по собственным числам). Полный верификационный
прогон — 185 с на одном ядре Xeon 2015 года; BiCGSTAB с точным
предобуславливателем — 1 итерация вместо 40.

## Документация

| Документ | Содержание |
|---|---|
| [docs/THEORY.md](docs/THEORY.md) | модель, доказательство инварианта, Якоби, BDF2, оценка ошибки, PI, BiCGSTAB, теорема о предобуславливателе |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | полный лог ALL PASS, метрики, журнал исправлений по рецензии |
| [docs/LOCAL-MODEL.md](docs/LOCAL-MODEL.md) | как это написано локальной моделью: стенд, цикл, уроки |

## Структура

```text
ignition-solver/
├── stiff_thermal_solver.py   # ~660 строк: шапка-вывод → солвер → самопроверки
└── docs/
    ├── THEORY.md
    ├── VERIFICATION.md
    └── LOCAL-MODEL.md
```

## Честные ограничения

- 1D по пространству; предобуславливатель точен именно из-за
  блок-трёхдиагональной структуры (в 2D/3D BiCGSTAB остаётся итерационным
  с тем же кодом matvec).
- y в конце выгорания даёт перелёт −5·10⁻²² (round-off); на инвариант не влияет.
- Оценка погрешности по разделённым разностям — стандартный компромисс
  LMM-кодов; на очень неровных последовательностях шага возможна недооценка
  в единицы раз (клампы PI и отклонение шага это страхуют).

## Воспроизводимость

- Python 3.12.3, NumPy 2.4.6, Linux x86_64.
- Железо разработки и прогонов: Intel Xeon E5-2699 v3 (36 нит.), 64 GB RAM,
  2×RTX 3090 24 GB (GPU для солвера не используется — чистый NumPy).
- Полный лог прогона, придающий силу таблице результатов:
  [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Происхождение

Проект выполнен в октябре 2026 г. полностью на локальном контуре; математика
прошла независимую скептическую рецензию вторым экземпляром модели (исправлены
рассогласования документации с реализацией; число отказов шага упало
2254 → 202 без потери точности). Подробно: [docs/LOCAL-MODEL.md](docs/LOCAL-MODEL.md).

## References

- Hindmarsh, A.C. *ODE integration in SUNDIALS: CVODE and CVODES*, LLNL, 2005.
- Ascher, U., Petzold, L. *Computer Methods for Ordinary Differential Equations
  and Differential-Algebraic Equations*, SIAM, 1998 — BDF, переменный шаг, zero-stability.
- Hairer, E., Wanner, G. *Solving Ordinary Differential Equations II: Stiff and
  Differential-Algebraic Problems*, Springer, 1991 — BDF, Rosenbrock, PI-константы.
- van der Vorst, H.A. *BiCGSTAB: A Fast and Smoothly Converging Variant of Bi-CG*,
  SIAM J. Sci. Stat. Comput. 13(2), 1992.
- Gustafsson, K., Söderlind, G. *On implicit explicit step number controllers*,
  BIT 29, 1989 — PI-регулятор шага.
- Frank-Kamenetskii, D.A. *Diffusion and Heat Exchange in Chemical Kinetics*.

## 🗺️ Дорожная карта

- [ ] Разбить скрипт на модуль и тесты
- [ ] Сравнение с SciPy `solve_ivp` (BDF) в отдельном отчёте

Полный список задач и идей — в [issues](https://github.com/Bezooom/ignition-solver/issues).

## 🤝 Участие в разработке

Вклад в проект приветствуется. Если есть идея, как сделать лучше, откройте issue или pull request:

1. Сделайте форк репозитория
2. Создайте ветку (`git checkout -b feature/my-idea`)
3. Сделайте коммит (`git commit -m 'feat: my idea'`)
4. Отправьте ветку (`git push origin feature/my-idea`)
5. Откройте pull request

## Лицензия

MIT — см. [LICENSE](LICENSE).

## 🌍 English summary

Self-contained stiff heat/mass-transfer solver in pure NumPy (no SciPy): BDF2 time stepping, preconditioned BiCGSTAB, PI step control and invariant checks, with an acceptance table printed by one run (`python3 stiff_thermal_solver.py`, about 3 minutes on CPU, ends with `ALL PASS`). The same run is executed in GitHub Actions. Part of the “Matchast” series together with takt-rt.

## 📬 Контакты

**Павел Наумов** (Bezooom) — [naumovai.ru](https://naumovai.ru) · [X @Bezooooom](https://x.com/Bezooooom) · naumov122@gmail.com

Проект: [https://github.com/Bezooom/ignition-solver](https://github.com/Bezooom/ignition-solver)

## 🙏 Благодарности

- [ТАКТ / takt-rt](https://github.com/Bezooom/takt-rt) — вторая половина серии «Матчасть»
- [NumPy](https://numpy.org)
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- [Shields.io](https://shields.io)

<p align="right">(<a href="#readme-top">наверх</a>)</p>
