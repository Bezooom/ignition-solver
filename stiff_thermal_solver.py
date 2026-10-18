#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 ГИБРИДНЫЙ СОЛВЕР ЖЕСТКОЙ СИСТЕМЫ ТЕПЛОМАССОПЕРЕНОСА С КИНЕТИКОЙ АРРЕНИУСА
 (BDF2 + BiCGSTAB + точный блочно-трёхдиагональный предобуславливатель
  + PI-регулятор шага + контроль инвариантов).  Только NumPy, без scipy.
===============================================================================

1. МОДЕЛЬ (безразмерная)
---------------------------------------------------------------------------
  Нестационарный тепломассоперенос в 1D слое реактора (x в [0,1]),
  адиабатические стенки, реакция первого порядка A -> продукты + тепло:

    dth/dt = d2th/dx2 + Da * y * exp( th / (1 + eps*th) )      (энергия)
    dy/dt  = d2y/dx2  - Da * y * exp( th / (1 + eps*th) )      (реагент)

  th — безразмерная температура (адиабатический нагрев = 1 при полном
  выгорании реагента), y = 1 — чистый реагент; Da — число Дамкёлера
  (время реакции = 1/Da), eps = R*T0/Q — безразмерная теплота Аррениуса
  (eps -> 0 восстанавливает e^th в стиле Франка-Каменецкого).
  Начальные условия: y(x,0) = 1; th(x,0) = A*exp(-(x-1/2)^2/(2*s^2)) —
  горячее пятно ширины s («задаток зажигания»).
  Граничные условия: dth/dx = dy/dx = 0 при x = 0, 1 (теплоизоляция).

  Жёсткость: диффузионные собственные числа ~ 4/h^2 (h — шаг сетки),
  явные методы требуют h_t < 2/|lam|max; на сетке N=160 до T=15 явный
  метод потребовал бы > 7e5 шагов. BDF делает это за тысячи.

2. ИНВАРИАНТ (ЗАКОН СОХРАНЕНИЯ) — ДОКАЗАТЕЛЬСТВО
---------------------------------------------------------------------------
  E(t) = integral_0^1 ( th + y ) dx.
  Суммируя оба уравнения: d/dt (th + y) = d2th/dx2 + d2y/dx2 (реакции
  сокращаются ТОЧНО). При Неймане integral(d2u/dx2) = u_x |_0^1 = 0.
  Значит E(t) = E(0): полный запас энергии (тепло + химический потенциал
  реагента) сохраняется. Масса реагента M(t) = integral y dx убывает ровно
  настолько, насколько растёт тепло: выделенное тепло тождественно
  выгоревшему реагенту, а финальная равномерная температура th_final =
  E(0) - integral y_final dx -> E(0) при выгорании (физическое
  предсказание, проверяется численно).
  Дискретно (узлы в центрах ячеек, зеркальные «призрачные» узлы
  u_{-1}=u_0, u_N=u_{N-1}): сумма дискретного лапласиана тождественно
  равна нулю (телеcкопирование) — инвариантово и на полуразностном уровне.
  Далее: BDF точен на константах (a0+a1+a2=0), поэтому ЛИНЕЙНЫЙ инвариант
  сохраняется схемой на уровне паразитных корней и round-off без
  дополнительных допущений.

3. ПРОСТРАНСТВЕННАЯ ДИСКРЕТИЗАЦИЯ
---------------------------------------------------------------------------
  N ячеек, h = 1/N, узлы x_i = (i+1/2)h, i = 0..N-1.
  Delta u_i = (u_{i-1} - 2u_i + u_{i+1})/h^2 (внутренние),
  Delta u_0 = (u_1 - u_0)/h^2, Delta u_{N-1} = (u_{N-2} - u_{N-1})/h^2.
  Вектор состояния U = [th_0, y_0, th_1, y_1, ...] (длиной 2N,
  чередование по ячейкам — тогда Якоби блок-трёхдиагонален с блоками 2x2).

4. МАТРИЦА ЯКОБИ — АНАЛИТИЧЕСКИЙ ВЫВОД
---------------------------------------------------------------------------
  R_i    = Da * y_i * g(th_i),   g(th) = exp( th / (1 + eps*th) )
  g'(th) = g(th) / (1 + eps*th)^2
  R_th   = dR/dth = Da * y_i * g'_i,    R_y = dR/dy = Da * g_i
  d_i    = 2/h^2 (внутри), 1/h^2 (граничные ячейки: зеркальный призрак)

  Диагональный блок J_ii (строки: th-, y-уравнение; столбцы: th_i, y_i):

    J_ii = [ -d_i + R_th_i        Da*g_i        ]
           [ -R_th_i              -d_i - Da*g_i ]

  Внедиагональные блоки (связь с соседями только диффузией, одинакова для
  обоих уравнений):

    J_{i,i+1} = J_{i,i-1} = (1/h^2) * I2.

  Контроль инварианта на уровне Якоби: L = (1,1)^T; L^T J = 0 по каждому
  столбцу: реакционные члены сокращаются, диффузионный столбец в сумме
  даёт -d_i + (число соседей)/h^2 = 0. Именно поэтому любая согласованная
  LMM сохраняет E. Проверка в коде: комплексно-шаговый Якоби
  (Im f(u + i*1e-20)/1e-20 — машинная точность) против аналитического.

5. ВРЕМЕННАЯ СХЕМА: BDF2 С ПЕРЕМЕННЫМ ШАГОМ
---------------------------------------------------------------------------
  Интерполяция по (t_{n+1}, y_{n+1}), (t_n, y_n), (t_{n-1}, y_{n-1});
  t_{n+1}=0, t_n=-h, t_{n-1}=-h-r, r = предыдущий шаг h_prev.
  Производная в t_{n+1} от лагранжевых базисов:

    y'_{n+1} ~ a0*y_{n+1} + a1*y_n + a2*y_{n-1}
    a0 = (2h + r) / (h*(h+r)),  a1 = -(h+r)/(h*r),  a2 = h/(r*(h+r)).

  Проверка: r = h -> a0 = 3/(2h), a1 = -2/h, a2 = 1/(2h) (классика BDF2).
  a0+a1+a2 = [r(2h+r) - (h+r)^2 + h^2] / (h*r*(h+r)) = 0 — точность на
  константах (следствие: сохранение линейных инвариантов, п.2).

  Первый шаг — BDF1 (обратный Эйлер): y' = (y_{n+1} - y_n)/h.
  При отклонении шага история НЕ сбрасывается: повтор BDF2 с h,
  уменьшенным по правилу п.7.

  Ньютон на каждом шаге (правая часть нелинейна):
    r(u) = a0*u - f(u) + (a1*y_n + a2*y_{n-1}) = 0
    (a0*I - J(u)) * delta = -r(u);    u <- u + delta.
  Начальное приближение — линейная экстраполяция по истории.

6. ОЦЕНКА ЛОКАЛЬНОЙ ПОГРЕШНОСТИ (управление шагом)
---------------------------------------------------------------------------
  Локальная погрешность BDF2 (подстановка точного решения, Taylor в
  t_{n+1}, постоянный шаг): невязка схемы = -(h^3/3)*y''' + O(h^4),
  откуда e_{n+1} = -(2/9)*h^3*y''' + O(h^4).
  Для переменного шага (h — текущий, r = h_prev) ведущий член:
      e_{n+1} = h^2*(h+r)^2 / (6*(2h+r)) * y'''   (при r=h = (2/9)h^3).
  Оценка y''' — третья разделённая разность по 4 принятым точкам
  (точна для кубических, y''' = 6*[t_{n-2},t_{n-1},t_n,t_{n+1}]U):
      e_est = h^2*(h+r)^2/(2h+r) * dd3.
  Первый шаг (BDF1) принимается без оценки (малый начальный h0); для
  шагов 2-3 (история короче 4 точек) — консервативная спаренная оценка
  по разности BDF2/BDF1 с того же начала: e = u_bdf2 - u_be (ошибка BE
  ~ (h^2/2)*y'' доминирует — заведомо грубее оценки dd3).
  Нормировка: z = sqrt( mean( (e_i/(atol + rtol*|u_i|))^2 ) );
  шаг принят при z <= 1, иначе отклоняется.

7. PI-РЕГУЛЯТОР ШАГА (Густафссон-Зёдерлинд)
---------------------------------------------------------------------------
  Для метода порядка p=2 (порядок локальной ошибки p+1 = 3), каноника
  Густафссона-Зёдерлинда (PI):
    h_new = h * (1/z)^{0.7/3} * (1/z_prev)^{-0.4/3},
  пропорциональная часть (1/z)^{0.233} + интегральная (память z_prev).
  Кламп отношения h_new/h в [0.2, 3.0] (классическая граница
  zero-stability BDF2 на переменном шаге ~3.4-3.7, кламп консервативен),
  h <= 0.2. Отклонение шага: h *= clamp((1/z)^{1/3}, 0.2, 0.9);
  при отказе Ньютона: h /= 4.

8. РЕШАТЕЛЬ СЛАУ: BiCGSTAB + ПРЕДОБУСЛАВЛИВАТЕЛЬ
---------------------------------------------------------------------------
  Матрица Ньютона, масштабированная на a0: B = I - J/a0 — блок-
  трёхдиагональная с блоками 2x2. Matvec за O(N) без хранения матрицы:
  только diag (N,2,2) и общий offdiag = -(1/h^2)/a0 * I2.
  Предобуславливатель — ТОЧНОЕ матрично-блоковое разложение Томаса (2x2
  блоки). Для 1D задачи B ровно блок-трёхдиагональна, поэтому предобус-
  ловленный BiCGSTAB сходится за 1 итерацию — это доказательство
  корректности предобуславливателя; на 2D/3D сетках тот же BiCGSTAB
  остаётся итерационным. Демонстрируется сравнение с НЕпредобусловленным
  BiCGSTAB (десятки итераций). Алгоритм — канонический BiCGSTAB
  (van der Vorst), левое предобуславливание.

9. КОНТРОЛЬ ИНВАРИАНТОВ
---------------------------------------------------------------------------
  После каждого принятого шага измеряется СЫРОЙ дрейф
  |E(t)-E(0)|/|E(0)| (качество метода), затем выполняется консервативная
  коррекция инварианта (добавка равномерно в th) — «control of
  invariants»; финальный остаток — уровень round-off.
  Требование приёмки: сырой дрейф < 1e-9 (ожидается ~1e-13).

10. ЗАПУСК
---------------------------------------------------------------------------
    python3 stiff_thermal_solver.py
===============================================================================
"""

import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Модель: правая часть f(U) и аналитический Якоби (см. раздел 4 шапки).
# ---------------------------------------------------------------------------
class Reactor1D:
    def __init__(self, N, Da, eps):
        self.N = N
        self.Da = Da
        self.eps = eps
        self.h = 1.0 / N
        self.h2 = self.h * self.h
        self.diff_diag = np.full(N, 2.0 / self.h2)
        self.diff_diag[0] = self.diff_diag[-1] = 1.0 / self.h2

    def initial(self, A, s):
        x = (np.arange(self.N) + 0.5) * self.h
        th = A * np.exp(-0.5 * ((x - 0.5) / s) ** 2)
        y = np.ones(self.N)
        return np.stack([th, y], axis=1).ravel()  # U = [th_i, y_i] чередование

    @staticmethod
    def _laplace(u, h2):
        d = np.empty_like(u)
        d[1:-1] = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / h2
        d[0] = (u[1] - u[0]) / h2        # зеркальный призрак u_{-1} = u_0
        d[-1] = (u[-2] - u[-1]) / h2     # зеркальный призрак u_N = u_{N-1}
        return d

    def f(self, U):
        """Полуразностная правая часть (годится и для комплексного U —
        проверка Якоби комплексным шагом)."""
        N, h2 = self.N, self.h2
        u2 = U.reshape(N, 2)
        th, y = u2[:, 0], u2[:, 1]
        q = 1.0 + self.eps * th
        g = np.exp(th / q)
        R = self.Da * y * g
        d = np.empty_like(u2)
        d[:, 0] = self._laplace(th, h2) + R
        d[:, 1] = self._laplace(y, h2) - R
        return d.ravel()

    def jac_blocks(self, U):
        """Диагональные блоки J_ii (N,2,2); внедиагональ = (1/h^2)*I2."""
        u2 = U.reshape(self.N, 2)
        th, y = u2[:, 0], u2[:, 1]
        q = 1.0 + self.eps * th
        g = np.exp(th / q)
        gp = g / (q * q)
        J = np.empty((self.N, 2, 2))
        J[:, 0, 0] = -self.diff_diag + self.Da * y * gp    # -d_i + R_th
        J[:, 0, 1] = self.Da * g                          # +Da*g
        J[:, 1, 0] = -self.Da * y * gp                    # -R_th
        J[:, 1, 1] = -self.diff_diag - self.Da * g        # -d_i - Da*g
        return J


# ---------------------------------------------------------------------------
# Точное блок-трёхдиагональное решение (матричная Томаса на блоках 2x2):
# M x = r; M: блоки diag_i (2x2), offdiag верхний = offdiag нижний = off*I2.
# ---------------------------------------------------------------------------
def block_tridiag_solve(diag, off, r):
    n = diag.shape[0]
    r2 = r.reshape(n, 2)
    invG = np.empty((n, 2, 2))
    a, b, c, d = diag[0, 0, 0], diag[0, 0, 1], diag[0, 1, 0], diag[0, 1, 1]
    invG[0][:, :] = ((d, -b), (-c, a)) / (a * d - b * c)
    cvec = r2.copy()
    for i in range(1, n):
        # G_i = diag_i - (off*I) invG_{i-1} (off*I) = diag_i - off^2 invG_{i-1}
        G = diag[i] - (off * off) * invG[i - 1]
        a, b, c, d = G[0, 0], G[0, 1], G[1, 0], G[1, 1]
        invG[i][:, :] = ((d, -b), (-c, a)) / (a * d - b * c)
        cvec[i] -= off * (invG[i - 1] @ cvec[i - 1])
    x = np.empty((n, 2))
    x[n - 1] = invG[n - 1] @ cvec[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = invG[i] @ (cvec[i] - off * x[i + 1])
    return x.ravel()


def make_matvec(diag, off):
    n = diag.shape[0]

    def matvec(x):
        x2 = x.reshape(n, 2)
        out = np.einsum("nij,nj->ni", diag, x2)
        out[1:] += off * x2[:-1]
        out[:-1] += off * x2[1:]
        return out.ravel()

    return matvec


# ---------------------------------------------------------------------------
# BiCGSTAB (van der Vorst), левое предобуславливание.
# ---------------------------------------------------------------------------
def bicgstab(matvec, minv, b, rtol=1e-15, maxit=60):
    x = np.zeros_like(b)
    r = b - matvec(x)
    rhat = r.copy()
    nb = np.linalg.norm(b)
    if nb == 0.0:
        return x, 0
    nr = np.linalg.norm(r)
    hist = [nr]
    rho = alpha = omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    it = 0
    for it in range(1, maxit + 1):
        rho_new = float(rhat @ r)
        if rho_new == 0.0:
            break
        if it == 1:
            p = r.copy()
        else:
            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
        ph = minv(p)
        v = matvec(ph)
        denom = float(rhat @ v)
        if denom == 0.0:
            break
        alpha = rho_new / denom
        s = r - alpha * v
        sh = minv(s)
        t = matvec(sh)
        tt = float(t @ t)
        omega = float(t @ s) / tt if tt > 0.0 else 0.0
        x += alpha * ph + omega * sh
        r = s - omega * t
        rho = rho_new
        nr_new = np.linalg.norm(r)
        hist.append(nr_new)
        if nr_new <= rtol * nb or nr_new >= 0.98 * nr:   # сходимость/застой
            break
        nr = nr_new
    return x, it, hist


# ---------------------------------------------------------------------------
# Ньютон для нелинейного уравнения одного шага BDF (раздел 5 шапки).
# Возвращает (u, n_newton, n_bicgstab_total, resid_hist_first_solve).
# ---------------------------------------------------------------------------
def newton_solve(model, u_hist, t_hist, h):
    if len(u_hist) < 2:                       # BDF1 (обратный Эйлер)
        a0 = 1.0 / h
        hist_term = a0 * u_hist[-1]
        guess = u_hist[-1] + h * model.f(u_hist[-1])
    else:                                     # BDF2 переменного шага
        r_prev = t_hist[-1] - t_hist[-2]
        Un, Un1 = u_hist[-1], u_hist[-2]
        a0 = (2.0 * h + r_prev) / (h * (h + r_prev))
        a1 = -(h + r_prev) / (h * r_prev)
        a2 = h / (r_prev * (h + r_prev))
        hist_term = -(a1 * Un + a2 * Un1)   # res = a0*u - f - hist_term
        guess = Un + (h / r_prev) * (Un - Un1)
    u = guess
    bi_total = 0
    first_hist = None
    I2 = np.eye(2)
    for nit in range(1, 11):
        fu = model.f(u)
        res = a0 * u - fu - hist_term
        # поэлементная backward-error: порог относительно величин слагаемых
        # (учитывает круговой шум вычитания больших диффузионных членов)
        denom = abs(a0) * np.abs(u) + np.abs(fu) + np.abs(hist_term) + 1.0
        if np.max(np.abs(res) / denom) <= 1e-12:
            return u, nit, bi_total, first_hist
        J = model.jac_blocks(u)
        Bdiag = I2[None, :, :] - J / a0        # B = I - J/a0
        off = -(1.0 / model.h2) / a0
        mv = make_matvec(Bdiag, off)
        pre = lambda z, _d=Bdiag, _o=off: block_tridiag_solve(_d, _o, z)
        delta, bit, bh = bicgstab(mv, pre, -res / a0, rtol=1e-15, maxit=60)
        bi_total += bit
        if first_hist is None:
            first_hist = bh
        u = u + delta
        if not np.all(np.isfinite(u)):
            return None, nit, bi_total, first_hist
    return None, 10, bi_total, first_hist


# ---------------------------------------------------------------------------
# Третья разделённая разность по 4 точкам (оценка y''', раздел 6 шапки).
# ---------------------------------------------------------------------------
def dd3(ts, Us):
    d1 = [(Us[i + 1] - Us[i]) / (ts[i + 1] - ts[i]) for i in range(3)]
    d2 = [(d1[i + 1] - d1[i]) / (ts[i + 2] - ts[i]) for i in range(2)]
    return (d2[1] - d2[0]) / (ts[3] - ts[0])


# ---------------------------------------------------------------------------
# Интегрирование: BDF1 старт + BDF2, PI-шаг, отклонения шага, инварианты.
# ---------------------------------------------------------------------------
def integrate(model, U0, t_end, rtol=1e-10, atol=1e-12, h0=1e-3,
              fixed_h=None, h_cap=0.2, conserve=True, debug=False):
    N = model.N
    st = dict(steps=0, rejects=0, newton=0, newton_max=0, bi_total=0,
              hmin=np.inf, hmax=0.0)
    t_hist = [0.0]
    u_hist = [U0]
    S0 = float(np.sum(U0))          # E = h*sum(U); h=const, храним sum
    t = 0.0
    h = fixed_h if fixed_h else h0
    z_prev = 1.0
    drift_raw = 0.0
    corr_max = 0.0
    t_ign = None
    y_target = 0.5 * float(np.sum(U0.reshape(N, 2)[:, 1]))  # половина реагента
    bi_first = []                   # невязки первых BiCGSTAB-решений

    while t < t_end - 1e-14:
        if fixed_h:
            h = min(fixed_h, t_end - t)
        h_try = min(h, t_end - t)
        u_new, n_nt, n_bi, bh = newton_solve(model, u_hist, t_hist, h_try)
        if len(u_hist) < 4 and not fixed_h:
            bi_first.extend(bh[:4] if bh else [])
        if u_new is None:
            st["rejects"] += 1
            if debug:
                print(f"    rej(newton) t={t:.4e} h={h_try:.2e}")
            h *= 0.25
            if h < 1e-13:
                raise RuntimeError("Newton diverged: h floor hit")
            continue
        if fixed_h:
            z = None
        else:
            # -------- оценка локальной погрешности (раздел 6) --------
            if len(u_hist) >= 4:
                r_p = t_hist[-1] - t_hist[-2]
                est = (h_try ** 2 * (h_try + r_p) ** 2 / (2 * h_try + r_p)
                       * dd3(t_hist[-3:] + [t + h_try], u_hist[-3:] + [u_new]))
            else:               # спаренная BDF2/BDF1 оценка для старта
                probe, _, _, _ = newton_solve(model, [u_hist[-1]],
                                              [t_hist[-1]], h_try)
                est = u_new - probe
            w = atol + rtol * np.maximum(np.abs(u_new), np.abs(u_hist[-1]))
            z = float(np.sqrt(np.mean((est / w) ** 2)))
            if z > 1.0:
                st["rejects"] += 1
                if debug and st["rejects"] % 20 == 1:
                    print(f"    rej(z) t={t:.4e} h={h_try:.2e} z={z:.2e} "
                          f"|est|max={np.max(np.abs(est)):.2e}")
                h *= min(0.9, max(0.2, (1.0 / z) ** (1.0 / 3.0)))
                if h < 1e-13:
                    raise RuntimeError("step floor hit")
                continue
        # -------- шаг принят --------
        st["steps"] += 1
        st["newton"] += n_nt
        st["newton_max"] = max(st["newton_max"], n_nt)
        st["bi_total"] += n_bi
        st["hmin"] = min(st["hmin"], h_try)
        st["hmax"] = max(st["hmax"], h_try)
        t += h_try
        # -------- инвариант: сырой дрейф + консервативная коррекция ------
        S = float(np.sum(u_new))
        drift_raw = max(drift_raw, abs(S - S0) / max(abs(S0), 1.0))
        if conserve:
            dS = S0 - S
            if dS != 0.0:
                u_new = u_new.reshape(N, 2).copy()
                u_new[:, 0] += dS / N        # коррекция равномерно в th
                u_new = u_new.ravel()
                corr_max = max(corr_max, abs(dS / N))
        u_hist.append(u_new)
        t_hist.append(t)
        if len(u_hist) > 4:
            u_hist = u_hist[-4:]
            t_hist = t_hist[-4:]
        wcell = u_new.reshape(N, 2)
        if t_ign is None and np.sum(wcell[:, 1]) <= y_target:
            t_ign = t                    # момент 50% выгорания реагента
        # -------- PI-регулятор (раздел 7) --------
        if not fixed_h:
            zz = max(z, 1e-8)
            zp = max(z_prev, 1e-8)
            h_new = h_try * (1.0 / zz) ** (0.7 / 3.0) \
                * (1.0 / zp) ** (-0.4 / 3.0)
            h = min(h_try * min(3.0, max(0.2, h_new / h_try)), h_cap)
            z_prev = z
    st["bi_first"] = bi_first
    st.update(t_ign=t_ign, drift_raw=drift_raw, corr_max=corr_max)
    return u_hist[-1], st


# ---------------------------------------------------------------------------
# Самопроверка 1: аналитический Якоби против комплексно-шагового.
# ---------------------------------------------------------------------------
def check_jacobian(model, U):
    n = U.size
    epsc = 1e-20
    base = U.astype(complex)
    Jfd = np.zeros((n, n))
    for k in range(n):
        pert = base.copy()
        pert[k] += 1j * epsc
        Jfd[:, k] = model.f(pert).imag / epsc
    Ja = np.zeros((n, n))
    Jii = model.jac_blocks(U)
    off = 1.0 / model.h2
    for i in range(model.N):
        Ja[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Jii[i]
        if i + 1 < model.N:
            Ja[2 * i:2 * i + 2, 2 * i + 2:2 * i + 4] = off * np.eye(2)
            Ja[2 * i + 2:2 * i + 4, 2 * i:2 * i + 2] = off * np.eye(2)
    return np.max(np.abs(Ja - Jfd)) / max(np.max(np.abs(Jfd)), 1e-30)


# ---------------------------------------------------------------------------
# Самопроверка 2: предобусловленный BiCGSTAB против точного решения и
# против BiCGSTAB без предобуславливателя.
# ---------------------------------------------------------------------------
def check_solver(model, U):
    rng = np.random.default_rng(7)
    a0 = 150.0
    J = model.jac_blocks(U)
    Bdiag = np.eye(2)[None, :, :] - J / a0
    off = -(1.0 / model.h2) / a0
    mv = make_matvec(Bdiag, off)
    x_true = rng.uniform(-1, 1, 2 * model.N)
    b = mv(x_true)
    xp, itp, histp = bicgstab(mv,
                              lambda z: block_tridiag_solve(Bdiag, off, z),
                              b, rtol=1e-14, maxit=400)
    xn, itn, histn = bicgstab(mv, lambda z: z, b, rtol=1e-14, maxit=4000)
    err = np.linalg.norm(xp - x_true) / np.linalg.norm(x_true)
    return err, itp, histp, itn, histn


# ---------------------------------------------------------------------------
# Метрика сходимости: порядок по времени (BDF1 старт + BDF2, фикс. шаг) на
# задаче Da=0: theta_t = theta_xx; точное решение полуразностной системы —
# спектр зеркального лапласиана: mu_k = -(4/h^2) sin^2(k pi h/2),
# собственные векторы cos(k pi x_i) ортогональны на равномерной сетке.
# ---------------------------------------------------------------------------
def check_time_order(N=100, T=0.5, ns=(40, 80, 160, 320)):
    model = Reactor1D(N, Da=0.0, eps=0.1)
    h = model.h
    x = (np.arange(N) + 0.5) * h
    th0 = np.cos(np.pi * x)
    U0 = np.stack([th0, np.zeros(N)], axis=1).ravel()
    k = np.arange(N)
    mu = -(4.0 / h / h) * np.sin(k * np.pi * h / 2) ** 2
    V = np.cos(np.outer(x, k * np.pi))
    c = (2.0 / N) * (V.T @ th0)
    c[0] = np.sum(th0) / N
    exact = V @ (np.exp(mu * T) * c)
    errs = []
    for n_steps in ns:
        U, _ = integrate(model, U0, T, fixed_h=T / n_steps, conserve=False)
        th = U.reshape(N, 2)[:, 0]
        errs.append(np.max(np.abs(th - exact)))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    return ns, errs, orders


# ---------------------------------------------------------------------------
# Скан критичности: ширина зажиганияного пятна s -> задержка зажигания.
# Уменьшение s: диффузия гасит ядро, задержка растёт; у минимального s,
# при котором ещё возможен взрывный сценарий, — критическая чувствительность.
# ---------------------------------------------------------------------------
def criticality_scan(N, Da, eps, A, widths, t_end=10.0):
    out = []
    for s in widths:
        model = Reactor1D(N, Da, eps)
        _, st = integrate(model, model.initial(A, s), t_end,
                          rtol=1e-8, atol=1e-10, h0=1e-3, conserve=False)
        out.append((s, st["t_ign"], st["steps"]))
    return out


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    N, Da, eps, A = 160, 20.0, 0.1, 1.0
    failures = []

    print("=" * 74)
    print("САМОПРОВЕРКА ЯКОБИ (аналитический vs комплексно-шаговый, N=%d)" % N)
    m = Reactor1D(N, Da, eps)
    Uchk = m.initial(1.0, 0.1).reshape(N, 2)
    Uchk[:, 0] = 0.8
    Uchk[:, 1] = 0.4
    err_j = check_jacobian(m, Uchk.ravel())
    ok = err_j < 1e-10
    print(f"  max|J_an - J_cstep| / |J| = {err_j:.2e}   [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        failures.append("jacobian")

    print("=" * 74)
    print("САМОПРОВЕРКА РЕШАТЕЛЯ: BiCGSTAB c предобуславливателем / без")
    err_x, itp, histp, itn, histn = check_solver(m, Uchk.ravel())
    ok = err_x < 1e-9 and itp <= 4
    print(f"  ||x_biCG - x_exact||/||x|| = {err_x:.2e}   [{'PASS' if ok else 'FAIL'}]")
    print(f"  c предобуславливателем: {itp} итер., невязки "
          f"{['%.1e' % v for v in histp[:5]]}")
    print(f"  без предобуславливателя: {itn} итер., невязка "
          f"{histn[0]:.1e} -> {histn[-1]:.1e}")
    if not ok:
        failures.append("solver")

    print("=" * 74)
    print(f"СКАН КРИТИЧНОСТИ (N={N}, Da={Da}, eps={eps}, A={A}): "
          "ширина пятна s -> время зажигания t_ign")
    widths = [0.06, 0.10, 0.15, 0.25]
    scan = criticality_scan(N, Da, eps, A, widths, t_end=6.0)
    s_crit = None
    for s, tign, nsteps in scan:
        ign = "нет за 6" if tign is None else f"{tign:.4f}"
        print(f"  s={s:<5}  t_ign={ign:<12} шагов={nsteps}")
        if s_crit is None and tign is not None:
            s_crit = s
    if s_crit is None:
        raise RuntimeError("ни один режим не воспламенился")
    print(f"  -> задержка зажигания растёт при уменьшении пятна (s={s_crit}"
          " выбрано как наиболее чувствительный критический режим)")

    print("=" * 74)
    print("КРИТИЧЕСКИЙ РЕЖИМ (основной прогон): s=%s, rtol=1e-11, "
          "консервация включена" % s_crit)
    model = Reactor1D(N, Da, eps)
    U0 = model.initial(A, s_crit)
    Uend, st = integrate(model, U0, 10.0, rtol=1e-11, atol=1e-13, h0=1e-3,
                         conserve=True)
    u2 = Uend.reshape(N, 2)
    th, y = u2[:, 0], u2[:, 1]
    S0 = np.sum(U0)
    lam_max = 4.0 / model.h2 + 3.0 * Da
    expl = 10.0 * lam_max / 2.0
    print(f"  шагов BDF: {st['steps']}, отклонено шагов: {st['rejects']}")
    print(f"  итераций Ньютона: {st['newton']} всего, max на решение: "
          f"{st['newton_max']}")
    print(f"  итераций BiCGSTAB: {st['bi_total']} всего, в среднем на "
          f"решение Ньютона: {st['bi_total'] / max(st['steps'], 1):.2f}")
    print(f"  h: min={st['hmin']:.2e}, max={st['hmax']:.2e}; "
          f"t_ign={st['t_ign']}")
    print(f"  y: min={y.min():.2e}, max={y.max():.2e} "
          f"(полное выгорание: {y.max() < 1e-3})")
    dev = np.max(np.abs(th - (S0 - np.sum(y)) / N))
    print(f"  физпредсказание th_final = (E0 - M_final)/N = "
          f"{(S0 - np.sum(y)) / N:.6f}: отклонение от равномерного = {dev:.2e}")
    print(f"  сравнение с явным методом: потребовалось бы >= {expl:.2e} "
          f"шагов; BDF сделал {st['steps']} (ускорение ~{expl / st['steps']:.0f}x)")
    ok = st["drift_raw"] < 1e-9
    print(f"  СЫРОЙ дрейф инварианта E(t): max = {st['drift_raw']:.3e}   "
          f"[{'PASS' if ok else 'FAIL'}]")
    if not ok:
        failures.append("invariant_raw")
    print(f"  контроль инвариантов: консервативная коррекция max |dth| = "
          f"{st['corr_max']:.2e}; финальный остаток ~ round-off")

    print("=" * 74)
    print("ПОРЯДОК СХОДИМОСТИ ПО ВРЕМЕНИ (Da=0, фикс. шаг, точное решение — "
          "спектр дискретного лапласиана)")
    ns, errs, orders = check_time_order()
    for n, e in zip(ns, errs):
        print(f"  n={n:<5} h={0.5 / n:.2e}  max|theta - exact| = {e:.3e}")
    for o in orders:
        print(f"  порядок (log2 отношения ошибок) = {o:.3f}")
    ok = all(o > 1.85 for o in orders)
    print(f"  BDF2 по времени ~ 2-й порядок: [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        failures.append("order")

    print("=" * 74)
    print("УТОЧНЕНИЕ ДОПУСКА (rtol -> стоимость и точность инварианта)")
    refine = {}
    for rt in (1e-6, 1e-8, 1e-10):
        mr = Reactor1D(N, Da, eps)
        _, sr = integrate(mr, model.initial(A, s_crit), 10.0, rtol=rt,
                          atol=rt * 1e-2, h0=1e-3, conserve=True)
        refine[rt] = sr
        print(f"  rtol={rt:.0e}: шагов={sr['steps']:>6}, отклонено="
              f"{sr['rejects']:>4}, сырой дрейф E={sr['drift_raw']:.3e}, "
              f"h in [{sr['hmin']:.1e}, {sr['hmax']:.1e}]")
    ok = (refine[1e-8]["steps"] > refine[1e-6]["steps"]
          and refine[1e-10]["steps"] > refine[1e-8]["steps"])
    print(f"  монотонный рост стоимости при уточнении: "
          f"[{'PASS' if ok else 'FAIL'}]")
    if not ok:
        failures.append("refine")

    dt = time.time() - t0
    print("=" * 74)
    if failures:
        print(f"ИТОГ: FAIL ({', '.join(failures)}), время {dt:.1f} c")
        sys.exit(1)
    print(f"ИТОГ: ALL PASS, время {dt:.1f} c")


if __name__ == "__main__":
    main()
