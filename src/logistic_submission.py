from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import Settings
from logistic_features import LogisticFeatureEngineer, NumericPairwiseInteractions
from preprocessing import load_data


GROUPS = (
    "aw", "am", "ad", "bw", "bm", "bd", "l c", "lm c", "r room",
    "rate room", "lead c", "lead room", "aw c", "am c", "aw room", "ad c",
    "bw c", "bd c", "aw room c", "r c room", "r aw room", "r am room",
    "l room", "t aw c", "req c lm", "req room lm", "meal room c", "room c l",
    "room c lm r", "ad lead", "ad r room", "bw aw c", "rate lead", "rate lead room",
)
CONTEXT_GROUPS = (
    "ad room", "ad r", "aw r", "aw l c", "am l c", "ad c room", "aw c l room",
    "bd room", "bd c room", "bd c l", "bw c l", "bm c l", "bw room c",
    "ad bd", "ad bd c", "ad bd room", "ad bd c room",
    "ad lead room", "ad lead c", "ad lead rate", "ad rate c", "ad rate room",
    "rate room c", "rate room meal", "rate ad", "rate aw c", "rate am c",
    "rate bw c", "rate rshift c t", "rshift aw room", "rshift bw room",
    "rshift c room t", "rshift c l", "rshift c lm", "r c req", "r room req",
    "req l c", "req l room", "req l c room", "req r c room", "req c room t",
    "t r c room", "t l c room", "stay r c room", "stay l c room",
    "lshift room c", "lshift c req", "lshift ad c", "lshift r c room",
    "lshift aw c room", "guests c room", "guests r c", "guests l c",
    "adw wd c", "adw wd l c", "bdw wd r c", "adw r c room", "bdw l c room",
    "repeated c", "bookings c", "cancel c", "parking c room",
)
WINDOWS = (
    ("lead", "ad c", (3, 14)), ("lead", "ad room", (3, 14)),
    ("lead", "aw c room", (3, 14)), ("lead", "am c room req", (3, 14)),
    ("lead", "c room req", (3, 14)), ("lead", "bd c", (3, 14)),
    ("lead", "rate c room", (3, 14)), ("lead", "rshift c room", (3, 14)),
    ("price", "ad c room", (25000, 100000)), ("price", "aw c room", (25000, 100000)),
    ("price", "aw room", (25000, 100000)), ("price", "am room", (25000, 100000)),
    ("price", "c room l", (25000, 100000)), ("price", "c room lshift req", (25000, 100000)),
    ("price", "bw c room", (25000, 100000)), ("price", "c room t", (25000, 100000)),
    ("arrival", "c room rshift", (3, 14)), ("arrival", "c room l", (3, 14)),
    ("arrival", "c room rshift l", (3, 14)), ("arrival", "rate room", (3, 14)),
    ("booking", "c room rshift", (3, 14)), ("booking", "c room l", (3, 14)),
    ("booking", "ad c", (3, 14)), ("booking", "ad room rshift", (3, 14)),
)
DISTANCE_COLUMNS = (
    "days_before_arrival", "nightly_rate_idr", "special_requests", "arrival_year",
    "arrival_month_sin", "arrival_month_cos", "num_adults", "num_children",
    "total_nights", "weekend_nights", "log_days_before_arrival",
)
THRESHOLD = 0.42
OUTPUT_NAME = "lr_local_features.csv"


class LogisticSubmissionPreprocessor:
    def __init__(
        self, zero_as_missing=(), num_strategy="median", drop=(), cap=(),
        cap_factor=1.5, ordinal=None, scaler="standard",
    ):
        self.zero_as_missing = list(zero_as_missing)
        self.num_strategy = num_strategy
        self.drop = list(drop)
        self.cap = list(cap)
        self.cap_factor = cap_factor
        self.ordinal = ordinal or {}
        self.scaler = scaler

    def fit(self, X):
        X = self.zero_to_nan(X)
        numeric = X.select_dtypes("number").columns
        categorical = X.columns.difference(numeric)
        self.fill = {
            **getattr(X[numeric], self.num_strategy)().to_dict(),
            **X[categorical].mode().iloc[0].to_dict(),
        }
        X = self.engineer(X.fillna(self.fill)).drop(columns=self.drop)

        q1, q3 = X[self.cap].quantile(0.25), X[self.cap].quantile(0.75)
        self.bounds = (
            q1 - self.cap_factor * (q3 - q1),
            q3 + self.cap_factor * (q3 - q1),
        )
        X = self.clip(X)

        categorical = X.select_dtypes(exclude="number").columns
        self.onehot = {
            column: sorted(X[column].unique())
            for column in categorical if column not in self.ordinal
        }
        X = self.encode(X)
        self.columns = X.columns

        if self.scaler:
            center, spread = {
                "standard": lambda: (X.mean(), X.std()),
                "minmax": lambda: (X.min(), X.max() - X.min()),
                "robust": lambda: (
                    X.median(), X.quantile(0.75) - X.quantile(0.25),
                ),
            }[self.scaler]()
            self.center, self.spread = center, spread.replace(0, 1)
        return self

    def transform(self, X):
        X = self.zero_to_nan(X).fillna(self.fill)
        X = self.clip(self.engineer(X).drop(columns=self.drop))
        return self.scale(
            self.encode(X).reindex(columns=self.columns, fill_value=0)
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @staticmethod
    def engineer(X):
        X = X.copy()
        X["total_nights"] = X["weekend_nights"] + X["weekday_nights"]
        X["total_guests"] = X["num_adults"] + X["num_children"]
        return X

    def zero_to_nan(self, X):
        X = X.copy()
        X[self.zero_as_missing] = X[self.zero_as_missing].replace(0, np.nan)
        return X

    def clip(self, X):
        if self.cap:
            X[self.cap] = X[self.cap].clip(*self.bounds, axis=1)
        return X

    def encode(self, X):
        X = X.copy()
        for column, order in self.ordinal.items():
            if column in X:
                X[column] = X[column].map(
                    {value: index for index, value in enumerate(order)}
                ).fillna(-1)
        for column, categories in self.onehot.items():
            X[column] = pd.Categorical(X[column], categories=categories)
        return pd.get_dummies(X, columns=list(self.onehot), dtype=float)

    def scale(self, X):
        return (X - self.center) / self.spread if self.scaler else X


def reservation_key_frame(frame, context=False):
    arrival = pd.to_datetime(
        dict(year=frame.arrival_year, month=frame.arrival_month, day=frame.arrival_day),
        errors="coerce",
    )
    booked = arrival - pd.to_timedelta(frame.days_before_arrival, unit="D")
    keys = pd.DataFrame({
        "aw": arrival.dt.strftime("%Y-%W"), "am": arrival.dt.strftime("%Y-%m"),
        "ad": arrival.dt.strftime("%Y-%m-%d"), "bw": booked.dt.strftime("%Y-%W"),
        "bm": booked.dt.strftime("%Y-%m"), "bd": booked.dt.strftime("%Y-%m-%d"),
        "l": frame.days_before_arrival // 7, "lm": frame.days_before_arrival // 30,
        "r": (frame.nightly_rate_idr / 100000).round(),
        "t": frame.weekend_nights + frame.weekday_nights,
        "c": frame.booking_channel, "room": frame.room_category,
        "req": frame.special_requests, "rate": frame.nightly_rate_idr,
        "lead": frame.days_before_arrival, "meal": frame.meal_package,
    }, index=frame.index)
    if context:
        keys = keys.assign(
            wd=arrival.dt.day_of_week,
            adw=(arrival - pd.Timedelta(days=3)).dt.strftime("%Y-%W"),
            bdw=(booked - pd.Timedelta(days=3)).dt.strftime("%Y-%W"),
            lshift=(frame.days_before_arrival + 3) // 7,
            rshift=(frame.nightly_rate_idr + 25000) // 50000,
            guests=frame.num_adults * 10 + frame.num_children,
            stay=frame.weekend_nights * 100 + frame.weekday_nights,
            repeated=frame.is_returning_guest,
            bookings=frame.prior_completed_bookings,
            cancel=frame.prior_cancellations,
            parking=frame.needs_parking,
        )
    return keys.astype("string").fillna("missing"), arrival, booked


def group_keys(frame, context=False):
    keys, _, _ = reservation_key_frame(frame, context)
    groups = GROUPS + CONTEXT_GROUPS if context else GROUPS
    return [pd.util.hash_pandas_object(keys[group.split()], index=False) for group in groups]


def encode_group(train_key, labels, eval_key, smoothing=5.0):
    stats = pd.DataFrame({"key": train_key.to_numpy(), "label": labels}).groupby("key")
    stats = stats.label.agg(["sum", "count"])
    count = eval_key.map(stats["count"]).fillna(0).to_numpy()
    positive = eval_key.map(stats["sum"]).fillna(0).to_numpy()
    return np.column_stack(((positive + smoothing * labels.mean()) / (count + smoothing), np.log1p(count)))


def group_features(train, labels, evaluation, context=False, smoothing=5.0):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=Settings.SEED).split(train, labels))
    train_features, eval_features = [], []
    for train_key, eval_key in zip(group_keys(train, context), group_keys(evaluation, context)):
        cross_fitted = np.empty((len(train), 2))
        for fit_rows, predict_rows in folds:
            cross_fitted[predict_rows] = encode_group(
                train_key.iloc[fit_rows], labels[fit_rows], train_key.iloc[predict_rows], smoothing=smoothing
            )
        train_features.append(cross_fitted)
        eval_features.append(encode_group(train_key, labels, eval_key, smoothing=smoothing))
    return np.column_stack(train_features), np.column_stack(eval_features)


def window_inputs(frame):
    keys, arrival, booked = reservation_key_frame(frame, context=True)
    date_origin = pd.Timestamp("2023-01-01")
    values = {
        "lead": frame.days_before_arrival.to_numpy(dtype=float),
        "price": frame.nightly_rate_idr.to_numpy(dtype=float),
        "arrival": (arrival - date_origin).dt.days.to_numpy(dtype=float),
        "booking": (booked - date_origin).dt.days.to_numpy(dtype=float),
    }
    return keys, values


def window_stats(keys, values, labels, eval_keys, eval_values, radii):
    dtype = np.dtype([("group", np.uint64), ("value", np.float64)])
    valid = np.isfinite(values)
    eval_valid = np.isfinite(eval_values)
    training = np.rec.fromarrays((keys[valid], values[valid]), dtype=dtype)
    order = np.argsort(training)
    training = training[order]
    cumulative = np.r_[0, np.cumsum(labels[valid][order])]
    prior = labels.mean()
    columns = []
    for radius in radii:
        lower = np.rec.fromarrays((eval_keys, eval_values - radius), dtype=dtype)
        upper = np.rec.fromarrays((eval_keys, eval_values + radius), dtype=dtype)
        left = np.searchsorted(training, lower)
        right = np.searchsorted(training, upper, side="right")
        count = np.where(eval_valid, right - left, 0)
        cancellations = np.where(eval_valid, cumulative[right] - cumulative[left], 0)
        columns.extend(((cancellations + 3 * prior) / (count + 3), np.log1p(count)))
    return np.column_stack(columns)


def window_features(train, labels, evaluation):
    train_keys, train_values = window_inputs(train)
    eval_keys, eval_values = window_inputs(evaluation)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=Settings.SEED).split(train, labels))
    train_blocks, eval_blocks = [], []
    for numeric, group, radii in WINDOWS:
        cols = group.split()
        ka = pd.util.hash_pandas_object(train_keys[cols], index=False).to_numpy()
        kb = pd.util.hash_pandas_object(eval_keys[cols], index=False).to_numpy()
        va, vb = train_values[numeric], eval_values[numeric]
        cross_fitted = np.empty((len(train), 2 * len(radii)))
        for fit_rows, predict_rows in folds:
            cross_fitted[predict_rows] = window_stats(
                ka[fit_rows], va[fit_rows], labels[fit_rows], ka[predict_rows], va[predict_rows], radii,
            )
        train_blocks.append(cross_fitted)
        eval_blocks.append(window_stats(ka, va, labels, kb, vb, radii))
    return np.column_stack(train_blocks), np.column_stack(eval_blocks)


def radial_features(values, landmarks, gamma):
    distances = values @ landmarks.T
    distances *= -2
    distances += np.square(values).sum(axis=1, keepdims=True)
    distances += np.square(landmarks).sum(axis=1)
    np.maximum(distances, 0, out=distances)
    distances *= -gamma
    np.exp(distances, out=distances)
    return 4 * distances


def prepare_numeric_features(train, evaluation):
    engineer = LogisticFeatureEngineer().fit(train)
    train, evaluation = engineer.transform(train), engineer.transform(evaluation)
    prep = LogisticSubmissionPreprocessor().fit(train)
    prepared_train, prepared_eval = prep.transform(train), prep.transform(evaluation)
    pairs = NumericPairwiseInteractions().fit(prepared_train)
    return prepared_train, prepared_eval, pairs


def booking_distance(frame, fitted):
    arrival = pd.to_datetime(
        dict(year=frame.arrival_year, month=frame.arrival_month, day=frame.arrival_day),
        errors="coerce",
    )
    arrival_days = (arrival - pd.Timestamp("2023-01-01")).dt.days
    lead = frame.days_before_arrival
    price = frame.nightly_rate_idr
    raw = pd.DataFrame({
        "arrival": arrival_days / 95,
        "booking": (arrival_days - lead) / 110,
        "lead": np.log1p(lead.clip(lower=0)) / 1.3,
        "price": price / 430000,
        "stay": (frame.weekend_nights + frame.weekday_nights) / 3,
        "weekend": frame.weekend_nights / 2,
        "requests": frame.special_requests / 1.4,
        "adults": frame.num_adults / 2,
        "children": frame.num_children / 2,
        "weekday_sin": 0.35 * np.sin(2 * np.pi * arrival.dt.day_of_week / 7),
        "weekday_cos": 0.35 * np.cos(2 * np.pi * arrival.dt.day_of_week / 7),
    }, index=frame.index)
    category = pd.get_dummies(frame[["booking_channel", "room_category", "meal_package"]], dtype=float)
    if fitted is None:
        fitted = raw.median().fillna(0), category.columns
    fill, columns = fitted
    values = np.column_stack((raw.fillna(fill), category.reindex(columns=columns, fill_value=0)))
    return np.clip(values, -12, 12), fitted


def density_coordinates(frame):
    keys, values = window_inputs(frame)
    lead = values["lead"]
    price = values["price"]
    request = frame.special_requests.to_numpy(dtype=float)
    nights = keys["t"].replace("missing", np.nan).to_numpy(dtype=float)
    fields = {
        "a": values["arrival"] / 12,
        "b": values["booking"] / 15,
        "l": lead / 22,
        "logl": np.log1p(np.maximum(lead, 0)) / 0.55,
        "p": price / 95000,
        "q": request * 0.8,
        "n": nights * 0.3,
    }
    return fields, {
        "c": keys["c"].to_numpy(),
        "cr": (keys["c"] + "|" + keys["room"]).to_numpy(),
    }


def neighborhood_counts(x, y, groups, evaluation, eval_groups):
    columns = np.zeros((len(evaluation), 6))
    for group in np.unique(eval_groups):
        fit_rows = (groups == group) & np.isfinite(x).all(axis=1)
        eval_rows = np.flatnonzero((eval_groups == group) & np.isfinite(evaluation).all(axis=1))
        if not fit_rows.any() or not len(eval_rows):
            continue
        local = x[fit_rows]
        positive = cKDTree(local[y[fit_rows] == 1])
        negative = cKDTree(local[y[fit_rows] == 0])
        for j, radius in enumerate((0.65, 1.15, 1.9)):
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html
            a = positive.query_ball_point(evaluation[eval_rows], radius, return_length=True)
            b = negative.query_ball_point(evaluation[eval_rows], radius, return_length=True)
            columns[eval_rows, 2 * j] = np.log1p(a)
            columns[eval_rows, 2 * j + 1] = np.log1p(b)
    return columns


def density_features(train, labels, evaluation):
    train_values, train_groups = density_coordinates(train)
    eval_values, eval_groups = density_coordinates(evaluation)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=Settings.SEED).split(train, labels))
    train_blocks, eval_blocks = [], []
    for group, names in (
        ("cr", "a p l"), ("cr", "a p q logl"), ("cr", "b p q logl"),
        ("cr", "a b p n"), ("c", "a p q l"), ("c", "b p q logl"),
        ("c", "a b q l"), ("cr", "p q logl n"), ("c", "a b q n"),
    ):
        x = np.column_stack([train_values[name] for name in names.split()])
        z = np.column_stack([eval_values[name] for name in names.split()])
        ka, kb = train_groups[group], eval_groups[group]
        oof = np.empty((len(x), 6))
        for train_rows, val_rows in folds:
            oof[val_rows] = neighborhood_counts(x[train_rows], labels[train_rows], ka[train_rows], x[val_rows], ka[val_rows])
        train_blocks.append(oof)
        eval_blocks.append(neighborhood_counts(x, labels, ka, z, kb))
    return np.column_stack(train_blocks), np.column_stack(eval_blocks)


def design_matrices(
    train, labels, evaluation, n_landmarks=1024, context=False, calendar_landmarks=512, windows=False, density=False,
    target_smoothing=5.0,
):
    group_train, group_eval = group_features(
        train, labels, evaluation, context=context, smoothing=target_smoothing,
    )
    u, v, pairs = prepare_numeric_features(train, evaluation)
    cols = list(DISTANCE_COLUMNS) + [
        column for column in u if column.startswith(("booking_channel_", "room_category_", "meal_package_"))
    ]
    train_distance = np.clip(u[cols].to_numpy(), -5, 5)
    eval_distance = np.clip(v[cols].to_numpy(), -5, 5)
    rng = np.random.default_rng(Settings.SEED)
    landmarks = train_distance[rng.choice(len(train_distance), min(n_landmarks, len(train_distance)), replace=False)]
    blocks_train = [pairs.transform(u).to_numpy(), group_train, radial_features(train_distance, landmarks, 0.24)]
    blocks_eval = [pairs.transform(v).to_numpy(), group_eval, radial_features(eval_distance, landmarks, 0.24)]
    if windows:
        window_train, window_eval = window_features(train, labels, evaluation)
        blocks_train.append(window_train)
        blocks_eval.append(window_eval)
    if density:
        density_train, density_eval = density_features(train, labels, evaluation)
        blocks_train.append(density_train)
        blocks_eval.append(density_eval)
    if context and calendar_landmarks:
        calendar_train, fitted = booking_distance(train, None)
        calendar_eval, _ = booking_distance(evaluation, fitted)
        indices = rng.choice(len(train), min(calendar_landmarks, len(train)), replace=False)
        calendar_centers = calendar_train[indices]
        blocks_train.append(radial_features(calendar_train, calendar_centers, 0.65))
        blocks_eval.append(radial_features(calendar_eval, calendar_centers, 0.65))
    x, v = np.column_stack(blocks_train), np.column_stack(blocks_eval)
    np.clip(x, -12, 12, out=x)
    np.clip(v, -12, 12, out=v)
    return x, v


def probabilities(
    train, labels, evaluation, strength=0.5, max_iter=500, positive_weight=1.0, **feature_options,
):
    labels = np.asarray(labels, dtype=float)
    x, v = design_matrices(train, labels, evaluation, **feature_options)
    return fit_logistic(
        x, labels, v, strength, max_iter=max_iter, positive_weight=positive_weight,
    )


def _strong_wolfe_line_search(objective, parameters, loss, gradient, direction):
    # Find a step satisfying Wolfe's strong curvature and decrease conditions

    # Reference: Wolfram Language Documentation, Unconstrained Optimization: Step Control
    # URL: https://reference.wolframcloud.com/language/tutorial/UnconstrainedOptimizationStepControl.html

    c1, c2 = 1e-4, 0.9
    initial_slope = np.dot(gradient, direction)
    if initial_slope >= 0:
        return None

    def zoom(lower, upper, lower_loss):
        for _ in range(60):
            step = 0.5 * (lower + upper)
            candidate = parameters + step * direction
            candidate_loss, candidate_gradient = objective(candidate)
            if (
                not np.isfinite(candidate_loss)
                or not np.isfinite(candidate_gradient).all()
                or candidate_loss > loss + c1 * step * initial_slope
                or candidate_loss >= lower_loss
            ):
                upper = step
                continue

            slope = np.dot(candidate_gradient, direction)
            if abs(slope) <= -c2 * initial_slope:
                return candidate, candidate_loss, candidate_gradient
            if slope * (upper - lower) >= 0:
                upper = lower
            lower, lower_loss = step, candidate_loss
        return None

    previous_step, previous_loss = 0.0, loss
    step = 1.0
    for iteration in range(40):
        candidate = parameters + step * direction
        candidate_loss, candidate_gradient = objective(candidate)
        if (
            not np.isfinite(candidate_loss)
            or not np.isfinite(candidate_gradient).all()
            or candidate_loss > loss + c1 * step * initial_slope
            or (iteration and candidate_loss >= previous_loss)
        ):
            return zoom(previous_step, step, previous_loss)

        slope = np.dot(candidate_gradient, direction)
        if abs(slope) <= -c2 * initial_slope:
            return candidate, candidate_loss, candidate_gradient
        if slope >= 0:
            return zoom(step, previous_step, candidate_loss)
        previous_step, previous_loss = step, candidate_loss
        step = min(2.0 * step, 1e8)
    return None


def lbfgs_minimize(
    objective, initial, *, max_iter=500, memory=10, gradient_tolerance=1e-5,
    function_tolerance=1e-9, armijo=1e-4,
):
    # Minimize a smooth objective with limited-memory BFGS

    # Implements the L-BFGS two-loop recursion and strong-Wolfe line search
    # using NumPy only, with Armijo backtracking as a fallback. There are no box constraints in this logistic objective.
    # Original L-BFGS reference: Liu & Nocedal (1989), Mathematical Programming 45, 503-528
    # Direct PDF mirror: https://people.sc.fsu.edu/~inavon/5420a/liu89limited.pdf
    # Newer L-BFGS discussion: Saratchandran et al. (2023), arXiv:2305.08552, section 3.2
    # HTML: https://arxiv.org/html/2305.08552
    # Armijo and strong-Wolfe conditions: Wolfram Language Documentation, Step Control
    # URL: https://reference.wolframcloud.com/language/tutorial/UnconstrainedOptimizationStepControl.html

    parameters = np.asarray(initial, dtype=float).copy()
    loss, gradient = objective(parameters)
    s_history, y_history, rho_history = [], [], []
    message = "maximum iterations reached"
    success = False

    for iteration in range(1, max_iter + 1):
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            message = "non-finite objective or gradient"
            break
        if np.linalg.norm(gradient, ord=np.inf) <= gradient_tolerance:
            message, success = "gradient tolerance reached", True
            break

        q = gradient.copy()
        alpha_history = []
        for s, y, rho in zip(
            reversed(s_history), reversed(y_history), reversed(rho_history),
        ):
            alpha = rho * np.dot(s, q)
            alpha_history.append(alpha)
            q -= alpha * y

        if s_history:
            sy = np.dot(s_history[-1], y_history[-1])
            yy = np.dot(y_history[-1], y_history[-1])
            initial_scale = sy / yy if yy > 0 else 1.0
        else:
            initial_scale = 1.0
        direction = initial_scale * q
        for (s, y, rho), alpha in zip(
            zip(s_history, y_history, rho_history), reversed(alpha_history),
        ):
            beta = rho * np.dot(y, direction)
            direction += s * (alpha - beta)
        direction = -direction

        slope = np.dot(gradient, direction)
        if not np.isfinite(slope) or slope >= 0:
            direction = -gradient
            slope = -np.dot(gradient, gradient)

        old_loss = loss
        line_search_result = _strong_wolfe_line_search(
            objective, parameters, loss, gradient, direction,
        )
        if line_search_result is None:
            step, accepted = 1.0, False
            for _ in range(60):
                candidate = parameters + step * direction
                candidate_loss, candidate_gradient = objective(candidate)
                if (
                    np.isfinite(candidate_loss)
                    and np.isfinite(candidate_gradient).all()
                    and candidate_loss <= loss + armijo * step * slope
                ):
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                message = "line search could not find a decreasing step"
                break
        else:
            candidate, candidate_loss, candidate_gradient = line_search_result

        s = candidate - parameters
        y = candidate_gradient - gradient
        curvature = np.dot(s, y)
        curvature_floor = 1e-12 * np.linalg.norm(s) * np.linalg.norm(y)
        if curvature > curvature_floor:
            if len(s_history) == memory:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / curvature)

        parameters, loss, gradient = candidate, candidate_loss, candidate_gradient
        if abs(old_loss - loss) <= function_tolerance * max(1.0, abs(old_loss)):
            message, success = "function tolerance reached", True
            break
    else:
        iteration = max_iter

    success = success or np.linalg.norm(gradient, ord=np.inf) <= gradient_tolerance
    return {
        "x": parameters,
        "success": success,
        "message": message,
        "nit": iteration,
        "fun": loss,
        "gradient_inf_norm": float(np.linalg.norm(gradient, ord=np.inf)),
    }


def _weighted_logistic_loss_gradient(
    x, labels, sample_weight, parameters, penalty,
):
    bias, weights = parameters[0], parameters[1:]
    logits = x @ weights + bias
    per_sample_loss = np.logaddexp(0, (1 - 2 * labels) * logits)
    loss = (sample_weight * per_sample_loss).mean()
    loss += 0.5 * penalty * (weights @ weights)
    residual = (np.exp(-np.logaddexp(0, -logits)) - labels) * sample_weight
    feature_gradient = x.T @ residual
    gradient = np.r_[
        residual.mean(), feature_gradient / len(labels) + penalty * weights,
    ]
    return float(loss), gradient


def _logistic_objective(x, labels, strength, positive_weight):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if strength <= 0:
        raise ValueError("strength must be positive")
    if len(x) != len(labels):
        raise ValueError("x and labels must contain the same number of rows")
    sample_weight = np.where(labels == 1, positive_weight, 1.0)
    sample_weight = sample_weight / sample_weight.mean()
    penalty = 1.0 / (strength * len(labels))

    def objective(parameters):
        return _weighted_logistic_loss_gradient(
            x, labels, sample_weight, parameters, penalty,
        )

    return objective, sample_weight, penalty


def gradient_descent_minimize(
    objective, initial, *, max_iter=1000, gradient_tolerance=1e-5,
    function_tolerance=1e-9, armijo=1e-4, initial_step=1.0,
):
    # Minimize an objective with full-batch gradient descent from scratch

    # Gradient descent overview: GeeksforGeeks
    # URL: https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/
    # Gradient descent overview: IBM Think
    # URL: https://www.ibm.com/think/topics/gradient-descent
    # Armijo backtracking reference: Armijo (1966), Pacific Journal of Mathematics 16(1), 1-3
    # Direct PDF: https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-p.pdf

    parameters = np.asarray(initial, dtype=float).copy()
    loss, gradient = objective(parameters)
    step_size = initial_step
    loss_history = [loss]
    message, success = "maximum iterations reached", False

    for iteration in range(1, max_iter + 1):
        gradient_norm = np.linalg.norm(gradient, ord=np.inf)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            message = "non-finite objective or gradient"
            break
        if gradient_norm <= gradient_tolerance:
            message, success = "gradient tolerance reached", True
            break

        direction = -gradient
        slope = -np.dot(gradient, gradient)
        step = step_size
        accepted = False
        for _ in range(60):
            candidate = parameters + step * direction
            candidate_loss, candidate_gradient = objective(candidate)
            if (
                np.isfinite(candidate_loss)
                and np.isfinite(candidate_gradient).all()
                and candidate_loss <= loss + armijo * step * slope
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "Armijo line search could not find a decreasing step"
            break

        old_loss = loss
        parameters, loss, gradient = candidate, candidate_loss, candidate_gradient
        loss_history.append(loss)
        step_size = min(step * 1.25, 1.0)
        if abs(old_loss - loss) <= function_tolerance * max(1.0, abs(old_loss)):
            message, success = "function tolerance reached", True
            break
    else:
        iteration = max_iter

    success = success or np.linalg.norm(gradient, ord=np.inf) <= gradient_tolerance
    return {
        "x": parameters,
        "success": success,
        "message": message,
        "nit": iteration,
        "fun": loss,
        "gradient_inf_norm": float(np.linalg.norm(gradient, ord=np.inf)),
        "loss_history": np.asarray(loss_history),
    }


def minibatch_sgd_minimize(
    x, labels, sample_weight, penalty, initial, *, max_epochs=80,
    batch_size=512, learning_rate=0.05, decay=1e-3, patience=10,
    min_delta=1e-7, random_state=Settings.SEED,
):
    # Minimize the regularized logistic objective with mini-batch SGD

    # Gradient-descent variants overview: GeeksforGeeks
    # URL: https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/
    # Gradient descent overview: IBM Think
    # URL: https://www.ibm.com/think/topics/gradient-descent

    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    parameters = np.asarray(initial, dtype=float).copy()
    rng = np.random.default_rng(random_state)
    best_parameters = parameters.copy()
    best_loss, _ = _weighted_logistic_loss_gradient(
        x, labels, sample_weight, parameters, penalty,
    )
    loss_history = [best_loss]
    stale_epochs, updates = 0, 0
    message, success = "maximum epochs reached", False

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            _, gradient = _weighted_logistic_loss_gradient(
                x[rows], labels[rows], sample_weight[rows], parameters, penalty,
            )
            step = learning_rate / np.sqrt(1.0 + decay * updates)
            parameters -= step * gradient
            updates += 1

        loss, _ = _weighted_logistic_loss_gradient(
            x, labels, sample_weight, parameters, penalty,
        )
        loss_history.append(loss)
        required_improvement = min_delta * max(1.0, abs(best_loss))
        if np.isfinite(loss) and best_loss - loss > required_improvement:
            best_loss = loss
            best_parameters = parameters.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            message, success = "objective plateau reached", True
            break

    final_loss, final_gradient = _weighted_logistic_loss_gradient(
        x, labels, sample_weight, best_parameters, penalty,
    )
    return {
        "x": best_parameters,
        "success": success,
        "message": message,
        "nit": epoch,
        "updates": updates,
        "fun": final_loss,
        "gradient_inf_norm": float(np.linalg.norm(final_gradient, ord=np.inf)),
        "loss_history": np.asarray(loss_history),
    }


def fit_logistic_with_lbfgs(
    x, labels, v, strength, max_iter=500, positive_weight=1.0,
):
    objective, _, _ = _logistic_objective(
        x, labels, strength, positive_weight,
    )
    result = lbfgs_minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        max_iter=max_iter,
        memory=10,
        gradient_tolerance=1e-5,
    )
    scores = v @ result["x"][1:] + result["x"][0]
    return np.exp(-np.logaddexp(0, -scores)), result


def fit_logistic_with_gradient_descent(
    x, labels, v, strength, max_iter=1000, positive_weight=1.0,
):
    objective, _, _ = _logistic_objective(
        x, labels, strength, positive_weight,
    )
    result = gradient_descent_minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        max_iter=max_iter,
        gradient_tolerance=1e-5,
    )
    scores = v @ result["x"][1:] + result["x"][0]
    return np.exp(-np.logaddexp(0, -scores)), result


def fit_logistic_with_sgd(
    x, labels, v, strength, max_epochs=80, positive_weight=1.0,
    batch_size=512, learning_rate=0.005, random_state=Settings.SEED,
):
    objective, sample_weight, penalty = _logistic_objective(
        x, labels, strength, positive_weight,
    )
    result = minibatch_sgd_minimize(
        x, labels, sample_weight, penalty,
        np.zeros(x.shape[1] + 1),
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    scores = v @ result["x"][1:] + result["x"][0]
    return np.exp(-np.logaddexp(0, -scores)), result


def fit_logistic(x, labels, v, strength, max_iter=500, positive_weight=1.0):
    scores, result = fit_logistic_with_lbfgs(
        x, labels, v, strength, max_iter=max_iter,
        positive_weight=positive_weight,
    )
    if not result["success"]:
        print(f"Optimizer status: {result['message']}", flush=True)
    return scores


def run(
    *, output_name=OUTPUT_NAME, seeds=(42, 137), threshold=THRESHOLD,
    strength=0.5, optimizer_iterations=500, **feature_options,
):
    train, test = load_data()
    x = train.drop(columns=[Settings.ID, Settings.TARGET])
    y = train[Settings.TARGET]
    for seed in seeds:
        a, b, labels, expected = train_test_split(x, y, stratify=y, random_state=seed, test_size=0.2)
        scores = probabilities(a, labels, b, strength=strength, max_iter=optimizer_iterations, **feature_options)
        pred = scores >= threshold
        print(f"seed={seed} threshold={threshold:.2f} macro-F1={f1_score(expected, pred, average='macro'):.5f} "
              f"accuracy={accuracy_score(expected, pred):.5f}", flush=True)
    predicted = probabilities(
        x, y, test.drop(columns=Settings.ID), strength=strength,
        max_iter=optimizer_iterations, **feature_options,
    ) >= threshold
    output = pd.DataFrame({Settings.ID: test[Settings.ID], Settings.TARGET: predicted.astype(int)})
    Settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Settings.OUTPUT_DIR / output_name
    output.to_csv(path, index=False)
    print(f"{len(output)} predictions saved to {path}", flush=True)
    return path


def run_calendar_submission(*, seeds=(42, 731)):
    return run(
        output_name="lr_calendar_windows.csv",
        seeds=seeds,
        strength=0.2,
        context=True,
        windows=True,
    )


def run_density_submission(
    *, seeds=(), output_name="lr_calendar_density_c010_lbfgs_scratch.csv",
):
    return run(
        output_name=output_name,
        seeds=seeds,
        threshold=0.42,
        strength=0.1,
        optimizer_iterations=3000,
        target_smoothing=5.0,
        context=True,
        windows=True,
        density=True,
    )


def run_three_solver_submissions(
    *, validation_seed=Settings.SEED, validation_size=0.2, strength=0.1,
    threshold=THRESHOLD, gd_iterations=1500, sgd_epochs=120,
    lbfgs_iterations=3000, batch_size=512, sgd_learning_rate=0.005,
    feature_options=None,
):
    # Compare batch GD, mini-batch SGD, and scratch L-BFGS on shared features

    train, test = load_data()
    x_all = train.drop(columns=[Settings.ID, Settings.TARGET])
    y_all = train[Settings.TARGET].to_numpy(dtype=float)
    x_test_raw = test.drop(columns=Settings.ID)
    if feature_options is None:
        feature_options = {
            "target_smoothing": 5.0,
            "context": True,
            "windows": True,
            "density": True,
        }

    x_fit_raw, x_val_raw, y_fit, y_val = train_test_split(
        x_all, y_all, test_size=validation_size, random_state=validation_seed,
        stratify=y_all,
    )
    x_fit, x_val = design_matrices(
        x_fit_raw, y_fit, x_val_raw, **feature_options,
    )
    x_val_labels = np.asarray(y_val, dtype=int)

    solver_specs = (
        ("batch_gradient_descent", "lr_c010_batch_gd.csv"),
        ("mini_batch_sgd", "lr_c010_minibatch_sgd.csv"),
        ("lbfgs_scratch", "lr_calendar_density_c010_lbfgs_scratch.csv"),
    )
    solver_settings = {
        "batch_gradient_descent": (
            f"max_iter={gd_iterations}; initial_step=1.0; armijo=1e-4; "
            "gradient_tolerance=1e-5; function_tolerance=1e-9"
        ),
        "mini_batch_sgd": (
            f"max_epochs={sgd_epochs}; batch_size={batch_size}; "
            f"learning_rate={sgd_learning_rate}; decay=1e-3; patience=10; "
            f"seed={validation_seed}"
        ),
        "lbfgs_scratch": (
            f"max_iter={lbfgs_iterations}; memory=10; "
            "strong_wolfe=(1e-4,0.9); armijo_fallback=1e-4; "
            "gradient_tolerance=1e-5; function_tolerance=1e-9"
        ),
    }
    fit_functions = {
        "batch_gradient_descent": lambda a, b, c: fit_logistic_with_gradient_descent(
            a, b, c, strength=strength, max_iter=gd_iterations,
        ),
        "mini_batch_sgd": lambda a, b, c: fit_logistic_with_sgd(
            a, b, c, strength=strength, max_epochs=sgd_epochs,
            batch_size=batch_size, learning_rate=sgd_learning_rate,
            random_state=validation_seed,
        ),
        "lbfgs_scratch": lambda a, b, c: fit_logistic_with_lbfgs(
            a, b, c, strength=strength, max_iter=lbfgs_iterations,
        ),
    }

    metrics = []
    for name, _ in solver_specs:
        started = perf_counter()
        scores, result = fit_functions[name](x_fit, y_fit, x_val)
        elapsed = perf_counter() - started
        predictions = (scores >= threshold).astype(int)
        row = {
            "model": name,
            "validation_seed": validation_seed,
            "validation_size": validation_size,
            "C": strength,
            "threshold": threshold,
            "feature_options": ";".join(
                f"{key}={value}" for key, value in sorted(feature_options.items())
            ),
            "solver_settings": solver_settings[name],
            "validation_macro_f1": f1_score(
                x_val_labels, predictions, average="macro",
            ),
            "validation_accuracy": accuracy_score(x_val_labels, predictions),
            "validation_log_loss": log_loss(
                x_val_labels, scores, labels=[0, 1],
            ),
            "training_objective": result["fun"],
            "iterations": result["nit"],
            "converged": result["success"],
            "optimizer_status": result["message"],
            "gradient_inf_norm": result["gradient_inf_norm"],
            "validation_fit_seconds": elapsed,
        }
        if "updates" in result:
            row["updates"] = result["updates"]
        metrics.append(row)
        print(
            f"{name}: macro-F1={row['validation_macro_f1']:.5f}, "
            f"accuracy={row['validation_accuracy']:.5f}, "
            f"log-loss={row['validation_log_loss']:.5f}, "
            f"iterations={row['iterations']}, converged={row['converged']}",
            flush=True,
        )

    x_full, x_test = design_matrices(
        x_all, y_all, x_test_raw, **feature_options,
    )
    Settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, filename in solver_specs:
        started = perf_counter()
        scores, result = fit_functions[name](x_full, y_all, x_test)
        predicted = (scores >= threshold).astype(int)
        output = pd.DataFrame({
            Settings.ID: test[Settings.ID],
            Settings.TARGET: predicted,
        })
        path = Settings.OUTPUT_DIR / filename
        output.to_csv(path, index=False)
        output_paths[name] = path
        metrics_row = next(row for row in metrics if row["model"] == name)
        metrics_row["full_fit_seconds"] = perf_counter() - started
        metrics_row["full_train_objective"] = result["fun"]
        metrics_row["full_train_iterations"] = result["nit"]
        print(f"{len(output)} predictions saved to {path}", flush=True)

    comparison = pd.DataFrame(metrics).sort_values(
        "validation_macro_f1", ascending=False,
    )
    comparison_path = Settings.OUTPUT_DIR / "lr_c010_solver_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    output_paths["comparison"] = comparison_path
    print(f"Validation comparison saved to {comparison_path}", flush=True)
    return comparison, output_paths


if __name__ == "__main__":
    run()
