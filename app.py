import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

st.set_page_config(
    page_title="Hotel Booking Probability Dashboard",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Probability Analysis")
st.subheader("Application of Binomial and Poisson Distributions in Hotel Booking Analysis")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("hotel_bookings_cleaned.csv")

df = load_data()

# -----------------------------
# Basic calculations
# -----------------------------
total_bookings = len(df)

if "is_canceled" in df.columns:
    cancelled = int(df["is_canceled"].sum())
    not_cancelled = total_bookings - cancelled
    p = cancelled / total_bookings
else:
    cancelled = 0
    not_cancelled = total_bookings
    p = 0.0

q = 1 - p

# Daily bookings
if "arrival_date" in df.columns:
    daily_bookings = df.groupby("arrival_date").size()
    lam = float(daily_bookings.mean())
else:
    daily_bookings = pd.Series(dtype=float)
    lam = 0.0

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Dashboard Controls")

n = st.sidebar.slider(
    "Binomial sample size (n)",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

selected_x = st.sidebar.slider(
    "Number of cancellations (X)",
    min_value=0,
    max_value=n,
    value=min(5, n),
    step=1
)

# Binomial calculations
binom_mean = n * p
binom_variance = n * p * q

def binomial_probability(n, x, p):
    return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

selected_prob = binomial_probability(n, selected_x, p)

# -----------------------------
# KPI cards
# -----------------------------
st.markdown("### 📊 Key Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Bookings", f"{total_bookings:,}")
c2.metric("Cancelled Bookings", f"{cancelled:,}")
c3.metric("Cancellation Probability", f"{p:.4f}")
c4.metric("Daily Poisson λ", f"{lam:.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Binomial Mean", f"{binom_mean:.4f}")
c6.metric("Binomial Variance", f"{binom_variance:.4f}")
c7.metric(f"P(X={selected_x})", f"{selected_prob:.6f}")
c8.metric("Poisson Variance", f"{lam:.4f}")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Binomial Distribution",
    "Poisson Distribution",
    "MGF",
    "Binomial vs Poisson"
])

# -----------------------------
# Overview
# -----------------------------
with tab1:
    st.header("Overview")

    st.write(
        "This dashboard applies discrete probability distributions to hotel "
        "booking data. Booking cancellations are studied using the Binomial "
        "Distribution, while daily booking counts are studied using the "
        "Poisson Distribution."
    )

    overview = pd.DataFrame({
        "Measure": [
            "Total bookings",
            "Cancelled bookings",
            "Not cancelled bookings",
            "Cancellation probability (p)",
            "No-cancellation probability (q)",
            "Average daily bookings (λ)"
        ],
        "Value": [
            total_bookings,
            cancelled,
            not_cancelled,
            round(p, 6),
            round(q, 6),
            round(lam, 6)
        ]
    })

    st.dataframe(overview, use_container_width=True, hide_index=True)

    if total_bookings > 0:
        fig, ax = plt.subplots()
        ax.bar(
            ["Cancelled", "Not Cancelled"],
            [cancelled, not_cancelled]
        )
        ax.set_title("Hotel Booking Cancellation Status")
        ax.set_ylabel("Number of Bookings")
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------
# Binomial
# -----------------------------
with tab2:
    st.header("Binomial Distribution")

    st.latex(
        r"P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}"
    )

    st.write(f"Cancellation probability p = {p:.6f}")
    st.write(f"No-cancellation probability q = {q:.6f}")
    st.write(f"Current n = {n}")

    x_values = np.arange(0, n + 1)
    probabilities = np.array([
        binomial_probability(n, int(x), p)
        for x in x_values
    ])

    binom_table = pd.DataFrame({
        "X": x_values,
        "P(X)": probabilities
    })

    st.dataframe(
        binom_table.style.format({"P(X)": "{:.8f}"}),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots()
    ax.bar(x_values, probabilities)
    ax.set_title(f"Binomial Distribution (n={n}, p={p:.4f})")
    ax.set_xlabel("Number of Cancellations (X)")
    ax.set_ylabel("Probability")
    st.pyplot(fig)
    plt.close(fig)

    st.success(
        f"P(X={selected_x}) = {selected_prob:.8f}"
    )

    st.write("### Mean and Variance")
    st.latex(r"\mu=np")
    st.latex(r"\sigma^2=npq")
    st.write(f"Mean = {binom_mean:.6f}")
    st.write(f"Variance = {binom_variance:.6f}")

# -----------------------------
# Poisson
# -----------------------------
with tab3:
    st.header("Poisson Distribution")

    st.latex(
        r"P(X=x)=\frac{e^{-\lambda}\lambda^x}{x!}"
    )

    st.write(f"Average daily bookings (λ) = {lam:.6f}")
    st.write(f"Poisson mean = {lam:.6f}")
    st.write(f"Poisson variance = {lam:.6f}")

    if len(daily_bookings) > 0:
        st.write("### Daily Booking Counts")
        daily_df = daily_bookings.reset_index()
        daily_df.columns = ["arrival_date", "bookings"]
        st.dataframe(
            daily_df.head(50),
            use_container_width=True,
            hide_index=True
        )

        # Show a manageable range around lambda
        center = int(round(lam))
        lower = max(0, center - 20)
        upper = center + 20
        poisson_x = np.arange(lower, upper + 1)

        poisson_probs = np.array([
            math.exp(-lam) * (lam ** int(x)) / math.factorial(int(x))
            if x < 171 else 0
            for x in poisson_x
        ])

        # Normalize only for numerical display if needed
        if poisson_probs.sum() > 0:
            poisson_probs = poisson_probs / poisson_probs.sum()

        poisson_table = pd.DataFrame({
            "X": poisson_x,
            "P(X)": poisson_probs
        })

        st.write("### Poisson probabilities around the average")
        st.dataframe(
            poisson_table.style.format({"P(X)": "{:.8f}"}),
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots()
        ax.bar(poisson_x, poisson_probs)
        ax.set_title("Poisson Distribution Around Average Daily Bookings")
        ax.set_xlabel("Daily Bookings (X)")
        ax.set_ylabel("Probability")
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------
# MGF
# -----------------------------
with tab4:
    st.header("Moment Generating Function (MGF)")

    st.write("For a Binomial random variable:")
    st.latex(
        r"M_X(t)=(q+pe^t)^n"
    )

    st.write(
        f"Using p = {p:.6f}, q = {q:.6f}, and n = {n}:"
    )

    t_values = np.linspace(-2, 2, 200)
    mgf_values = (q + p * np.exp(t_values)) ** n

    fig, ax = plt.subplots()
    ax.plot(t_values, mgf_values)
    ax.set_title("Binomial Moment Generating Function")
    ax.set_xlabel("t")
    ax.set_ylabel("M_X(t)")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "The MGF is useful for obtaining moments such as the mean and variance "
        "of a probability distribution."
    )

# -----------------------------
# Comparison
# -----------------------------
with tab5:
    st.header("Binomial vs Poisson")

    comp_n = st.number_input(
        "Comparison n",
        min_value=1,
        max_value=10000,
        value=100,
        step=1
    )

    comp_p = st.number_input(
        "Comparison p",
        min_value=0.0001,
        max_value=0.9999,
        value=0.05,
        step=0.01,
        format="%.4f"
    )

    comp_lambda = comp_n * comp_p

    st.write(f"λ = np = {comp_lambda:.4f}")

    x_max = min(25, comp_n)
    xs = np.arange(0, x_max + 1)

    bin_probs = np.array([
        binomial_probability(comp_n, int(x), comp_p)
        for x in xs
    ])

    pois_probs = np.array([
        math.exp(-comp_lambda) * (comp_lambda ** int(x)) / math.factorial(int(x))
        for x in xs
    ])

    comparison = pd.DataFrame({
        "X": xs,
        "Binomial P(X)": bin_probs,
        "Poisson P(X)": pois_probs,
        "Difference": bin_probs - pois_probs
    })

    st.dataframe(
        comparison.style.format({
            "Binomial P(X)": "{:.8f}",
            "Poisson P(X)": "{:.8f}",
            "Difference": "{:.8f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots()
    ax.plot(xs, bin_probs, marker="o", label="Binomial")
    ax.plot(xs, pois_probs, marker="x", label="Poisson")
    ax.set_title(
        f"Binomial vs Poisson (n={comp_n}, p={comp_p:.2f}, λ={comp_lambda:.2f})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Poisson can approximate a Binomial distribution when n is large "
        "and p is relatively small, with λ = np."
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Group 4 | Maths Module IV | Discrete Probability Distributions")
