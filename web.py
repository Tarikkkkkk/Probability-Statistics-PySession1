import matplotlib.pyplot as plt
import streamlit as st

from practice import (
    posterior,
    question1_population_comparison,
    question1_parameter_curves,
    question1_repeated_tests,
    question2_playlist_experiment,
    question2_size_comparison,
    question3_registration,
    poisson_pmf,
)


st.set_page_config(
    page_title="BS202 — Guided Practice 1",
    page_icon="📊",
    layout="wide",
)


def show_figure(fig):
    """Render a Matplotlib figure and release its server-side resources."""
    st.pyplot(fig, clear_figure=False)
    plt.close(fig)


st.sidebar.title("BS202 — Guided Practice 1")
selected_question = st.sidebar.radio(
    "Choose a question",
    ["Question 1", "Question 2", "Question 3"],
    key="selected_question",
)
st.sidebar.caption("Only the selected question is computed on each rerun.")

st.title("BS202 — Introduction to Probability & Mathematical Statistics")
st.caption(f"Guided Practice 1 · {selected_question}")


def render_question_1():
    with st.sidebar:
        st.header("Question 1 parameters")

        population = st.number_input(
            "Population",
            min_value=100,
            max_value=1_000_000,
            value=10_000,
            step=100,
            key="q1_population",
        )

        prevalence = st.slider(
            "Prevalence  P(I)",
            min_value=0.001,
            max_value=0.100,
            value=0.010,
            step=0.001,
            format="%.3f",
            key="q1_prevalence",
        )

        sensitivity = st.slider(
            "Sensitivity  P(+ | I)",
            min_value=0.00,
            max_value=1.00,
            value=0.80,
            step=0.01,
            key="q1_sensitivity",
        )

        specificity = st.slider(
            "Specificity  P(− | Iᶜ)",
            min_value=0.00,
            max_value=1.00,
            value=0.95,
            step=0.01,
            key="q1_specificity",
        )

        step = st.slider(
            "Increase used in Cells 1–2",
            min_value=0.01,
            max_value=0.20,
            value=0.04,
            step=0.01,
            key="q1_step",
        )

    st.header("Question 1 — Diagnostic testing")

    base_posterior = posterior(
        prevalence=prevalence,
        sensitivity=sensitivity,
        specificity=specificity,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("P(I)", f"{prevalence:.3f}")
    metric_col2.metric("Sensitivity", f"{sensitivity:.2f}")
    metric_col3.metric("Specificity", f"{specificity:.2f}")
    metric_col4.metric("P(I | +)", f"{base_posterior:.4f}")

    tab0, tab1, tab2, tab3 = st.tabs(
        [
            "Cell 0 · Question",
            "Cell 1 · Single positive test",
            "Cell 2 · Parameter sensitivity",
            "Cell 3 · Repeated tests",
        ]
    )

    with tab0:
        st.subheader("Question 1")
        st.markdown(
            r"""
A disease has prevalence $P(I)=0.01$, and a test for this disease has sensitivity
$P(+\mid I)=0.80$ and specificity $P(-\mid I^c)=0.95$, where $I$ is the event that a randomly
chosen person is infected and $+$ the event that the test is positive.

**(a)** Find the probability that a person who tests positive is infected, $P(I\mid +)$.

**(b)** Find $P(I\mid +)$ when we increase

(i) the prevalence to 0.05;  
(ii) the sensitivity to 0.84;  
(iii) the specificity to 0.99.

Assume the other two values are kept as in (a). **[cell 1]**

**(c)** Find the maximum $P(I\mid +)$ when only the sensitivity can change. **[cell 2]**

**(d)** Find $P(I\mid +,+)$ for two tests independent given infection status. **[cell 3]**
"""
        )

    with tab1:
        st.subheader("Cell 1 — Single positive test")
        st.markdown(
            """
            Explore how the posterior probability **P(I | +)** changes when prevalence,
            sensitivity, or specificity is increased while the other parameters remain fixed.
            """
        )

        result_1 = question1_population_comparison(
            population=population,
            prevalence=prevalence,
            sensitivity=sensitivity,
            specificity=specificity,
            step=step,
        )
        show_figure(result_1["figure"])

        st.markdown("#### Numerical results")
        st.dataframe(
            result_1["rows"],
            hide_index=True,
            width="stretch",
            column_config={
                "P(I)": st.column_config.NumberColumn("P(I)", format="%.4f"),
                "sensitivity": st.column_config.NumberColumn("Sensitivity", format="%.4f"),
                "specificity": st.column_config.NumberColumn("Specificity", format="%.4f"),
                "P(I|+) counts": st.column_config.NumberColumn(
                    "P(I | +), from counts", format="%.4f"
                ),
                "exact": st.column_config.NumberColumn("P(I | +), exact", format="%.4f"),
            },
        )

    with tab2:
        st.subheader("Cell 2 — How the inputs affect P(I | +)")
        st.markdown(
            """
            Each curve changes **one** input while holding the other two at the current
            base values selected in the sidebar.
            """
        )

        result_2 = question1_parameter_curves(
            prevalence=prevalence,
            sensitivity=sensitivity,
            specificity=specificity,
            step=step,
        )
        show_figure(result_2)

    with tab3:
        st.subheader("Cell 3 — Repeated independent tests")
        st.markdown(
            """
            Compare the posterior probability after one positive result with the posterior
            after repeated test-result sequences such as **++**, **+-**, and **+++**.
            """
        )

        result_3 = question1_repeated_tests(
            population=population,
            prevalence=prevalence,
            sensitivity=sensitivity,
            specificity=specificity,
        )
        show_figure(result_3["figure"])

        st.markdown("#### Repeated-test probabilities")
        st.dataframe(
            result_3["rows"],
            hide_index=True,
            width="stretch",
            column_config={
                "P(results | I)": st.column_config.NumberColumn(
                    "P(results | I)", format="%.6f"
                ),
                "P(results | not I)": st.column_config.NumberColumn(
                    "P(results | Iᶜ)", format="%.6f"
                ),
                "from the counts": st.column_config.NumberColumn(
                    "From counts", format="%.4f"
                ),
                "exact": st.column_config.NumberColumn("Exact", format="%.4f"),
            },
        )
        st.warning(
            "For some repeated-test cases the rounded finite-population "
            "count can differ slightly from the exact Bayes value."
        )


def render_question_2():
    with st.sidebar:
        st.divider()
        st.header("Question 2 parameters")

        songs = st.slider(
            "Playlist size  N",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
            key="q2_songs",
        )

        by_artist = st.slider(
            "Songs by the artist  K",
            min_value=1,
            max_value=songs,
            value=min(4, songs),
            step=1,
            key="q2_by_artist",
        )

        played = st.slider(
            "Songs counted for X  n",
            min_value=1,
            max_value=songs,
            value=min(10, songs),
            step=1,
            key="q2_played",
        )

        target = st.slider(
            "Target artist song for Y  r",
            min_value=1,
            max_value=by_artist,
            value=min(2, by_artist),
            step=1,
            key="q2_target",
        )

        runs = st.select_slider(
            "Simulation runs",
            options=[10, 100, 10_000],
            value=10_000,
            key="q2_runs",
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            value=2029,
            step=1,
            key="q2_seed",
        )

        st.markdown("##### Cell 5")
        sizes_text = st.text_input(
            "Playlist sizes",
            value="20, 40, 200, 2000",
            help="Comma-separated playlist sizes for the Cell 5 comparison.",
            key="q2_sizes_text",
        )

        fraction_by_artist = st.slider(
            "Artist fraction  K/N",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.05,
            key="q2_fraction_by_artist",
        )

    st.divider()
    st.header("Question 2 — Playlist experiment")

    q2_tab0, q2_tab1, q2_tab2 = st.tabs(
        [
            "Cell 0 · Question",
            "Cell 4 · Player A vs Player B",
            "Cell 5 · Increasing playlist size",
        ]
    )

    with q2_tab0:
        st.subheader("Question 2")
        st.markdown(
            r"""
A playlist contains $N=20$ songs, $K=4$ by one artist. Player A selects each song
independently at random from the whole playlist; songs can repeat. Player B plays the
songs in a random order; each song only once. Let $X$ count the number of songs by the
artist among the first 10 songs played, and $Y$ the number of songs played up to and
including the second song by the artist. Answer for both players.

**(a)** Calculate $E[X]$, $\mathrm{Var}(X)$ and $E[Y]$.

**(b)** Find $P(X\geq2)$ and $P(Y\leq10)$. **[cell 4]**

**(c)** Identify the runs with $X\geq2$ and those with $Y\leq10$. **[cell 4]**

**(d)** Compare $P(X=0)$, $P(X\geq4)$ and $P(Y>15)$. Explain in one sentence. **[cell 4]**

**(e)** Compare the distributions of $X$ for $N=40,200,2000$, with $K/N=0.2$. **[cell 5]**
"""
        )

    with q2_tab1:
        st.subheader("Cell 4 — Sampling with and without replacement")

        p_artist = by_artist / songs
        q2m1, q2m2, q2m3, q2m4 = st.columns(4)
        q2m1.metric("N", f"{songs}")
        q2m2.metric("K", f"{by_artist}")
        q2m3.metric("K / N", f"{p_artist:.3f}")
        q2m4.metric("Runs", f"{runs:,}")

        result_4 = question2_playlist_experiment(
            songs=songs,
            by_artist=by_artist,
            played=played,
            target=target,
            runs=runs,
            seed=int(seed),
        )

        st.markdown("#### Example simulated runs")
        show_figure(result_4["strip_figure"])

        st.markdown("#### Simulated distributions and exact PMFs")
        show_figure(result_4["distribution_figure"])

        st.markdown("#### Numerical comparison")
        q2_column_config = {
            "mean X": st.column_config.NumberColumn("Mean X", format="%.3f"),
            "var X": st.column_config.NumberColumn("Var(X)", format="%.3f"),
            "mean Y": st.column_config.NumberColumn("Mean Y", format="%.3f"),
        }
        for event_name in result_4["event_headers"]:
            q2_column_config[event_name] = st.column_config.NumberColumn(
                event_name, format="%.4f"
            )

        st.dataframe(
            result_4["summary_rows"],
            hide_index=True,
            width="stretch",
            column_config=q2_column_config,
        )

        st.markdown("#### How Player B's next-song probability changes")
        next_probability_rows = [
            {
                "song": i + 1,
                "run 1 outcome": (
                    "artist" if bool(result_4["player_b_first_run"][i]) else "other"
                ),
                "P(next song is by artist)": result_4["player_b_next_probabilities"][i],
            }
            for i in range(len(result_4["player_b_first_run"]))
        ]

        st.dataframe(
            next_probability_rows,
            hide_index=True,
            width="stretch",
            column_config={
                "P(next song is by artist)": st.column_config.NumberColumn(
                    "P(next song is by artist)", format="%.3f"
                )
            },
        )
        st.caption(
            f"For Player A this probability remains constant at "
            f"K/N = {result_4['player_a_next_probability']:.3f} before every draw."
        )

    with q2_tab2:
        st.subheader("Cell 5 — What happens as the playlist becomes larger?")
        st.markdown(
            """
            The fraction of songs by the artist is held approximately fixed while the
            playlist size increases. Compare the distribution of **X** for Player A
            (with replacement) and Player B (without replacement).
            """
        )

        try:
            playlist_sizes = tuple(
                int(value.strip())
                for value in sizes_text.replace(";", ",").split(",")
                if value.strip()
            )

            if not playlist_sizes:
                raise ValueError("Enter at least one playlist size.")
            if any(N < played for N in playlist_sizes):
                raise ValueError(f"Every playlist size must be at least n = {played}.")
            if any(N < 2 for N in playlist_sizes):
                raise ValueError("Playlist sizes must be at least 2.")

            result_5 = question2_size_comparison(
                lengths=playlist_sizes,
                played=played,
                fraction_by_artist=fraction_by_artist,
                seed=int(seed),
            )

            show_figure(result_5["figure"])
            st.markdown("#### Variance comparison")
            st.dataframe(
                result_5["rows"],
                hide_index=True,
                width="stretch",
                column_config={
                    "var X, player A": st.column_config.NumberColumn(
                        "Var(X), Player A simulated", format="%.3f"
                    ),
                    "exact B": st.column_config.NumberColumn(
                        "Var(X), Player A exact", format="%.3f"
                    ),
                    "var X, player B": st.column_config.NumberColumn(
                        "Var(X), Player B simulated", format="%.3f"
                    ),
                    "exact HG": st.column_config.NumberColumn(
                        "Var(X), Player B exact", format="%.3f"
                    ),
                },
            )
        except ValueError as exc:
            st.error(str(exc))


def render_question_3():
    with st.sidebar:
        st.divider()
        st.header("Question 3 parameters")

        rate = st.slider(
            "Request rate  λ (per second)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="q3_rate",
        )

        seats = st.number_input(
            "Number of seats",
            min_value=1,
            max_value=500,
            value=60,
            step=1,
            key="q3_seats",
        )

        time_cutoff = st.slider(
            "Time cutoff  t (seconds)",
            min_value=0.0,
            max_value=120.0,
            value=30.0,
            step=1.0,
            key="q3_time_cutoff",
        )

        q3_seed = st.number_input(
            "Question 3 random seed",
            min_value=0,
            value=2029,
            step=1,
            key="q3_seed",
        )

    st.divider()
    st.header("Question 3 — Course registration")

    q3m1, q3m2, q3m3, q3m4 = st.columns(4)
    q3m1.metric("Rate λ", f"{rate:.1f} / s")
    q3m2.metric("Seats", f"{int(seats)}")
    q3m3.metric("P(R(1) = 0)", f"{poisson_pmf(0, rate):.4f}")
    q3m4.metric(f"E[T({int(seats)})]", f"{int(seats) / rate:.2f} s")

    result_6 = question3_registration(
        rate=rate,
        seats=int(seats),
        time_cutoff=time_cutoff,
        seed=int(q3_seed),
    )

    q3_tab0, q3_tab1, q3_tab2, q3_tab3 = st.tabs(
        [
            "Cell 0 · Question",
            "Simulated openings",
            "Distribution and seat-open probability",
            "Numerical results",
        ]
    )

    with q3_tab0:
        st.subheader("Question 3")
        st.markdown(
            r"""
A course has 60 seats. After registration opens, requests arrive following a Poisson
model at rate 2 per second. Let $R(t)$ be the number of requests in the first $t$ seconds
and $T(r)$ be the time of the $r$-th request. The course fills at $T(60)$.

**(a)** Find $P(R(1)=0)$, $E[T(1)]$ and $E[T(60)]$.

**(b)** Express “a seat is still open at time $t$” using $R(t)$, and using $T(60)$.

**(c)** Find the mean and standard deviation of $T(60)$ from 10 000 simulated openings.
Compare with (a). **[cell 6]**

**(d)** Find the probability that a seat is still open at 20, 30 and 40 seconds. **[cell 6]**
"""
        )

    with q3_tab1:
        st.subheader("Example simulated registration openings")
        st.markdown(
            f"""
Each row represents one simulated course opening.

- Each dot is a registration request.
- The red star is request **{int(seats)}**, when the course becomes full.
- The dashed vertical line marks **t = {time_cutoff:g} seconds**.
"""
        )
        show_figure(result_6["timeline_figure"])

    with q3_tab2:
        st.subheader(f"Distribution of T({int(seats)}) and seat availability")
        st.markdown(
            """
            The left panel shows the simulated time until the course becomes full.
            The right panel compares the simulated fraction of openings with a seat
            still available to the corresponding exact Poisson probability.
            """
        )
        show_figure(result_6["summary_figure"])

    with q3_tab3:
        st.subheader("Simulated vs exact results")
        st.dataframe(
            result_6["rows"],
            hide_index=True,
            width="stretch",
            column_config={
                "simulated": st.column_config.NumberColumn("Simulated", format="%.4f"),
                "exact": st.column_config.NumberColumn("Exact", format="%.4f"),
            },
        )

        theoretical_col1, theoretical_col2 = st.columns(2)
        theoretical_col1.metric("E[T(1)]", f"{1 / rate:.3f} s")
        theoretical_col2.metric(
            f"E[T({int(seats)})]", f"{int(seats) / rate:.3f} s"
        )


# Streamlit reruns the script after a widget change, but only the selected
# question function is executed. Expensive simulations from other questions are skipped.
if selected_question == "Question 1":
    render_question_1()
elif selected_question == "Question 2":
    render_question_2()
else:
    render_question_3()

st.divider()
st.caption("BS202 · Practice 1 · Questions 1–3")